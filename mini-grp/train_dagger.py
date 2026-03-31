"""
HW3 Part 3 (Optional): DAgger — distill a dense teacher into a transformer student,
then fine-tune the student with RL.

Usage:
    python hw3/train_dagger.py \
        experiment.name=hw3_dagger_seed0 \
        r_seed=0 \
        teacher_checkpoint=/path/to/dense_policy.pth \
        student_init_checkpoint=/path/to/hw1/miniGRP.pth \
        dagger.num_rounds=10 \
        dagger.rollouts_per_round=20
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../mini-grp'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import wandb

from hw3.libero_env_fast import FastLIBEROEnv
from hw3.train_dense_rl import DensePolicy
from hw3.train_transformer_rl import TransformerPolicyWrapper, evaluate_policy


# ---------------------------------------------------------------------------
# DAgger dataset
# ---------------------------------------------------------------------------

class DAggerDataset(torch.utils.data.Dataset):
    """
    Aggregated dataset for DAgger.
    Each entry is an (obs, teacher_action) pair collected across all rounds.
    """

    def __init__(self):
        self.obs_list = []
        self.action_list = []

    def add_rollout(self, obs_seq: list, actions_seq: list):
        """Append a rollout's (obs, teacher_action) pairs to the dataset."""
        self.obs_list.extend(obs_seq)
        self.action_list.extend(actions_seq)

    def __len__(self):
        return len(self.obs_list)

    def __getitem__(self, idx):
        obs = torch.tensor(self.obs_list[idx], dtype=torch.float32)
        action = torch.tensor(self.action_list[idx], dtype=torch.float32)
        return obs, action

    def save(self, path: str):
        torch.save({"obs": self.obs_list, "actions": self.action_list}, path)

    def load(self, path: str):
        data = torch.load(path)
        self.obs_list = data["obs"]
        self.action_list = data["actions"]


# ---------------------------------------------------------------------------
# Teacher wrapper
# ---------------------------------------------------------------------------

class DensePolicyTeacher:
    """
    Wraps the trained DensePolicy checkpoint for action labeling.
    """

    def __init__(self, checkpoint_path: str, obs_dim: int, action_dim: int,
                 hidden_dim: int, n_layers: int, device: torch.device):
        ckpt = torch.load(checkpoint_path, map_location=device)
        self.policy = DensePolicy(obs_dim, action_dim, hidden_dim, n_layers).to(device)
        self.policy.load_state_dict(ckpt["policy"])
        self.policy.eval()
        self.device = device

    def get_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Query teacher for a deterministic action label."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            # get_action returns (action_np, log_prob, entropy, pre_tanh) — unpack all 4.
            action, _, _, _ = self.policy.get_action(obs_t, deterministic=deterministic)
        return action


# ---------------------------------------------------------------------------
# DAgger rollout collection
# ---------------------------------------------------------------------------

def collect_dagger_rollout(env: FastLIBEROEnv,
                            student: TransformerPolicyWrapper,
                            teacher: DensePolicyTeacher,
                            beta: float,
                            max_steps: int):
    """
    Collect one DAgger rollout.

    At each step:
      - With probability beta, follow the teacher; else follow the student.
      - Always label the visited state with the teacher's action.

    The env is configured with ``fast_env_output_image=True``, so:
      - ``obs`` (the env return) is a (H, W, C) uint8 image → fed to the student.
      - ``info["state_obs"]`` is the (obs_dim,) state vector → fed to the teacher.
      - The student needs image obs; the teacher needs state obs.

    Args:
        beta: teacher mixing probability (1 = always teacher, 0 = always student)
    Returns:
        obs_list:        list of (H, W, C) uint8 image arrays visited during rollout
        teacher_actions: list of (action_dim,) teacher-labeled actions at each obs
        total_return:    sum of environment rewards
        success:         whether the task was completed
    """
    obs_list: list = []
    teacher_actions: list = []
    total_return = 0.0
    success = False

    use_pose = student.model._cfg.policy.use_pose_data
    instruction = env.instruction

    obs, info = env.reset()
    obs = np.ascontiguousarray(obs)           # (H, W, C) image

    # Encode goal once per episode (pure function, no side effects).
    txt_goal, goal_state = student.encode_goals(obs, instruction)

    # Extract pose for the student if needed.
    from hw3.train_transformer_rl import _extract_pose_from_info
    device = student.device
    pose = _extract_pose_from_info(info, student, device) if use_pose else None

    was_training = student.training
    student.eval()

    with torch.no_grad():
        for _step in range(max_steps):
            # ---- Teacher label (always, on the current state obs) ----
            state_obs = info["state_obs"]                          # (obs_dim,) float32
            teacher_action = teacher.get_action(state_obs, deterministic=True)

            # ---- Decide who drives ----
            if np.random.random() < beta:
                # Follow teacher
                action_np = teacher_action
            else:
                # Follow student
                obs_t = torch.from_numpy(obs).float().to(device)
                action_np, _, _, _ = student.get_action(
                    obs_t.unsqueeze(0), txt_goal, goal_state, pose, deterministic=False
                )

            # ---- Store (image obs, teacher label) ----
            obs_list.append(obs.copy())
            teacher_actions.append(teacher_action.copy())

            # ---- Step environment ----
            next_obs, reward, done, truncated, info = env.step(action_np)
            next_obs = np.ascontiguousarray(next_obs)

            total_return += float(reward)
            pose = _extract_pose_from_info(info, student, device) if use_pose else None

            if done or truncated:
                success = bool(info.get("success_placed", 0.0))
                break

            obs = next_obs

    student.train(was_training)
    return obs_list, teacher_actions, total_return, success


# ---------------------------------------------------------------------------
# Behavior cloning update step
# ---------------------------------------------------------------------------

def bc_update(student: TransformerPolicyWrapper,
              dataset: DAggerDataset,
              optimizer: torch.optim.Optimizer,
              cfg: DictConfig,
              device: torch.device,
              instruction: str = ""):
    """
    Run ``bc_epochs_per_round`` supervised epochs on the aggregated DAgger dataset.

    Loss: negative log-likelihood of teacher actions under the student's Gaussian.
    The student distribution is in z-score space; teacher actions are raw env-space
    actions, so we first encode them: z = (a - action_mean) / action_std
    (the inverse of ``_decode_action``), then compute the Gaussian log-prob.

    Args:
        instruction: task language instruction, used to encode the text goal.
                     The goal image is set to zeros (single-task setup — the
                     language goal dominates over the visual goal for BC).
    Returns dict with "bc_loss".
    """
    if len(dataset) == 0:
        return {"bc_loss": 0.0}

    n_epochs  = int(cfg.dagger.bc_epochs_per_round)
    mb_size   = int(cfg.training.minibatch_size)
    max_grad_norm = float(cfg.training.max_grad_norm)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=mb_size,
        shuffle=True,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    # Encode goal once — visual goal_state is zeros (same fallback as encode_goals
    # with first_obs=None), which is fine for single-task BC where the language
    # instruction carries most of the task information.
    txt_goal, goal_state = student.encode_goals(None, instruction)

    total_loss  = 0.0
    total_steps = 0

    student.train()
    for _epoch in range(n_epochs):
        for obs_batch, action_batch in loader:
            # obs_batch:    (B, H, W, C) float32 — raw [0,255] image obs
            # action_batch: (B, action_dim) float32 — raw env-space teacher actions
            obs_batch    = obs_batch.to(device)
            action_batch = action_batch.to(device)

            # Encode teacher raw actions → z-score space to match the student's
            # distribution (which lives in z-score space, not raw action space).
            # z = (a - action_mean) / action_std  (inverse of _decode_action)
            z_teacher = (action_batch - student._action_mean) / (student._action_std + 1e-8)

            # Forward pass — goal tensors broadcast from (1, …) to (B, …).
            dist = student.forward(obs_batch, txt_goal, goal_state)

            # Negative log-likelihood of teacher actions under the student Gaussian.
            loss = -TransformerPolicyWrapper._log_prob(dist, z_teacher).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(student.parameters()), max_grad_norm)
            optimizer.step()

            total_loss  += loss.item()
            total_steps += 1

    avg_loss = total_loss / max(1, total_steps)
    return {"bc_loss": avg_loss}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@hydra.main(config_path="conf", config_name="dagger", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.r_seed)
    np.random.seed(cfg.r_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project=cfg.experiment.project,
        name=cfg.experiment.name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    task_id = int(cfg.sim.eval_tasks[0])
    env = FastLIBEROEnv(task_id=task_id, max_episode_steps=cfg.sim.episode_length, cfg=cfg)
    eval_env = FastLIBEROEnv(task_id=task_id, max_episode_steps=cfg.sim.episode_length,
                              cfg=cfg, render_mode="rgb_array")

    obs_dim     = env.obs_dim
    action_dim  = env._action_dim
    instruction = env.instruction       # task language goal — constant for one task

    log_dir = HydraConfig.get().runtime.output_dir

    # Teacher: dense PPO policy from Part 1
    teacher = DensePolicyTeacher(
        cfg.teacher_checkpoint, obs_dim, action_dim,
        hidden_dim=256, n_layers=3, device=device
    )

    # Student: transformer policy initialized from HW1
    student = TransformerPolicyWrapper(cfg.student_init_checkpoint, device, cfg)
    student_optimizer = torch.optim.Adam(student.parameters(), lr=cfg.training.learning_rate)

    # Aggregated DAgger dataset
    dataset = DAggerDataset()
    os.makedirs(cfg.dagger.dataset_save_dir, exist_ok=True)

    all_returns = []
    all_successes = []

    # Pre-compute constants for beta schedule so we don't repeat the string
    # comparison and int() conversion on every round.
    _beta_schedule  = str(cfg.dagger.beta_schedule).lower()
    _num_rounds     = int(cfg.dagger.num_rounds)
    _beta_init      = float(cfg.dagger.beta_init)
    _eval_interval  = int(cfg.eval_interval)
    _save_interval  = int(cfg.save_interval)

    # --- DAgger rounds ---
    for round_idx in range(_num_rounds):

        # Compute beta (teacher mixing coefficient)
        # linear: decay from beta_init → 0 over num_rounds
        # constant: always beta_init
        if _beta_schedule == "linear" and _num_rounds > 1:
            beta = _beta_init * (1.0 - round_idx / (_num_rounds - 1))
        else:
            beta = _beta_init

        print(f"\n=== DAgger Round {round_idx + 1}/{_num_rounds} | beta={beta:.3f} ===")

        # Collect rollouts and label with teacher
        round_returns = []
        round_successes = []
        for _ in range(cfg.dagger.rollouts_per_round):
            obs_seq, teacher_acts, ret, success = collect_dagger_rollout(
                env, student, teacher, beta=beta, max_steps=cfg.sim.episode_length
            )
            dataset.add_rollout(obs_seq, teacher_acts)
            round_returns.append(ret)
            round_successes.append(float(success))

        all_returns.extend(round_returns)
        all_successes.extend(round_successes)

        # Save dataset for this round
        dataset.save(os.path.join(cfg.dagger.dataset_save_dir, f"dagger_round_{round_idx:03d}.pth"))

        # Behavior cloning update on aggregated dataset
        bc_info = bc_update(student, dataset, student_optimizer, cfg, device,
                            instruction=instruction)

        log = {
            "dagger/round": round_idx,
            "dagger/beta": beta,
            "dagger/dataset_size": len(dataset),
            "dagger/episode_return": np.mean(round_returns),
            "dagger/success_rate": np.mean(round_successes),
            **{f"dagger/{k}": v for k, v in bc_info.items()},
        }
        wandb.log(log, step=round_idx)
        print(f"  return={np.mean(round_returns):.3f}  "
              f"success={np.mean(round_successes):.2f}  "
              f"bc_loss={bc_info['bc_loss']:.4f}  "
              f"dataset_size={len(dataset)}")

        # Periodic evaluation using the same evaluate_policy as the RL trainers.
        if (round_idx + 1) % _eval_interval == 0:
            eval_metrics = evaluate_policy(
                student, eval_env, cfg, device,
                total_steps=round_idx,
                log_dir=log_dir,
            )
            wandb.log(eval_metrics, step=round_idx)

        # Periodic checkpoint
        if (round_idx + 1) % _save_interval == 0:
            ckpt_dir = os.path.join(log_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({
                "student": student.model.state_dict(),
                "round": round_idx,
                "cfg": OmegaConf.to_container(cfg),
            }, os.path.join(ckpt_dir, f"dagger_student_round{round_idx:03d}.pth"))

    # Save final student
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({
        "student": student.model.state_dict(),
        "cfg": OmegaConf.to_container(cfg),
    }, os.path.join(ckpt_dir, "dagger_student_final.pth"))
    print(f"\nDAgger training complete. Final student saved to {ckpt_dir}/dagger_student_final.pth")
    print("You can now fine-tune with RL using train_transformer_rl.py and "
          "init_checkpoint=<above path>")

    env.close()
    eval_env.close()
    wandb.finish()


if __name__ == "__main__":
    main()
