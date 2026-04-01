"""
HW3 Part 2: Fine-tune a transformer policy from HW1 with PPO or GRPO.

Usage (PPO):
    python hw3/train_transformer_rl.py \
        experiment.name=hw3_transformer_ppo_seed0 \
        r_seed=0 \
        init_checkpoint=/path/to/hw1/miniGRP.pth \
        rl.algorithm=ppo \
        sim.task_set=libero_spatial \
        sim.eval_tasks=[9]

Usage (GRPO with ground-truth resets):
    python hw3/train_transformer_rl.py \
        experiment.name=hw3_transformer_grpo_seed0 \
        r_seed=0 \
        init_checkpoint=/path/to/hw1/miniGRP.pth \
        rl.algorithm=grpo \
        sim.task_set=libero_spatial \
        sim.eval_tasks=[9]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../mini-grp'))
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import dill
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from hydra.core.hydra_config import HydraConfig
from hw3.train_dense_rl import RolloutBuffer, ppo_update
from hw3.libero_env_fast import FastLIBEROEnv


# ---------------------------------------------------------------------------
# Transformer backbone for the value function
# ---------------------------------------------------------------------------

class ValueFunction(nn.Module):
    """Transformer critic V(s), trained from scratch.  Token layout:
    [CLS] | obs patches | (pose) | text tokens | goal patches → scalar value.
    All hyper-parameters come from ``cfg.value``; only ``encode_with_t5`` and
    ``grp_n_embd`` / ``pose_mean`` are borrowed from the policy checkpoint.
    """

    def __init__(self, policy: "TransformerPolicyWrapper", device: torch.device, cfg: DictConfig):
        super().__init__()
        self.device = device
        from hw3.grp_model import Block, get_patches_fast, calc_positional_embeddings
        self._get_patches = get_patches_fast

        grp_cfg = policy.model._cfg
        self.encode_with_t5 = grp_cfg.dataset.encode_with_t5
        grp_n_embd = int(grp_cfg.n_embd)

        v = cfg.value
        n_embd, n_head, n_blocks = int(v.n_embd), int(v.n_head), int(v.n_blocks)
        dropout, hidden          = float(v.dropout), int(v.hidden_dim)
        patch_size, n_patches    = int(v.patch_size), int(v.n_patches)
        obs_stacking             = int(v.obs_stacking)
        image_shape              = list(v.image_shape)   # [H, W, C]
        self.use_pose            = bool(v.use_pose)
        self.max_block_size      = int(v.max_block_size)
        self.n_embd              = n_embd

        # cfg stub for get_patches_fast
        self._vcfg = OmegaConf.create({
            "patch_size": patch_size, "n_patches": n_patches,
            "n_embd": n_embd, "n_head": n_head, "dropout": dropout,
            "image_shape": image_shape,
            "policy": {"obs_stacking": obs_stacking, "use_pose_data": self.use_pose},
        })

        px = patch_size * patch_size * image_shape[2]
        self.obs_proj  = nn.Linear(px * obs_stacking, n_embd, bias=False)
        self.goal_proj = nn.Linear(px,                n_embd, bias=False)

        if self.encode_with_t5:
            self.txt_proj  = nn.Linear(grp_n_embd, n_embd, bias=False)
        else:
            self.txt_emb   = nn.Embedding(len(grp_cfg.dataset.chars_list), n_embd)

        if self.use_pose:
            self.pose_proj = nn.Linear(len(grp_cfg.dataset.pose_mean), n_embd, bias=False)

        self.cls_token = nn.Parameter(torch.randn(1, 1, n_embd) * 0.02)

        num_pose_tok = 1 if self.use_pose else 0
        seq_len = 1 + (n_patches ** 2) * obs_stacking + num_pose_tok + self.max_block_size + n_patches ** 2
        self.register_buffer("pos_emb", calc_positional_embeddings(seq_len, n_embd), persistent=False)

        self.blocks     = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_blocks)])
        self.ln_f       = nn.LayerNorm(n_embd)
        self.value_head = nn.Sequential(nn.Linear(n_embd, hidden), nn.Tanh(), nn.Linear(hidden, 1))

        self.apply(self._init_weights)
        nn.init.orthogonal_(self.value_head[-1].weight, gain=0.01)
        nn.init.constant_(self.value_head[-1].bias, 0)
        self.to(device)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor, txt_goal: torch.Tensor,
                goal_state: torch.Tensor, pose: torch.Tensor = None) -> torch.Tensor:
        """Return V(s) as a (B,) scalar tensor."""
        obs = obs.to(self.device)
        if obs.dtype == torch.uint8:    obs = obs.float().div_(127.5).sub_(1.0)
        elif obs.max() > 1.5:           obs = obs.div(127.5).sub(1.0)
        B = obs.shape[0]

        txt_goal_b   = txt_goal.to(self.device).expand(B, *txt_goal.shape[1:])
        goal_state_b = goal_state.to(self.device).expand(B, *goal_state.shape[1:])

        obs_tok  = self.obs_proj( self._get_patches(obs,          self._vcfg))
        goal_tok = self.goal_proj(self._get_patches(goal_state_b, self._vcfg))

        if self.encode_with_t5:
            txt_tok = self.txt_proj(txt_goal_b)
        else:
            ids = txt_goal_b.long()
            if ids.dim() == 3: ids = ids.squeeze(-1)
            txt_tok = self.txt_emb(ids)
        # pad / truncate to max_block_size
        T = txt_tok.shape[1]
        if T < self.max_block_size:
            txt_tok = torch.cat([txt_tok, txt_tok.new_zeros(B, self.max_block_size - T, self.n_embd)], dim=1)
        else:
            txt_tok = txt_tok[:, :self.max_block_size]

        tokens = [self.cls_token.expand(B, -1, -1), obs_tok]
        if self.use_pose and pose is not None:
            pose_t = pose.to(self.device)
            if pose_t.dim() == 2: pose_t = pose_t.unsqueeze(1)
            tokens.append(self.pose_proj(pose_t))
        tokens += [txt_tok, goal_tok]

        x = torch.cat(tokens, dim=1)
        x = x + self.pos_emb[:x.size(1)].unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        return self.value_head(self.ln_f(x)[:, 0]).squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# Transformer policy wrapper
# ---------------------------------------------------------------------------

class TransformerPolicyWrapper(nn.Module):
    """PPO-compatible wrapper around the HW1 GRP transformer.

    Key contract for PPO (``ppo_update`` in ``train_dense_rl.py``):
      - callable: ``dist = policy(obs_batch, txt_goal, goal_state)`` → Normal
      - ``get_action()`` → ``(action_np, log_prob, entropy, z_sample)``

    Goal conditioning is passed **explicitly** on every call — there is no
    hidden cached state, which makes the importance-weight ratio in PPO/GRPO
    exact even when a minibatch spans multiple episodes with different goals.
    Use ``encode_goals()`` once per episode to obtain the tensors.
    """

    def __init__(self, checkpoint_path: str, device: torch.device, cfg: DictConfig):
        super().__init__()
        self.model  = torch.load(checkpoint_path, map_location=device, pickle_module=dill)
        self.action_head = self.model.mlp
        self.cfg         = cfg
        self.device      = device

        # action_dim: use model.mlp[0] (the Linear layer) regardless of whether
        # the model is discrete (mlp has 2 layers: Linear + LayerNorm) or
        # continuous (mlp has 1 layer: Linear).  mlp[0] is always the Linear.
        action_out_features = self.action_head[0].out_features

        # Learnable state-independent log std for Gaussian head.
        self._action_log_std = nn.Parameter(torch.zeros(
            action_out_features, device=device
        ))
        nn.init.constant_(self._action_log_std, 0)  # std ~ 0.45 at start of RL
        # Cache decode_action tensors on device so forward() never rebuilds them.
        self.register_buffer(
            "_action_mean",
            torch.tensor(self.model._cfg.dataset.action_mean, dtype=torch.float32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_action_std",
            torch.tensor(self.model._cfg.dataset.action_std, dtype=torch.float32, device=device),
            persistent=False,
        )
        self.to(device)

    def encode_goals(self, first_obs: np.ndarray, instruction: str):
        """Encode goal conditioning for one episode and return the tensors.

        Returns:
            txt_goal:   (1, T) int64 token ids  OR  (1, T, n_embd) T5 floats
            goal_state: (1, H, W, C) float tensor normalised to [-1, 1]

        The caller owns these tensors and passes them explicitly to
        ``forward()`` and ``get_action()`` on every step of the episode.
        This avoids any implicit cached state on the wrapper.
        """
        model = self.model
        txt_goal = model.encode_text_goal(instruction).to(self.device)
        if first_obs is not None:
            # preprocess_goal_image: resize + (img/255)*2-1 → (H, W, C) float [-1,1]
            goal_np    = model.preprocess_goal_image(first_obs)
            goal_state = torch.tensor(
                goal_np, dtype=torch.float32, device=self.device
            ).unsqueeze(0)   # (1, H, W, C)
        else:
            goal_state = torch.zeros(
                1, *model._cfg.image_shape, dtype=torch.float32, device=self.device
            )
        return txt_goal, goal_state

    def _preprocess_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalise a raw [0,255] image tensor to [-1,1]."""
        if obs.dtype == torch.uint8:
            return obs.float().div_(127.5).sub_(1.0)
        if obs.max() > 1.5:          # float still in [0, 255]
            return obs.div(127.5).sub(1.0)
        return obs                   # already in [-1, 1]

    def forward(
        self,
        obs: torch.Tensor,
        txt_goal: torch.Tensor,
        goal_state: torch.Tensor,
        pose: torch.Tensor = None,
    ) -> Normal:
        """Return the action distribution for a batch of observations.

        Args:
            obs:        (B, H, W, C) raw or normalised image observation.
            txt_goal:   (1, T[, n_embd]) text goal from ``encode_goals()``.
            goal_state: (1, H, W, C) or (B, H, W, C) goal image — broadcast to B
                        if shape[0]==1, used per-step if shape[0]==B.
            pose:       (B, pose_emb_dim) encoded pose, or None.
        Returns:
            Normal distribution in z-score (GRP normalised) action space.
        """
        obs = self._preprocess_obs(obs.to(self.device))
        B   = obs.shape[0]

        # Expand (1, …) goal tensors to the batch size — zero-copy views.
        txt_goal_b   = txt_goal.to(self.device).expand(B,   *txt_goal.shape[1:])
        goal_state_b = goal_state.to(self.device).expand(B, *goal_state.shape[1:])

        raw_logits, _ = self.model(obs, txt_goal_b, goal_state_b, pose=pose)
        # raw_logits is in the GRP normalised action space (z-scores).
        # Keep the distribution in that space — tanh squashing is done in
        # get_action(), so the pretrained mean is faithfully preserved at
        # the start of RL fine-tuning and the log-prob is self-consistent.
        log_std    = self._action_log_std.clamp(-5.0, 1.0)   # cap std at exp(0)=1.0 to prevent entropy explosion
        action_std = log_std.exp().expand_as(raw_logits)
        return Normal(raw_logits, action_std)

    def _decode_action(self, normalised_action: torch.Tensor) -> torch.Tensor:
        """Decode GRP normalised action (z-score) → raw action space."""
        return normalised_action * self._action_std + self._action_mean

    @staticmethod
    def _log_prob(dist: Normal, z_action: torch.Tensor) -> torch.Tensor:
        """Log-prob of a sampled z-score action under the Gaussian distribution.

        The policy distribution lives in z-score space and the env action is
        obtained via linear decode (z * std + mean), which is a volume-preserving
        affine map up to a constant Jacobian that cancels in importance ratios.
        No Jacobian correction is needed here.
        """
        return dist.log_prob(z_action).sum(-1)

    # Keep old name as alias so grpo_update and ppo_update call sites don't break.
    _tanh_log_prob = _log_prob

    def get_action(
        self,
        obs_t: torch.Tensor,
        txt_goal: torch.Tensor,
        goal_state: torch.Tensor,
        pose: torch.Tensor = None,
        deterministic: bool = False,
    ):
        """Sample an action and return rollout data.

        Args:
            obs_t:      (1, H, W, C) or (H, W, C) raw image tensor.
            txt_goal:   (1, T[, n_embd]) from ``encode_goals()``.
            goal_state: (1, H, W, C) from ``encode_goals()``.
            pose:       (1, pose_emb_dim) encoded pose, or None.
            deterministic: if True, use dist.mean (no sampling).
        Returns:
            action_np:  (action_dim,) numpy array in the env's action space.
            log_prob:   scalar tensor (detached).
            entropy:    scalar tensor (detached).
            z_sample:   (action_dim,) tensor in z-score space (for PPO buffer).
        """
        if obs_t.dim() == 3:
            obs_t = obs_t.unsqueeze(0)
        dist     = self.forward(obs_t, txt_goal, goal_state, pose)
        z_sample = dist.mean if deterministic else dist.rsample()  # z-score space
        log_prob = self._log_prob(dist, z_sample)
        entropy  = dist.entropy().sum(-1)
        # Decode z-score → raw action space (same transform used in BC training
        # and sim_eval.py: action = z * action_std + action_mean).
        action_t = self._decode_action(z_sample)
        return (
            action_t.squeeze(0).detach().cpu().numpy(),
            log_prob.squeeze(0),
            entropy.squeeze(0),
            z_sample.squeeze(0),
        )


# ---------------------------------------------------------------------------
# GRPO helpers
# ---------------------------------------------------------------------------

def _extract_pose_from_info(info: dict, policy: "TransformerPolicyWrapper", device: torch.device) -> torch.Tensor:
    """Extract and encode a (1, pose_emb_dim) pose tensor from an env info dict.

    ``FastLIBEROEnv`` stores the full state vector under ``info["state_obs"]``
    with layout ``[eef_pos(3), eef_quat_xyz(3), gripper(1), ...]``.
    Uses torch.from_numpy for a zero-copy transfer from the numpy buffer.
    """
    state_obs = info.get("state_obs", None)
    if state_obs is not None:
        # state_obs is already a numpy array; slice and copy minimally.
        pose_np = np.ascontiguousarray(state_obs[:7], dtype=np.float32)
    else:
        pose_np = np.concatenate([
            np.asarray(info["robot0_eef_pos"],           dtype=np.float32),
            np.asarray(info["robot0_eef_quat"][:3],      dtype=np.float32),
            np.asarray([info["robot0_gripper_qpos"][0]], dtype=np.float32),
        ], axis=-1)
    # from_numpy avoids a data copy; unsqueeze adds the batch dim.
    pose_t = torch.from_numpy(pose_np).unsqueeze(0)   # (1, 7)
    return policy.model.encode_pose(pose_t).to(device)  # (1, pose_emb_dim)


def collect_grpo_group(env: FastLIBEROEnv,
                       policy: TransformerPolicyWrapper,
                       init_state,
                       group_size: int,
                       max_steps: int,
                       device: torch.device):
    """
    Reset to the same initial state and collect `group_size` trajectories.

    Returns a list of trajectory dicts, each containing:
        obs, actions, log_probs, poses, txt_goal, goal_state, rewards, dones, total_return
    """
    trajectories = []
    instruction = env.instruction
    use_pose = policy.model._cfg.policy.use_pose_data

    if group_size <= 0:
        return trajectories

    # Disable dropout during collection so old_log_probs are deterministic
    # and the importance-weight ratio in grpo_update is meaningful.
    was_training = policy.training
    policy.eval()

    for _ in range(group_size):
        obs, info = env.reset(options={"init_state": init_state})
        obs = np.ascontiguousarray(obs)
        if obs.ndim != 3:
            raise ValueError(f"Expected image observations for transformer GRPO, got shape={obs.shape}")

        # Encode goal conditioning for this episode — returned as plain tensors,
        # no hidden state on the policy object.
        txt_goal, goal_state = policy.encode_goals(obs, instruction)

        traj_obs = []
        traj_actions = []
        traj_log_probs = []
        traj_poses = []
        traj_rewards = []
        traj_dones = []

        total_return = 0.0

        # Encode initial pose from reset info
        pose = _extract_pose_from_info(info, policy, device) if use_pose else None

        for _step in range(max_steps):
            # Convert obs to a GPU tensor for the network forward pass.
            obs_t = torch.from_numpy(obs).float().to(device)
            with torch.no_grad():
                action_np, log_prob, _entropy, z_sample = policy.get_action(
                    obs_t, txt_goal, goal_state, pose
                )

            next_obs, reward, done, truncated, _info = env.step(action_np)
            terminal = bool(done or truncated)

            # Store on CPU immediately to keep GPU memory free during collection.
            traj_obs.append(obs_t.cpu())
            traj_actions.append(z_sample.cpu())
            traj_log_probs.append(log_prob.cpu())
            if use_pose and pose is not None:
                traj_poses.append(pose.squeeze(0).cpu())
            traj_rewards.append(float(reward))
            traj_dones.append(float(terminal))

            total_return += float(reward)
            obs = np.ascontiguousarray(next_obs)

            if use_pose:
                pose = _extract_pose_from_info(_info, policy, device)

            if terminal:
                break

        if traj_obs:
            trajectory = {
                "obs":        torch.stack(traj_obs,    dim=0),
                "actions":    torch.stack(traj_actions, dim=0),
                "log_probs":  torch.stack(traj_log_probs, dim=0),
                "poses":      torch.stack(traj_poses, dim=0) if traj_poses else None,
                "txt_goal":   txt_goal.cpu(),    # (1, T[, n_embd]) — same for whole traj
                "goal_state": goal_state.cpu(),  # (1, H, W, C) — same for whole traj
                "rewards":    torch.tensor(traj_rewards, dtype=torch.float32),
                "dones":      torch.tensor(traj_dones,   dtype=torch.float32),
                "total_return": float(total_return),
            }
        else:
            # Extremely defensive fallback: empty trajectory if env terminates instantly.
            trajectory = {
                "obs":        torch.empty((0, *obs.shape), dtype=torch.float32),
                "actions":    torch.empty((0, env._action_dim), dtype=torch.float32),
                "log_probs":  torch.empty((0,), dtype=torch.float32),
                "poses":      None,
                "txt_goal":   txt_goal.cpu(),
                "goal_state": goal_state.cpu(),
                "rewards":    torch.empty((0,), dtype=torch.float32),
                "dones":      torch.empty((0,), dtype=torch.float32),
                "total_return": 0.0,
            }
        trajectories.append(trajectory)

    policy.train(was_training)
    return trajectories


def grpo_update(policy: TransformerPolicyWrapper,
                policy_optimizer: torch.optim.Optimizer,
                trajectories_per_group: list,
                cfg: DictConfig,
                device: torch.device,
                ref_policy: "TransformerPolicyWrapper | None" = None):
    """
    GRPO update: compute group-relative advantages and update policy.

    Implements the full GRPO surrogate objective from DeepSeekMath (Shao et al. 2024):

        J_GRPO = E[ (1/G) Σ_i (1/T_i) Σ_t [
                    min(r_it * Â_i, clip(r_it, 1±ε) * Â_i)
                    - β * KL[π_θ(·|s_t) ‖ π_ref(·|s_t)]
                 ]]

    where:
        r_it   = π_θ(a_it|s_it) / π_old(a_it|s_it)   (importance ratio)
        Â_i    = (R_i - mean(R)) / (std(R) + ε)        (group-relative advantage)
        π_ref  = frozen reference policy (BC init checkpoint, if provided)
        β      = cfg.grpo.kl_coef

    To avoid CUDA OOM with long trajectories, gradients are accumulated via
    chunked forward-backward passes (chunk_size steps at a time) rather than
    building one giant activation graph across all steps.  The final optimizer
    step fires once after all chunks have been accumulated.

    Args:
        trajectories_per_group: list of lists; each inner list is a group of
            trajectory dicts collected from the same initial state.
        ref_policy: frozen reference policy for KL penalty; if None the KL
            term is skipped (equivalent to β=0).
    Returns:
        dict with "policy_loss", "kl", "mean_return"
    """
    clip_eps      = float(getattr(cfg.training, "clip_epsilon", getattr(cfg.training, "clip_eps")))
    entropy_coef  = float(getattr(cfg.training, "entropy_coef", getattr(cfg.training, "entropy_coeff", 0.0)))
    max_grad_norm = float(getattr(cfg.training, "max_grad_norm", 0.5))
    kl_coef       = float(getattr(cfg.grpo, "kl_coef", 0.0))
    # chunk_size controls how many timesteps are forwarded at once to bound
    # peak activation memory.  Smaller = less VRAM, more backward calls.
    chunk_size    = int(getattr(cfg.grpo, "chunk_size", 32))

    # ---- 1. First pass (no grad): compute group-relative advantages ----
    # We need group returns *before* touching the GPU graph, so collect them
    # cheaply here.  All trajectory data stays on CPU until the chunked
    # forward pass below.
    mean_returns, group_adv_stats = [], []
    # flat list of (traj_dict, scalar_advantage) for the second pass
    flat_trajs: list[tuple[dict, float]] = []

    for group in trajectories_per_group:
        if not group:
            continue
        group_returns = torch.tensor(
            [float(t.get("total_return", 0.0)) for t in group],
            dtype=torch.float32,
        )
        if group_returns.numel() == 0:
            continue
        g_mean = group_returns.mean()
        g_std  = group_returns.std(unbiased=False)
        group_adv = (group_returns - g_mean) / (g_std + 1e-8)  # (G,)

        mean_returns.append(g_mean)
        group_adv_stats.append(group_adv.abs().mean())

        for traj, adv_scalar in zip(group, group_adv):
            if traj["obs"].numel() == 0:
                continue
            flat_trajs.append((traj, float(adv_scalar.item())))

    if not flat_trajs:
        return {"policy_loss": 0.0, "kl": 0.0, "entropy": 0.0, "mean_return": 0.0, "group_adv_mean_abs": 0.0}

    # Total number of timesteps across all trajectories — used to normalise the
    # gradient accumulation so each step contributes equally (same as .mean()).
    total_steps = sum(int(traj["obs"].shape[0]) for traj, _ in flat_trajs)

    # ---- 2. Chunked forward-backward with gradient accumulation ----
    policy_optimizer.zero_grad(set_to_none=True)
    was_training = policy.training
    policy.eval()  # no dropout during recomputation — importance weights must be exact

    # Accumulators for logging (detached scalars only — no retained graph)
    acc_policy_loss = 0.0
    acc_kl          = 0.0
    acc_entropy     = 0.0
    n_kl_chunks     = 0

    for traj, traj_adv in flat_trajs:
        obs_cpu    = traj["obs"]        # (T, H, W, C) — stays on CPU between chunks
        act_cpu    = traj["actions"]    # (T, action_dim)
        lp_old_cpu = traj["log_probs"]  # (T,)
        poses_cpu  = traj["poses"]      # (T, pose_dim) or None
        txt_goal   = traj["txt_goal"].to(device)    # (1, T_txt[, n_embd])
        goal_state = traj["goal_state"].to(device)  # (1, H, W, C)
        T          = obs_cpu.shape[0]

        for chunk_start in range(0, T, chunk_size):
            chunk_end = min(chunk_start + chunk_size, T)
            chunk_len = chunk_end - chunk_start

            obs_chunk    = obs_cpu[chunk_start:chunk_end].to(device)
            act_chunk    = act_cpu[chunk_start:chunk_end].to(device)
            lp_old_chunk = lp_old_cpu[chunk_start:chunk_end].to(device).detach()
            pose_chunk   = (poses_cpu[chunk_start:chunk_end].to(device)
                            if poses_cpu is not None else None)

            dist    = policy(obs_chunk, txt_goal, goal_state, pose_chunk)
            new_lp  = TransformerPolicyWrapper._log_prob(dist, act_chunk)   # (chunk,)
            ent     = dist.entropy().sum(-1)                                  # (chunk,)

            ratio = torch.exp(new_lp - lp_old_chunk)
            adv   = torch.full((chunk_len,), traj_adv, dtype=torch.float32, device=device)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            chunk_policy_loss = -torch.min(surr1, surr2).mean()
            chunk_entropy     = ent.mean()

            # KL penalty (analytically exact for Gaussians)
            chunk_kl = torch.tensor(0.0, device=device)
            if ref_policy is not None and kl_coef > 0.0:
                with torch.no_grad():
                    ref_dist = ref_policy(obs_chunk, txt_goal, goal_state, pose_chunk)
                chunk_kl = torch.distributions.kl_divergence(dist, ref_dist).sum(-1).mean()
                n_kl_chunks += 1

            chunk_loss = (chunk_policy_loss
                          - entropy_coef * chunk_entropy
                          + kl_coef * chunk_kl)

            # Scale by fraction of total steps so that summing all chunks equals
            # the true mean over all timesteps (equivalent to global .mean()).
            scale = chunk_len / total_steps
            (chunk_loss * scale).backward()

            # Logging accumulators (detached — no graph retained)
            acc_policy_loss += chunk_policy_loss.detach().item() * scale
            acc_entropy     += chunk_entropy.detach().item()     * scale
            acc_kl          += chunk_kl.detach().item()          * scale

            # Free the chunk's activation memory immediately before the next chunk.
            del obs_chunk, act_chunk, lp_old_chunk, pose_chunk, dist
            del new_lp, ent, ratio, adv, surr1, surr2, chunk_policy_loss, chunk_entropy, chunk_kl, chunk_loss

    nn.utils.clip_grad_norm_(list(policy.parameters()), max_grad_norm)
    policy_optimizer.step()
    policy.train(was_training)

    def _mean_tensor(xs):
        return torch.stack(list(xs)).mean().item() if xs else 0.0

    return {
        "policy_loss":        acc_policy_loss,
        "kl":                 acc_kl if n_kl_chunks > 0 else 0.0,
        "entropy":            acc_entropy,
        "mean_return":        _mean_tensor(mean_returns),
        "group_adv_mean_abs": _mean_tensor(group_adv_stats),
    }


# ---------------------------------------------------------------------------
# GRPO with world model (Part 2d)
# ---------------------------------------------------------------------------

def grpo_worldmodel_update(policy: TransformerPolicyWrapper,
                            world_model,
                            current_obs: np.ndarray,
                            instruction: str,
                            group_size: int,
                            horizon: int,
                            cfg: DictConfig,
                            device: torch.device,
                            policy_optimizer: torch.optim.Optimizer | None = None,
                            ref_policy: "TransformerPolicyWrapper | None" = None):
    """
    GRPO using the HW2 world model to generate imagined trajectories.

    Algorithm
    ---------
    For each of ``group_size`` imagined rollouts starting from ``current_obs``:
      1. Sample an action from the policy distribution (so gradients flow).
      2. Ask the (frozen) world model for (next_obs_recon, reward, continue).
      3. Repeat for ``horizon`` steps or until the world model predicts episode end.
      4. Compute per-trajectory return = sum of predicted rewards.
    Then normalise returns within the group to get group-relative advantages and
    update the policy with the REINFORCE / clipped-surrogate objective.

    Args:
        world_model:       Trained HW2 world model.  Must be a DreamerV3 instance
                           (has ``encoder``, ``rssm_step``, ``reward_head``,
                           ``continue_head``, ``decoder`` attributes).
        current_obs:       (H, W, C) uint8 or float image — real env observation
                           used as the imagination starting point.
        group_size:        Number of imagined trajectories per update.
        horizon:           Max imagination steps per trajectory.
        policy_optimizer:  When provided the policy is updated in-place; otherwise
                           only metrics are returned (useful for debugging).
        ref_policy:        Frozen reference policy π_ref for KL penalty
                           β·KL[π_θ(·|s_t) ‖ π_ref(·|s_t)].  If None the KL
                           term is skipped (equivalent to cfg.grpo.kl_coef=0).
    Returns:
        dict with "policy_loss", "kl", "entropy", "mean_imagined_return",
                  "group_adv_mean_abs", "imagined_steps".
    """
    _empty = {
        "policy_loss": 0.0,
        "kl": 0.0,
        "entropy": 0.0,
        "mean_imagined_return": 0.0,
        "group_adv_mean_abs": 0.0,
        "imagined_steps": 0,
    }
    if group_size <= 0 or horizon <= 0:
        return _empty

    clip_eps     = float(getattr(cfg.training, "clip_epsilon",
                                 getattr(cfg.training, "clip_eps", 0.2)))
    entropy_coef = float(getattr(cfg.training, "entropy_coef",
                                 getattr(cfg.training, "entropy_coeff", 0.0)))
    max_grad_norm = float(getattr(cfg.training, "max_grad_norm", 0.5))
    kl_coef       = float(getattr(cfg.grpo, "kl_coef", 0.0))

    # ------------------------------------------------------------------ #
    # Validate world-model type — only DreamerV3 supports image-space     #
    # imagination (obs_shape, encoder, rssm_step, decoder, reward_head).  #
    # ------------------------------------------------------------------ #
    model_type = getattr(world_model, "type", "").lower()
    if "dreamer" not in model_type:
        raise NotImplementedError(
            "grpo_worldmodel_update requires a DreamerV3 world model. "
            f"Got type={world_model.__class__.__name__!r}."
        )

    if not isinstance(current_obs, np.ndarray):
        current_obs = np.asarray(current_obs, dtype=np.float32)
    current_obs = np.ascontiguousarray(current_obs)
    if current_obs.ndim != 3:
        raise ValueError(
            f"current_obs must be (H, W, C), got shape {current_obs.shape}"
        )

    # Encode goal conditioning once for all imagined trajectories in this group.
    # encode_goals is a pure function (no side effects on policy state).
    txt_goal, goal_state = policy.encode_goals(current_obs, instruction)

    # ------------------------------------------------------------------ #
    # Helper: (H,W,C) uint8/float numpy → (1,C,H,W) float tensor [-1,1] #
    # ------------------------------------------------------------------ #
    def _obs_to_dreamer(obs_np: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(np.ascontiguousarray(obs_np)).float().to(device)
        if t.shape[-1] in (1, 3, 6, 9, 12):      # (H,W,C) → (C,H,W)
            t = t.permute(2, 0, 1)
        if t.max() > 1.5:                          # [0,255] → [-1,1]
            t = t / 127.5 - 1.0
        return t.unsqueeze(0).clamp(-1.0, 1.0)    # (1,C,H,W)

    # ------------------------------------------------------------------ #
    # Helper: dreamer recon (1,C,H,W) [-1,1] → (H,W,C) float [0,255]   #
    # for feeding back through the transformer policy.                   #
    # ------------------------------------------------------------------ #
    def _dreamer_to_policy_obs(recon: torch.Tensor) -> torch.Tensor:
        r = recon.squeeze(0)                       # (C,H,W)
        r = (r + 1.0) * 127.5                      # [-1,1] → [0,255]
        r = r.clamp(0.0, 255.0).permute(1, 2, 0)  # (H,W,C)
        return r.detach()                          # sever WM graph before policy fwd

    # ------------------------------------------------------------------ #
    # Freeze world model for imagination — we only update the policy.    #
    # ------------------------------------------------------------------ #
    was_training_wm = world_model.training
    world_model.eval()
    # Disable policy dropout during imagination — we want the score-function
    # gradient (REINFORCE) to estimate the *expected* advantage, not the
    # dropout-corrupted one.  We re-enable before the backward pass.
    was_training_policy = policy.training
    policy.eval()
    if policy_optimizer is not None:
        policy_optimizer.zero_grad(set_to_none=True)

    action_dim   = len(policy.model._cfg.dataset.action_mean)   # e.g. 7
    start_chw    = _obs_to_dreamer(current_obs)   # (1,C,H,W)

    # Per-trajectory accumulation for GRPO
    group_traj_log_probs : list[torch.Tensor] = []
    group_traj_entropies : list[torch.Tensor] = []
    group_traj_kls       : list[torch.Tensor] = []   # per-traj mean KL (scalar tensors)
    group_total_returns  : list[float]        = []
    imagined_steps = 0

    with torch.no_grad():
        # Pre-encode the starting frame once to build initial RSSM state.
        embed0 = world_model.encoder(start_chw)   # (1, hidden_dim)

    for _traj in range(group_size):
        # Each trajectory starts from the SAME real observation but samples
        # different stochastic actions → different imagined futures.
        rssm_state = world_model.get_initial_state(1, device)

        # Step the RSSM once on the real observation (posterior) so the
        # deterministic state h₀ is grounded in real data.
        with torch.no_grad():
            step_out = world_model.rssm_step(
                rssm_state,
                action=torch.zeros(1, action_dim, device=device),
                embed=embed0,
            )
        rssm_state = {
            "h": step_out["h"].detach(),
            "z": step_out["z"].detach(),
            "z_probs": step_out.get("z_probs", None),
        }

        # Current "imagined" obs for the policy (H,W,C float, on device)
        cur_policy_obs = _dreamer_to_policy_obs(start_chw)   # no grad needed

        traj_log_probs : list[torch.Tensor] = []
        traj_entropies : list[torch.Tensor] = []
        traj_kls       : list[torch.Tensor] = []
        traj_reward_sum = 0.0

        for _step in range(horizon):
            # ---- Policy samples action (grad flows through log_prob) ----
            dist     = policy(cur_policy_obs.unsqueeze(0), txt_goal, goal_state)  # → Normal
            z_sample = dist.rsample()                         # (1, action_dim), z-score
            action   = policy._decode_action(z_sample)        # decode to raw action space
            lp       = TransformerPolicyWrapper._log_prob(dist, z_sample)  # (1,)
            ent      = dist.entropy().sum(-1)                  # (1,)
            traj_log_probs.append(lp.squeeze(0))
            traj_entropies.append(ent.squeeze(0))

            # ---- KL penalty: KL[π_θ(·|s_t) ‖ π_ref(·|s_t)] ----
            if ref_policy is not None and kl_coef > 0.0:
                with torch.no_grad():
                    ref_dist = ref_policy(cur_policy_obs.unsqueeze(0), txt_goal, goal_state)
                kl_step = torch.distributions.kl_divergence(dist, ref_dist).sum(-1)  # (1,)
                traj_kls.append(kl_step.squeeze(0))

            # ---- World model step (no grad — WM is frozen) ----
            # The world model was trained with z-score encoded actions
            # (model.encode_action() in dreamer_model_trainer.py).
            # Pass z_sample (z-score space) — NOT the decoded raw action.
            with torch.no_grad():
                step_out = world_model.rssm_step(
                    rssm_state,
                    action=z_sample.detach(),
                    embed=None,              # imagination: use prior, not posterior
                )
                h_t = step_out["h"]          # (1, deter_dim)
                z_t = step_out["z"]          # (1, stoch_dim * discrete_dim)
                feat_t = torch.cat([h_t, z_t], dim=-1)   # (1, feat_dim)

                reward_pred  = world_model.reward_head(feat_t).squeeze()    # scalar
                cont_logit   = world_model.continue_head(feat_t).squeeze()  # scalar
                cont_prob    = torch.sigmoid(cont_logit).item()

                # Reconstruct next observation for the policy
                recon_t = world_model.decoder(feat_t)     # (1, C, H, W)

            traj_reward_sum += float(reward_pred.item())
            imagined_steps  += 1

            rssm_state = {
                "h": h_t,
                "z": z_t,
                "z_probs": step_out.get("z_probs", None),
            }
            cur_policy_obs = _dreamer_to_policy_obs(recon_t)   # (H,W,C)

            if cont_prob < 0.5:   # world model predicts episode end
                break

        if not traj_log_probs:
            continue

        group_traj_log_probs.append(torch.stack(traj_log_probs))
        group_traj_entropies.append(torch.stack(traj_entropies))
        group_traj_kls.append(torch.stack(traj_kls) if traj_kls else None)
        group_total_returns.append(traj_reward_sum)

    # Restore world model training mode
    if was_training_wm:
        world_model.train()
    policy.train(was_training_policy)

    if not group_total_returns:
        return _empty

    # ------------------------------------------------------------------ #
    # Group-relative advantage normalisation (GRPO core)                 #
    # ------------------------------------------------------------------ #
    returns_t   = torch.tensor(group_total_returns, dtype=torch.float32, device=device)
    group_mean  = returns_t.mean()
    group_std   = returns_t.std(unbiased=False)
    group_advs  = (returns_t - group_mean) / (group_std + 1e-8)   # (G,)

    # ------------------------------------------------------------------ #
    # Compute policy objective and backpropagate                          #
    # ------------------------------------------------------------------ #
    # Accumulate as a Python list instead of pre-allocating a zero tensor,
    # so total_loss is always a real computation graph node (requires_grad=True)
    # when there are trajectories — avoiding the "tensor(0.0).requires_grad=False"
    # pitfall that would silently skip the optimizer step.
    total_loss_terms : list[torch.Tensor] = []
    policy_losses : list[torch.Tensor] = []
    entropies_acc : list[torch.Tensor] = []
    kl_acc        : list[torch.Tensor] = []

    for log_probs_t, entropies_t, kls_t, adv_scalar in zip(
        group_traj_log_probs, group_traj_entropies, group_traj_kls, group_advs
    ):
        adv      = adv_scalar.detach()
        adv_vec  = adv.expand_as(log_probs_t)

        # Pure REINFORCE score-function estimator.
        # (No old-policy ratio is available because we generate fresh on-policy
        # actions every call — ratio = 1 identically — so the clipped-surrogate
        # collapses to a plain REINFORCE term.  Using log_probs directly keeps the
        # gradient flowing correctly through the sampled actions.)
        policy_loss = -(adv_vec.detach() * log_probs_t).mean()
        entropy     = entropies_t.mean()

        # KL penalty β·KL[π_θ ‖ π_ref] averaged over this trajectory's steps.
        if kls_t is not None and kl_coef > 0.0:
            kl_penalty = kls_t.mean()
            kl_acc.append(kl_penalty.detach())
        else:
            kl_penalty = torch.tensor(0.0, device=device)

        traj_loss = policy_loss - entropy_coef * entropy + kl_coef * kl_penalty

        total_loss_terms.append(traj_loss)
        policy_losses.append(policy_loss.detach())
        entropies_acc.append(entropy.detach())

    if policy_optimizer is not None and total_loss_terms:
        total_loss = torch.stack(total_loss_terms).mean()
        total_loss.backward()
        nn.utils.clip_grad_norm_(list(policy.parameters()), max_grad_norm)
        policy_optimizer.step()

    def _mean_scalar(xs: list) -> float:
        if not xs:
            return 0.0
        return torch.stack(xs).mean().item()

    return {
        "policy_loss":          _mean_scalar(policy_losses),
        "kl":                   _mean_scalar(kl_acc),
        "entropy":              _mean_scalar(entropies_acc),
        "mean_imagined_return": float(np.mean(group_total_returns)),
        "group_adv_mean_abs":   group_advs.abs().mean().item(),
        "imagined_steps":       imagined_steps,
    }


def evaluate_policy(
    policy: TransformerPolicyWrapper,
    eval_env: "FastLIBEROEnv",
    cfg: DictConfig,
    device: torch.device,
    total_steps: int,
    log_dir: str,
) -> dict:
    """Run deterministic evaluation episodes and log metrics + a video to W&B.

    One video is captured from the episode with the **highest success rate**
    (ties broken by lowest episode index, i.e. the first episode wins).
    Metrics are averaged over all ``cfg.sim.eval_episodes`` episodes and
    returned as a plain dict (keys: ``eval/success_rate``,
    ``eval/avg_reward``, ``eval/avg_episode_length``).

    Args:
        policy:      The policy network to evaluate (set to eval mode internally).
        eval_env:    A ``FastLIBEROEnv`` instantiated with ``render_mode='rgb_array'``
                     so that ``render()`` returns frames.
        cfg:         Hydra config (uses ``cfg.sim.eval_episodes``,
                     ``cfg.sim.episode_length``).
        device:      Torch device.
        total_steps: Current training step count (used as W&B x-axis).
        log_dir:     Hydra output directory; videos are saved there before upload.

    Returns:
        dict with scalar metrics (all under the ``eval/`` prefix).
    """
    n_episodes    = int(cfg.sim.eval_episodes)
    max_ep_steps  = int(cfg.sim.episode_length)
    video_fps     = int(getattr(cfg.sim, "video_fps", 20))
    cam_name      = str(getattr(cfg.sim, "fast_env_image_camera", "agentview"))
    # Use a dedicated video render size (default 256) that is independent of
    # fast_env_image_size, which is the low-res observation fed to the policy.
    render_size   = int(getattr(cfg.sim, "video_render_size", 256))

    ep_returns    : list[float]             = []
    ep_lengths    : list[int]               = []
    ep_successes  : list[float]             = []
    all_ep_frames : list[list[np.ndarray]]  = []   # one list of frames per episode

    was_training = policy.training
    policy.eval()

    use_pose = policy.model._cfg.policy.use_pose_data

    with torch.no_grad():
        for _ in range(n_episodes):
            obs, info = eval_env.reset()
            obs = np.ascontiguousarray(obs)
            # Encode goal conditioning for this episode — explicit, no hidden state.
            txt_goal, goal_state = policy.encode_goals(obs, eval_env.instruction)
            pose = _extract_pose_from_info(info, policy, device) if use_pose else None
            ep_return = 0.0
            ep_length = 0
            success   = 0.0
            ep_frames : list[np.ndarray] = []

            for _ in range(max_ep_steps):
                obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                action_np, _, _, _ = policy.get_action(
                    obs_t, txt_goal, goal_state, pose, deterministic=True
                )

                obs, reward, done, truncated, info = eval_env.step(action_np)
                obs = np.ascontiguousarray(obs)
                pose = _extract_pose_from_info(info, policy, device) if use_pose else None

                ep_return += float(reward)
                ep_length += 1

                frame = eval_env.render(camera_name=cam_name, width=render_size, height=render_size)
                if frame is not None:
                    ep_frames.append(frame)

                if done or truncated:
                    success = float(info.get("success_placed", 0.0))
                    break

            ep_returns.append(ep_return)
            ep_lengths.append(ep_length)
            ep_successes.append(success)
            all_ep_frames.append(ep_frames)

    policy.train(was_training)

    # ---- Pick the best episode for video (highest success; first episode on tie) ----
    best_idx = int(np.argmax(ep_successes))   # argmax returns the *first* max index
    video_frames = all_ep_frames[best_idx]

    # ---- Scalar metrics ----
    metrics = {
        "eval/success_rate":        float(np.mean(ep_successes)),
        "eval/avg_reward":          float(np.mean(ep_returns)),
        "eval/avg_episode_length":  float(np.mean(ep_lengths)),
    }

    # ---- Video logging (best episode) ----
    if video_frames:
        # video_frames: list of (H, W, 3) uint8 arrays
        # wandb.Video expects (T, C, H, W) when passing a numpy array
        video_array = np.stack(video_frames, axis=0)          # (T, H, W, 3)
        video_array = video_array.transpose(0, 3, 1, 2)       # (T, 3, H, W)

        # Also save a local mp4 for reference
        video_dir = os.path.join(log_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, f"eval_{total_steps:07d}.mp4")
        try:
            import imageio
            imageio.mimwrite(
                video_path,
                [f.transpose(1, 2, 0) for f in video_array],  # back to (H, W, C) per frame
                fps=video_fps,
                codec="libx264",
                quality=7,
            )
            metrics["eval/video"] = wandb.Video(video_path, fps=video_fps, format="mp4")
        except Exception as e:
            # imageio not available or codec missing — fall back to raw wandb.Video from array
            print(f"[eval] mp4 save failed ({e}); uploading raw frames to W&B.")
            metrics["eval/video"] = wandb.Video(video_array, fps=video_fps, format="gif")

    print(
        f"[eval @ {total_steps}] "
        f"success={metrics['eval/success_rate']:.2f}  "
        f"avg_reward={metrics['eval/avg_reward']:.3f}  "
        f"avg_ep_len={metrics['eval/avg_episode_length']:.1f}  "
        f"video=ep{best_idx} (success={ep_successes[best_idx]:.1f})"
    )
    return metrics

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@hydra.main(config_path="conf", config_name="transformer_rl", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    log_dir = HydraConfig.get().runtime.output_dir

    torch.manual_seed(cfg.r_seed)
    np.random.seed(cfg.r_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    wandb.init(
        project=cfg.experiment.project,
        name=cfg.experiment.name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    task_id = int(cfg.sim.eval_tasks[0])
    # Lazy import so this file can be imported without LIBERO installed (useful for unit tests).
    from hw3.libero_env_fast import FastLIBEROEnv
    env = FastLIBEROEnv(task_id=task_id, max_episode_steps=cfg.sim.episode_length, cfg=cfg)
    eval_env = FastLIBEROEnv(task_id=task_id, max_episode_steps=cfg.sim.episode_length, cfg=cfg,
                              render_mode="rgb_array")
    instruction = env.instruction
    print(f"Loaded environment with instruction: {instruction}")
    action_dim  = env._action_dim

    # Load transformer policy from HW1 checkpoint
    policy   = TransformerPolicyWrapper(cfg.init_checkpoint, device, cfg)

    algorithm = cfg.rl.algorithm.lower()

    if algorithm == "ppo":
        # ------------------------------------------------------------------
        # PPO loop (reuses RolloutBuffer + ppo_update from Part 1)
        # ------------------------------------------------------------------
        # Value function: fresh GRP backbone + value_head, trained from scratch.
        value_fn = ValueFunction(policy, device, cfg)
        print("Value network: fresh GRP transformer trained from scratch")

        # Policy and value_fn own entirely independent parameters.
        # Use separate learning rates so the value network can converge faster.
        value_lr = float(cfg.value.get("learning_rate", cfg.training.learning_rate))
        optimizer = torch.optim.Adam([
            {"params": list(policy.parameters()),   "lr": cfg.training.learning_rate},
            {"params": list(value_fn.parameters()), "lr": value_lr},
        ])
        obs, info = env.reset()
        obs    = np.ascontiguousarray(obs)
        if obs.ndim != 3:
            raise ValueError(f"Expected image obs (H,W,C), got shape={obs.shape}")

        use_pose = policy.model._cfg.policy.use_pose_data
        pose = _extract_pose_from_info(info, policy, device) if use_pose else None
        pose_dim = pose.shape[-1] if pose is not None else 7

        buffer = RolloutBuffer(cfg.training.rollout_length, obs.shape, action_dim, device, pose_dim=pose_dim)
        txt_goal, goal_state = policy.encode_goals(obs, instruction)

        obs_t = torch.from_numpy(obs).float().to(device)
        print(f"Observation shape: {obs.shape}")
        print(f"Pose shape: {pose.shape if pose is not None else 'N/A (pose disabled)'}")

        total_steps = 0
        episode_returns, episode_successes = [], []
        ep_ret = 0.0

        while total_steps < cfg.training.total_env_steps:
            buffer.reset()

            # --- Rollout collection ---
            policy.eval()
            value_fn.eval()
            with torch.no_grad():
                for _ in range(cfg.training.rollout_length):
                    action_np, log_prob, _, z_sample = policy.get_action(
                        obs_t.unsqueeze(0), txt_goal, goal_state, pose
                    )
                    value = value_fn(obs_t.unsqueeze(0), txt_goal, goal_state, pose)

                    next_obs, reward, done, truncated, info = env.step(action_np)
                    ep_ret      += reward
                    total_steps += 1
                    buffer.add(
                        obs_t,
                        z_sample.detach().to(device),
                        log_prob.detach(),
                        torch.tensor(reward,               device=device),
                        value.squeeze(0),
                        torch.tensor(float(done or truncated), device=device),
                        pose,
                        goal_state=goal_state,
                        txt_goal=txt_goal,
                    )

                    if done or truncated:
                        episode_returns.append(ep_ret)
                        episode_successes.append(float(info.get("success_placed", 0.0)))
                        ep_ret  = 0.0
                        obs, info = env.reset()
                        obs = np.ascontiguousarray(obs)   # always safe after reset
                        pose = _extract_pose_from_info(info, policy, device) if use_pose else None
                        txt_goal, goal_state = policy.encode_goals(obs, instruction)
                    else:
                        obs = next_obs
                        pose = _extract_pose_from_info(info, policy, device) if use_pose else None

                    # Ensure numpy array is C-contiguous (some envs return views with
                    # negative strides) before zero-copy torch.from_numpy.
                    if not obs.flags["C_CONTIGUOUS"]:
                        obs = np.ascontiguousarray(obs)
                    obs_t = torch.from_numpy(obs).float().to(device)
                    if buffer.full():
                        break

                last_value = value_fn(obs_t.unsqueeze(0), txt_goal, goal_state, pose).squeeze(0)

            policy.train()
            value_fn.train()
            returns, advantages = buffer.compute_returns_and_advantages(
                last_value, cfg.training.gamma, cfg.training.gae_lambda
            )

            # --- LR annealing ---
            if getattr(cfg.training, "anneal_lr", False):
                frac = 1.0 - total_steps / cfg.training.total_env_steps
                base_lrs = [cfg.training.learning_rate,
                            float(cfg.value.get("learning_rate", cfg.training.learning_rate))]
                for pg, base_lr in zip(optimizer.param_groups, base_lrs):
                    pg["lr"] = base_lr * frac

            update_info = ppo_update(policy, value_fn, optimizer, buffer, returns, advantages, cfg)

            # Hard-clamp log_std parameter so it can never drift above 0.0 (std > 1.0).
            # The forward() already clamps the *value* used for the distribution, but
            # without clamping the parameter itself the Adam momentum keeps accumulating
            # and the parameter creeps upward every update, causing entropy to rise
            # monotonically even when entropy_coeff=0.
            with torch.no_grad():
                policy._action_log_std.clamp_(min=-5.0, max=0.0)

            # --- Logging ---
            if total_steps % cfg.log_interval < cfg.training.rollout_length:
                log = {"train/total_steps": total_steps,
                       **{f"train/{k}": v for k, v in update_info.items()}}
                if episode_returns:
                    log["train/episode_return"] = np.mean(episode_returns[-10:])
                    log["train/success_rate"]   = np.mean(episode_successes[-10:])
                # Log log_std mean so entropy drift is immediately visible in W&B.
                log["train/log_std_mean"] = policy._action_log_std.mean().item()
                wandb.log(log, step=total_steps)
                print(f"[PPO {total_steps}] return={log.get('train/episode_return', float('nan')):.3f} "
                      f"policy_loss={update_info['policy_loss']:.4f}  "
                      f"log_std={log['train/log_std_mean']:.3f}")

            if total_steps % cfg.eval_interval < cfg.training.rollout_length:
                eval_metrics = evaluate_policy(
                    policy, eval_env, cfg, device, total_steps, log_dir
                )
                wandb.log(eval_metrics, step=total_steps)

            # --- Periodic checkpoint ---
            if total_steps % cfg.save_interval < cfg.training.rollout_length:
                ckpt_dir = os.path.join("checkpoints", cfg.experiment.name)
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save({
                    "policy":      policy.model.state_dict(),
                    "value_fn":    value_fn.state_dict(),
                    "optimizer":   optimizer.state_dict(),
                    "total_steps": total_steps,
                    "cfg":         OmegaConf.to_container(cfg),
                }, os.path.join(ckpt_dir, f"transformer_ppo_{total_steps}.pth"))

    elif algorithm == "grpo":
        # ------------------------------------------------------------------
        # GRPO loop with ground-truth resets (Part 2c)
        # ------------------------------------------------------------------
        from libero.libero import benchmark  # lazy import — only needed for GRPO
        total_steps  = 0
        update_count = 0
        all_returns  = []

        # Frozen reference policy π_ref — loaded from the same init checkpoint
        # and never updated.  Used for the KL penalty β·KL[π_θ ‖ π_ref] in the
        # GRPO surrogate (Shao et al. 2024, eq. 9).  Disabled when kl_coef = 0.
        #
        # Algorithm 1 (iterative GRPO) distinguishes two policies:
        #   π_θ_old  — refreshed every inner step (= lp_old stored in rollout buffer)
        #   π_ref    — refreshed every outer iteration (line 3 of Algorithm 1)
        # ref_update_interval=0 means single-iteration GRPO (π_ref = BC init forever).
        # ref_update_interval=N means iterative GRPO: re-snapshot π_ref every N updates.
        optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.training.learning_rate)
        kl_coef = float(getattr(cfg.grpo, "kl_coef", 0.0))
        ref_update_interval = int(getattr(cfg.grpo, "ref_update_interval", 0))
        if kl_coef > 0.0:
            ref_policy = copy.deepcopy(policy)
            for p in ref_policy.parameters():
                p.requires_grad_(False)
            ref_policy.eval()
            print(f"[GRPO] Loaded frozen reference policy from {cfg.init_checkpoint} "
                  f"(kl_coef={kl_coef}, ref_update_interval={ref_update_interval})")
        else:
            ref_policy = None
            print("[GRPO] kl_coef=0 — KL penalty disabled")

        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[cfg.sim.task_set]()
        init_states = task_suite.get_task_init_states(task_id)
        if len(init_states) == 0:
            raise RuntimeError(f"No init states found for task_id={task_id} in task_set={cfg.sim.task_set}")

        while total_steps < cfg.training.total_env_steps:
            trajectories_per_group = []
            steps_this_update = 0

            for _group_idx in range(int(cfg.grpo.num_groups)):
                init_state_idx = np.random.randint(len(init_states))
                init_state = init_states[init_state_idx]
                group = collect_grpo_group(
                    env=env,
                    policy=policy,
                    init_state=init_state,
                    group_size=int(cfg.grpo.group_size),
                    max_steps=int(cfg.sim.episode_length),
                    device=device,
                )
                trajectories_per_group.append(group)
                steps_this_update += sum(int(traj["rewards"].shape[0]) for traj in group)

            total_steps += steps_this_update

            update_info = grpo_update(policy, optimizer,
                                      trajectories_per_group, cfg, device,
                                      ref_policy=ref_policy)
            update_count += 1

            # Iterative GRPO (Algorithm 1, line 3): re-snapshot π_ref ← π_θ every
            # ref_update_interval gradient updates.  This lets the KL penalty track
            # the *current outer-iteration* policy rather than the original BC init,
            # which matches the paper's iterative variant.
            # ref_update_interval=0 (default) keeps π_ref frozen at BC init forever.
            if (ref_policy is not None
                    and ref_update_interval > 0
                    and update_count % ref_update_interval == 0):
                ref_policy.model.load_state_dict(policy.model.state_dict())
                ref_policy._action_log_std.data.copy_(policy._action_log_std.data)
                print(f"[GRPO] Refreshed π_ref ← π_θ at update {update_count}")
            all_returns.extend([t["total_return"] for g in trajectories_per_group for t in g])

            log = {
                "train/total_steps":    total_steps,
                "train/steps_this_update": steps_this_update,
                "train/update":         update_count,
                **{f"train/{k}": v for k, v in update_info.items()},
                "train/episode_return": np.mean(all_returns[-50:]) if all_returns else 0.0,
            }
            wandb.log(log, step=total_steps)
            print(f"[GRPO {total_steps}] return={log['train/episode_return']:.3f} "
                  f"policy_loss={update_info['policy_loss']:.4f}")

            if total_steps % cfg.eval_interval < max(1, steps_this_update):
                eval_metrics = evaluate_policy(
                    policy, eval_env, cfg, device, total_steps, log_dir
                )
                wandb.log(eval_metrics, step=total_steps)

            if total_steps % cfg.save_interval < max(1, steps_this_update):
                ckpt_dir = os.path.join("checkpoints", cfg.experiment.name)
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save({
                    "policy":      policy.model.state_dict(),
                    "optimizer":   optimizer.state_dict(),
                    "total_steps": total_steps,
                    "cfg":         OmegaConf.to_container(cfg),
                }, os.path.join(ckpt_dir, f"transformer_grpo_{total_steps}.pth"))

    elif algorithm == "grpo_worldmodel":
        # ------------------------------------------------------------------
        # GRPO with DreamerV3 world model imagination (Part 2d)
        # ------------------------------------------------------------------
        import sys as _sys
        _repo_root = os.path.join(os.path.dirname(__file__), '..')
        if _repo_root not in _sys.path:
            _sys.path.insert(0, _repo_root)

        # Lazy import — hw2 only needed for this algorithm branch.
        try:
            from hw2.dreamerV3 import DreamerV3
        except ImportError:
            _sys.path.insert(0, os.path.join(_repo_root, 'hw2'))
            from dreamerV3 import DreamerV3

        wm_cfg = cfg.world_model
        wm_obs_shape = list(wm_cfg.obs_shape)   # [C, H, W]

        # Build the DreamerV3 using fields borrowed from the GRP checkpoint config
        # so we don't need to import the HW2 Hydra config hierarchy.
        grp_cfg = policy.model._cfg

        # Construct a minimal omegaconf stub with the fields DreamerV3/GRPBase needs.
        from omegaconf import OmegaConf as _OC
        wm_base_cfg = _OC.create({
            "device": str(device),
            "dataset": {
                "action_mean":  list(grp_cfg.dataset.action_mean),
                "action_std":   list(grp_cfg.dataset.action_std),
                "pose_mean":    list(grp_cfg.dataset.pose_mean)
                                if hasattr(grp_cfg.dataset, "pose_mean")
                                else [0.0] * 7,
                "pose_std":     list(grp_cfg.dataset.pose_std)
                                if hasattr(grp_cfg.dataset, "pose_std")
                                else [1.0] * 7,
                "chars_list":   list(grp_cfg.dataset.chars_list)
                                if hasattr(grp_cfg.dataset, "chars_list")
                                else [],
                "encode_with_t5": False,
            },
            "policy": {
                "action_stacking": 1,
                "use_pose_data":   1,
            },
            "max_block_size": int(getattr(grp_cfg, "max_block_size", 16)),
        })

        world_model = DreamerV3(
            obs_shape  = wm_obs_shape,
            action_dim = int(wm_cfg.action_dim),
            stoch_dim  = int(wm_cfg.stoch_dim),
            discrete_dim = int(wm_cfg.discrete_dim),
            deter_dim  = int(wm_cfg.deter_dim),
            hidden_dim = int(wm_cfg.hidden_dim),
            cfg        = wm_base_cfg,
        ).to(device)

        wm_ckpt_path = str(wm_cfg.checkpoint)
        wm_state = torch.load(wm_ckpt_path, map_location=device)
        # Support both raw state-dicts and dicts with a "model" key.
        if isinstance(wm_state, dict) and "model" in wm_state:
            wm_state = wm_state["model"]
        world_model.load_state_dict(wm_state)
        world_model.eval()
        for p in world_model.parameters():
            p.requires_grad_(False)
        print(f"[WM] Loaded DreamerV3 world model from {wm_ckpt_path}")

        optimizer   = torch.optim.Adam(policy.parameters(), lr=cfg.training.learning_rate)
        total_steps = 0
        update_count = 0
        all_returns  = []

        # Frozen reference policy for KL penalty β·KL[π_θ ‖ π_ref].
        # Disabled automatically when kl_coef=0 (no deepcopy overhead).
        kl_coef_wm = float(getattr(cfg.grpo, "kl_coef", 0.0))
        if kl_coef_wm > 0.0:
            wm_ref_policy = copy.deepcopy(policy)
            for p in wm_ref_policy.parameters():
                p.requires_grad_(False)
            wm_ref_policy.eval()
            print(f"[WM-GRPO] Loaded frozen reference policy (kl_coef={kl_coef_wm})")
        else:
            wm_ref_policy = None
            print("[WM-GRPO] kl_coef=0 — KL penalty disabled")

        wm_horizon      = int(getattr(cfg.grpo, "wm_horizon",      15))
        wm_update_every = int(getattr(cfg.grpo, "wm_update_every",  4))
        use_pose        = policy.model._cfg.policy.use_pose_data

        obs, info = env.reset()
        obs = np.ascontiguousarray(obs)
        instruction = env.instruction
        txt_goal, goal_state = policy.encode_goals(obs, instruction)

        print(f"[WM-GRPO] wm_horizon={wm_horizon}, wm_update_every={wm_update_every}")

        while total_steps < cfg.training.total_env_steps:
            # --- Collect one real env step (keeps the env alive and advances
            #     total_steps so eval/save triggers fire normally). ---
            pose = (_extract_pose_from_info(info, policy, device)
                    if use_pose else None)
            obs_t = torch.from_numpy(obs).float().to(device)
            policy.eval()
            with torch.no_grad():
                action_np, _, _, _ = policy.get_action(
                    obs_t.unsqueeze(0), txt_goal, goal_state, pose
                )
            next_obs, reward, done, truncated, info = env.step(action_np)
            total_steps += 1
            all_returns.append(float(reward))

            if done or truncated:
                obs, info = env.reset()
                obs = np.ascontiguousarray(obs)
                txt_goal, goal_state = policy.encode_goals(obs, instruction)
            else:
                obs = np.ascontiguousarray(next_obs)

            # --- World-model GRPO update every wm_update_every steps ---
            if total_steps % wm_update_every == 0:
                policy.train()
                update_info = grpo_worldmodel_update(
                    policy        = policy,
                    world_model   = world_model,
                    current_obs   = obs,
                    instruction   = instruction,
                    group_size    = int(cfg.grpo.group_size),
                    horizon       = wm_horizon,
                    cfg           = cfg,
                    device        = device,
                    policy_optimizer = optimizer,
                    ref_policy    = wm_ref_policy,
                )
                update_count += 1

                if total_steps % cfg.log_interval < wm_update_every:
                    log = {
                        "train/total_steps":         total_steps,
                        "train/update":              update_count,
                        "train/episode_return":      float(np.mean(all_returns[-50:]))
                                                     if all_returns else 0.0,
                        **{f"train/{k}": v for k, v in update_info.items()},
                    }
                    wandb.log(log, step=total_steps)
                    print(
                        f"[WM-GRPO {total_steps}] "
                        f"return={log['train/episode_return']:.3f}  "
                        f"policy_loss={update_info['policy_loss']:.4f}  "
                        f"imagined_steps={update_info['imagined_steps']}"
                    )

            if total_steps % cfg.eval_interval < wm_update_every:
                eval_metrics = evaluate_policy(
                    policy, eval_env, cfg, device, total_steps, log_dir
                )
                wandb.log(eval_metrics, step=total_steps)

            if total_steps % cfg.save_interval < wm_update_every:
                ckpt_dir = os.path.join("checkpoints", cfg.experiment.name)
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save({
                    "policy":      policy.model.state_dict(),
                    "optimizer":   optimizer.state_dict(),
                    "total_steps": total_steps,
                    "cfg":         OmegaConf.to_container(cfg),
                }, os.path.join(ckpt_dir, f"transformer_grpo_wm_{total_steps}.pth"))

    else:
        raise ValueError(
            f"Unknown rl.algorithm: {algorithm!r}. "
            "Choose 'ppo', 'grpo', or 'grpo_worldmodel'."
        )

    # --- Final checkpoint ---
    ckpt_dir = os.path.join("checkpoints", cfg.experiment.name)
    os.makedirs(ckpt_dir, exist_ok=True)
    if algorithm == "ppo":
        torch.save({
            "policy":      policy.model.state_dict(),
            "value_fn":    value_fn.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "total_steps": total_steps,
            "cfg":         OmegaConf.to_container(cfg),
        }, os.path.join(ckpt_dir, f"transformer_ppo_final.pth"))
    elif algorithm == "grpo":
        torch.save({
            "policy":      policy.model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "total_steps": total_steps,
            "cfg":         OmegaConf.to_container(cfg),
        }, os.path.join(ckpt_dir, f"transformer_grpo_final.pth"))
    elif algorithm == "grpo_worldmodel":
        torch.save({
            "policy":      policy.model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "total_steps": total_steps,
            "cfg":         OmegaConf.to_container(cfg),
        }, os.path.join(ckpt_dir, f"transformer_grpo_wm_final.pth"))
    env.close()
    eval_env.close()
    wandb.finish()


if __name__ == "__main__":
    main()
