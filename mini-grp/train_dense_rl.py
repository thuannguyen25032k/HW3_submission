"""
HW3 Part 1: Train a dense MLP policy with PPO on privileged state.

Usage:
    python hw3/train_dense_rl.py \
        experiment.name=hw3_dense_ppo_seed0 \
        r_seed=0 \
        sim.task_set=libero_spatial \
        sim.eval_tasks=[9] \
        training.total_env_steps=200000 \
        training.rollout_length=128 \
        training.ppo_epochs=10 \
        training.minibatch_size=256
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pathlib import Path
os.environ["MUJOCO_GL"] = "egl"
# Ensure the vendored LIBERO package is importable even if it hasn't been pip-installed.
# Hydra may change the working directory, so we resolve relative to this file.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_LIBERO_ROOT = _REPO_ROOT / "LIBERO"
if _LIBERO_ROOT.exists():
    sys.path.insert(0, str(_LIBERO_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from hw3.libero_env_fast import FastLIBEROEnv


# ---------------------------------------------------------------------------
# Policy and value networks
# ---------------------------------------------------------------------------

class DensePolicy(nn.Module):
    """
    MLP policy that maps privileged state observations to action distributions.
    Outputs a Gaussian distribution (mean + log_std) over the 7-DoF action space.
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, n_layers: int = 5):
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        layers = []
        in_dim = obs_dim
        for _ in range(n_layers):
            module = nn.Linear(in_dim, hidden_dim)
            # Orthogonal init for stability
            nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('tanh'))
            nn.init.constant_(module.bias, 0)
            
            layers.append(module)
            layers.append(nn.Tanh())
            in_dim = hidden_dim

        self.net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        # Mean head usually initialized with gain 0.01 for smaller initial actions.
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)

        # PPO-friendly default: state-independent log std (one per action dim).
        # This is widely used and tends to be more stable than predicting std from obs.
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        # nn.init.constant_(self.log_std, -0.5)  # Start with std ~ exp(-0.5) ~ 0.6 for reasonable initial exploration.
        # Numerical safety bounds for std; can be overridden later if needed.
        self.log_std_min = -5.0
        self.log_std_max = 1.0

    def forward(self, obs: torch.Tensor):
        """
        Args:
            obs: (B, obs_dim) float tensor
        Returns:
            dist: torch.distributions.Normal over the pre-tanh Gaussian (used for log_prob)
        """
        h = self.net(obs)
        mean = self.mean_head(h)   # unbounded pre-tanh mean

        # Expand (action_dim,) -> (B, action_dim) without allocating new storage.
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp().expand_as(mean)
        # Return the base Gaussian. tanh squashing + log-prob correction is handled
        # in get_action() and ppo_update() via _tanh_log_prob().
        return Normal(mean, std)

    @staticmethod
    def _tanh_log_prob(dist: Normal, pre_tanh_action: torch.Tensor) -> torch.Tensor:
        """Log-prob of a tanh-squashed action with Jacobian correction.

        For action = tanh(u), u ~ Normal:
            log π(a) = log p(u) - sum log(1 - tanh²(u))
        This is the standard SAC/PPO-with-tanh correction.
        """
        log_prob = dist.log_prob(pre_tanh_action).sum(-1)
        # Jacobian correction: -sum log(1 - tanh(u)^2) = -sum log(sech^2(u))
        # Numerically stable form: 2*(log2 - u - softplus(-2u))
        correction = 2.0 * (
            torch.log(torch.tensor(2.0, device=pre_tanh_action.device))
            - pre_tanh_action
            - torch.nn.functional.softplus(-2.0 * pre_tanh_action)
        ).sum(-1)
        return log_prob - correction

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """Sample a tanh-squashed action and return (action_np, log_prob, entropy).

        The pre-tanh sample u is stored in the buffer (via the action field),
        so that ppo_update can recompute log_prob(tanh(u)) consistently.
        """
        dist = self.forward(obs)
        if deterministic:
            pre_tanh = dist.mean
        else:
            pre_tanh = dist.rsample()
        log_prob = self._tanh_log_prob(dist, pre_tanh)
        entropy  = dist.entropy().sum(-1)
        action   = torch.tanh(pre_tanh)          # squash to (-1, 1)
        # Return the PRE-TANH sample so the buffer stores it — allows exact
        # log_prob recomputation in ppo_update without any clamp distortion.
        return action.squeeze(0).detach().cpu().numpy(), log_prob.squeeze(0), entropy.squeeze(0), pre_tanh.squeeze(0)


class DenseValueFunction(nn.Module):
    """MLP value function V(s) for PPO critic."""
    def __init__(self, obs_dim: int, hidden_dim: int = 256, n_layers: int = 5):
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        layers = []
        in_dim = obs_dim
        for _ in range(n_layers):
            module = nn.Linear(in_dim, hidden_dim)
            # Orthogonal init for stability
            nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('tanh'))
            nn.init.constant_(module.bias, 0)
            
            layers.append(module)
            layers.append(nn.Tanh())
            in_dim = hidden_dim

        self.net = nn.Sequential(*layers)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Value head usually initialized with gain 1.0
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns scalar value estimate of shape (B,)."""
        return self.value_head(self.net(obs)).squeeze(-1)


# ---------------------------------------------------------------------------
# PPO rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Stores a fixed-length on-policy rollout for PPO updates."""

    def __init__(self, rollout_length: int, obs_dim: int | tuple, action_dim: int, device: torch.device, pose_dim: int = 7):
        self.rollout_length = rollout_length
        self.device = device
        # Support both vector observations (obs_dim=int) and image observations (obs_dim=tuple like (C,H,W) or (H,W,C)).
        if isinstance(obs_dim, int):
            obs_shape = (obs_dim,)
        else:
            obs_shape = tuple(int(x) for x in obs_dim)
        self.obs_shape = obs_shape
        self.obs = torch.zeros((rollout_length, *obs_shape), device=device)
        self.actions = torch.zeros(rollout_length, action_dim, device=device)
        # pose_dim matches the encoded pose embedding dimension from backbone.model.encode_pose()
        self.poses = torch.zeros(rollout_length, pose_dim, device=device)
        self.log_probs = torch.zeros(rollout_length, device=device)
        self.rewards = torch.zeros(rollout_length, device=device)
        self.values = torch.zeros(rollout_length, device=device)
        self.dones = torch.zeros(rollout_length, device=device)
        self.ptr = 0
        # Optional: per-step goal image for transformer policies that change goal
        # conditioning each episode.  Shape (rollout_length, *obs_shape).
        # Allocated on first use (add() kwarg goal_state != None) so dense policies
        # pay zero memory cost.
        self.goal_states: torch.Tensor | None = None
        # txt_goal is the same across all episodes of one task; store once.
        self.txt_goal: torch.Tensor | None = None

    def add(self, obs, action, log_prob, reward, value, done, pose=None,
            goal_state: torch.Tensor | None = None,
            txt_goal: torch.Tensor | None = None):
        # Allow obs to be numpy or torch, and ensure it matches the buffer shape.
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=self.obs.dtype, device=self.device)
        else:
            obs = obs.to(device=self.device)

        if pose is not None:
            pose = torch.as_tensor(pose, dtype=self.poses.dtype, device=self.device)
            self.poses[self.ptr] = pose
        # If pose is None, leave self.poses[self.ptr] as the pre-allocated zeros.
        self.obs[self.ptr].copy_(obs)
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = done

        # Store per-step goal image for transformer policies (lazy allocation).
        if goal_state is not None:
            if self.goal_states is None:
                self.goal_states = torch.zeros(
                    (self.rollout_length, *self.obs_shape), dtype=torch.float32, device=self.device
                )
            self.goal_states[self.ptr].copy_(
                goal_state.to(self.device).squeeze(0) if goal_state.dim() == 4 else goal_state.to(self.device)
            )
        if txt_goal is not None:
            # Only update on first call or on episode transitions — always keep fresh.
            self.txt_goal = txt_goal.to(self.device)

        self.ptr += 1

    def full(self):
        return self.ptr >= self.rollout_length

    def reset(self):
        self.ptr = 0
        self.goal_states = None
        self.txt_goal    = None

    def compute_returns_and_advantages(self, last_value: torch.Tensor, gamma: float, gae_lambda: float):
        """
        Compute discounted returns and GAE advantages.

        Args:
            last_value: bootstrap value V(s_T) of shape ()
            gamma: discount factor
            gae_lambda: GAE lambda
        Returns:
            returns: (rollout_length,) tensor
            advantages: (rollout_length,) tensor
        """
        # Compute GAE advantages.
        # Returns are then R_t = A_t + V(s_t).
        returns = torch.zeros_like(self.rewards)
        advantages = torch.zeros_like(self.rewards)

        # Ensure scalar for bootstrap value.
        next_value = last_value.squeeze()
        next_adv = torch.zeros((), device=self.device)

        for t in reversed(range(self.rollout_length)):
            not_done = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * not_done - self.values[t]
            next_adv = delta + gamma * gae_lambda * not_done * next_adv
            advantages[t] = next_adv
            returns[t] = advantages[t] + self.values[t]
            next_value = self.values[t]

        return returns, advantages


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(policy: DensePolicy,
               value_fn: DenseValueFunction,
               optimizer: torch.optim.Optimizer,
               buffer: RolloutBuffer,
               returns: torch.Tensor,
               advantages: torch.Tensor,
               cfg: DictConfig):
    """
    Perform `ppo_epochs` passes of minibatch PPO updates on the stored rollout.

    Returns a dict of mean losses for logging.
    """
    # For transformer policies the backbone has dropout=0.2 (GRP checkpoint).
    # Importance ratios r = π_new / π_old MUST be computed with dropout disabled:
    # if dropout is ON during recomputation, new_logp differs from old_logp even
    # for identical inputs → ratio is random noise → clip_frac ≈ 80-90% →
    # policy gradient is pure noise → collapse within 50k steps.
    # policy.eval() disables dropout while leaving requires_grad=True so
    # gradients still flow through the transformer for the policy loss.
    # value_fn is a fresh network trained from scratch; keep it in train() mode
    # so its own dropout acts as regularization during critic training.
    _policy_was_training = policy.training
    if hasattr(policy, "encode_goals"):   # TransformerPolicyWrapper
        policy.eval()
    def _all_finite(module: nn.Module) -> bool:
        for p in module.parameters():
            if p is None:
                continue
            if not torch.isfinite(p).all():
                return False
        return True

    # Flatten rollout (T, ...) -> (N, ...)
    obs          = buffer.obs.detach()
    actions      = buffer.actions.detach()
    poses        = buffer.poses.detach() if buffer.poses is not None else None
    old_log_probs = buffer.log_probs.detach()
    returns      = returns.detach()
    advantages   = advantages.detach()
    # Per-step goal conditioning stored by transformer policies (None for dense).
    goal_states  = buffer.goal_states.detach() if buffer.goal_states is not None else None
    txt_goal_buf = buffer.txt_goal.detach()     if buffer.txt_goal    is not None else None
    # Detect whether the policy supports explicit goal arguments
    # (TransformerPolicyWrapper.forward accepts txt_goal and goal_state).
    is_transformer_policy = hasattr(policy, "encode_goals")

    # # PPO commonly normalizes advantages per update batch.
    # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Global advantage normalization before the minibatch loop: ensures
    # consistent gradient scale and sign across all minibatches in an epoch.
    # Per-minibatch normalization (done below) is kept for within-batch stability,
    # but without global normalization the sign of advantages can flip between
    # minibatches when a rollout contains a mix of positive and negative returns.
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    n = obs.shape[0]
    mb = int(cfg.training.minibatch_size)
    if mb <= 0:
        raise ValueError(f"training.minibatch_size must be > 0, got {mb}")
    if mb > n:
        mb = n

    # Support both naming conventions used across configs.
    clip_eps = float(getattr(cfg.training, "clip_epsilon", getattr(cfg.training, "clip_eps")))
    value_coef = float(getattr(cfg.training, "value_coef", getattr(cfg.training, "value_coeff")))
    entropy_coef = float(getattr(cfg.training, "entropy_coef", getattr(cfg.training, "entropy_coeff")))
    max_grad_norm = float(getattr(cfg.training, "max_grad_norm", 0.5))
    # Value clipping uses a separate, larger epsilon than the policy clip.
    # Value targets can be in the hundreds; policy clip_eps (0.2) would freeze the critic.
    value_clip_eps = float(getattr(cfg.training, "value_clip_eps", 10.0))
    # Recommended target_kl: 0.01 to 0.015 for stability
    target_kl = float(getattr(cfg.training, "target_kl", 0.0))

    # For logging
    policy_losses = []
    value_losses = []
    entropies = []
    approx_kls = []
    clip_fracs = []

    for _epoch in range(int(cfg.training.ppo_epochs)):
        epoch_kls = [] # Track KLs for this specific epoch
        perm = torch.randperm(n, device=obs.device)
        for start in range(0, n, mb):
            idx = perm[start:start + mb]

            # 1. Minibatch Advantage Normalization
            mb_adv = advantages[idx]
            mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std(unbiased=False) + 1e-8)

            # 1b. For transformer policies: pass explicit per-step goal conditioning.
            # goal_states[idx] has shape (mb, H, W, C) — one goal per rollout step.
            # We pass each observation's own goal_state directly to forward() by
            # batch-expanding it from (mb, H, W, C) to match the obs batch.
            # This is exact even when the rollout spans multiple episodes with
            # different goal images (goal re-encoded on every episode reset).
            if is_transformer_policy and goal_states is not None:
                mb_goal_state = goal_states[idx]                     # (mb, H, W, C)
                mb_txt_goal   = txt_goal_buf                         # (1, T, ...)
            else:
                mb_goal_state = None
                mb_txt_goal   = None

            # 2. Re-calculate distribution and values
            # actions[idx] are z-score samples stored in the buffer.
            # Recompute the log-prob so the importance ratio is exact.
            if is_transformer_policy and mb_txt_goal is not None:
                dist = policy.forward(obs[idx], mb_txt_goal, mb_goal_state,
                                      pose=poses[idx] if poses is not None else None)
                values_pred = value_fn(obs[idx], mb_txt_goal, mb_goal_state,
                                       pose=poses[idx] if poses is not None else None)
            else:
                dist = policy.forward(obs[idx])
                values_pred = value_fn(obs[idx])
            # Use _log_prob (new name) with _tanh_log_prob as fallback for DensePolicy.
            new_logp = policy._log_prob(dist, actions[idx]) if hasattr(policy, "_log_prob") else policy._tanh_log_prob(dist, actions[idx])
            mb_old_logp = old_log_probs[idx]
            
            # 3. Clip the value to reduce variability during Critic training and compute value loss.
            # 3.a. Get the value from the rollout (stored in buffer)
            mb_old_values = buffer.values[idx]
            # 3.b. Calculate the "unclipped" loss
            mb_returns = returns[idx]
            value_loss_unclipped = (values_pred - mb_returns)**2
            # 3.c. Calculate the "clipped" loss — use value_clip_eps, NOT policy clip_eps.
            # Policy clip_eps (0.2) is far too small for value targets in the range [0, ~2700];
            # it would freeze the critic. value_clip_eps defaults to 10.0.
            values_clipped = mb_old_values + torch.clamp(values_pred - mb_old_values, -value_clip_eps, value_clip_eps)
            value_loss_clipped = (values_clipped - mb_returns)**2
            # 3.d. Take the maximum of the two (pessimistic bound)
            v_loss_max = torch.max(value_loss_unclipped, value_loss_clipped)
            value_loss = 0.5 * v_loss_max.mean()
            
            # 4. Policy loss with clipped surrogate objective.
            ratio = torch.exp(new_logp - mb_old_logp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 5. Entropy bonus for exploration
            entropy = dist.entropy().sum(-1).mean()
            
            # 6. Total loss
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Abort-like behavior if gradients explode to NaN/Inf.
            grads_finite = True
            for p in list(policy.parameters()) + list(value_fn.parameters()):
                if p.grad is None:
                    continue
                if not torch.isfinite(p.grad).all():
                    grads_finite = False
                    break

            if not grads_finite:
                print("[ppo_update] Non-finite gradients detected; skipping optimizer step for this minibatch.")
                optimizer.zero_grad(set_to_none=True)
                continue

            nn.utils.clip_grad_norm_(list(policy.parameters()) + list(value_fn.parameters()), max_grad_norm)
            optimizer.step()

            # If params become non-finite (rare but possible), stop early to prevent poisoning the run.
            if not _all_finite(policy) or not _all_finite(value_fn):
                raise FloatingPointError("Non-finite parameters detected after optimizer step.")

            # Logging stats
            with torch.no_grad():
                # Approximate KL: E[log π_old - log π_new]  (reverse KL, matches OpenAI baselines)
                approx_kl = (mb_old_logp - new_logp).mean()
                clipped    = (torch.abs(ratio - 1.0) > clip_eps).float().mean()
                epoch_kls.append(approx_kl.item())
            policy_losses.append(policy_loss.detach())
            value_losses.append(value_loss.detach())
            entropies.append(entropy.detach())
            approx_kls.append(approx_kl.detach())
            clip_fracs.append(clipped.detach())
        # --- Check for Early Stopping after the epoch's minibatches ---
        if target_kl > 0.0:  # Only check if target_kl is set to a positive value
            avg_epoch_kl = np.mean(epoch_kls)
            if avg_epoch_kl > target_kl:
                print(f"Early stopping at epoch {_epoch} due to reaching target KL: {avg_epoch_kl:.4f} > {target_kl}")
                break

    def _mean(xs):
        if not xs:
            return 0.0
        return torch.stack(list(xs)).mean().item()

    # Restore policy training mode
    policy.train(_policy_was_training)

    return {
        "policy_loss": _mean(policy_losses),
        "value_loss": _mean(value_losses),
        "entropy": _mean(entropies),
        "approx_kl": _mean(approx_kls),
        "clip_frac": _mean(clip_fracs),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_policy(
    policy: DensePolicy,
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
    render_size   = int(getattr(cfg.sim, "fast_env_image_size", 256))

    ep_returns    : list[float]             = []
    ep_lengths    : list[int]               = []
    ep_successes  : list[float]             = []
    all_ep_frames : list[list[np.ndarray]]  = []   # one list of frames per episode

    was_training = policy.training
    policy.eval()

    with torch.no_grad():
        for _ in range(n_episodes):
            obs, _ = eval_env.reset()
            ep_return = 0.0
            ep_length = 0
            success   = 0.0
            ep_frames : list[np.ndarray] = []

            for _ in range(max_ep_steps):
                # Deterministic action (use mean of the Gaussian)
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action_np, _, _, _ = policy.get_action(obs_t, deterministic=True)

                obs, reward, done, truncated, info = eval_env.step(action_np)

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

@hydra.main(config_path="conf", config_name="dense_ppo", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.r_seed)
    np.random.seed(cfg.r_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # --- Logging ---
    wandb.init(
        project=cfg.experiment.project,
        name=cfg.experiment.name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # --- Environment ---
    task_id = int(cfg.sim.eval_tasks[0])
    env = FastLIBEROEnv(
        task_id=task_id,
        max_episode_steps=cfg.sim.episode_length,
        cfg=cfg,
    )

    eval_env = FastLIBEROEnv(
        task_id=task_id,
        max_episode_steps=cfg.sim.episode_length,
        cfg=cfg,
        render_mode="rgb_array",
    )
    
    obs_dim = cfg.policy.obs_dim
    action_dim = cfg.policy.action_dim

    # --- Models ---
    policy = DensePolicy(obs_dim, action_dim, cfg.policy.hidden_dim, cfg.policy.n_layers).to(device)
    value_fn = DenseValueFunction(obs_dim, cfg.policy.hidden_dim, cfg.policy.n_layers).to(device)

    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value_fn.parameters()),
        lr=cfg.training.learning_rate,
    )

    buffer = RolloutBuffer(cfg.training.rollout_length, obs_dim, action_dim, device)

    # --- Rollout state ---
    obs, _ = env.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
    episode_return  = 0.0
    total_steps     = 0
    episode_returns = []
    episode_successes = []

    # --- Main loop ---
    while total_steps < cfg.training.total_env_steps:
        buffer.reset()

        # Collect one rollout
        with torch.no_grad():
            for _ in range(cfg.training.rollout_length):
                # If env ever returns NaNs/Infs, reset to avoid poisoning training.
                if not torch.isfinite(obs_tensor).all():
                    print("[train] Non-finite observation detected; resetting env.")
                    obs, _ = env.reset()
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
                    episode_return = 0.0
                    episode_steps = 0

                # get_action returns (action_np, log_prob, entropy, pre_tanh_action).
                # Store pre_tanh in the buffer so ppo_update can recompute the exact
                # tanh-corrected log-prob without clamp distortion.
                action_np, log_prob, _, pre_tanh = policy.get_action(obs_tensor.unsqueeze(0))
                value = value_fn(obs_tensor.unsqueeze(0))

                next_obs, reward, done, truncated, info = env.step(action_np)

                episode_return += reward
                total_steps += 1

                buffer.add(
                    obs_tensor,
                    pre_tanh.to(device),       # store pre-tanh action for exact log_prob recomputation
                    log_prob.to(device),
                    torch.tensor(reward, device=device),
                    value.squeeze(0),
                    torch.tensor(float(done or truncated), device=device),
                )

                if done or truncated:
                    episode_returns.append(episode_return)
                    episode_successes.append(float(info.get("success_placed", 0.0)))
                    episode_return = 0.0
                    obs, _ = env.reset()
                else:
                    obs = next_obs
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)

                if buffer.full():
                    break

            # Bootstrap last value
            last_value = value_fn(obs_tensor.unsqueeze(0)).squeeze(0)

        returns, advantages = buffer.compute_returns_and_advantages(
            last_value, cfg.training.gamma, cfg.training.gae_lambda
        )
        # Annealing the rate if instructed to do so.
        if cfg.training.anneal_lr:
            for param_group in optimizer.param_groups:
                param_group["lr"] = cfg.training.learning_rate * (1 - total_steps / cfg.training.total_env_steps)

        # PPO update
        update_info = ppo_update(policy, value_fn, optimizer, buffer, returns, advantages, cfg)

        # Logging
        if total_steps % cfg.log_interval < cfg.training.rollout_length:
            log_dict = {
                "train/total_steps": total_steps,
                **{f"train/{k}": v for k, v in update_info.items()},
            }
            if episode_returns:
                log_dict["train/episode_return"] = np.mean(episode_returns[-10:])
                log_dict["train/success_rate"] = np.mean(episode_successes[-10:])
            wandb.log(log_dict, step=total_steps)
            print(f"[{total_steps}/{cfg.training.total_env_steps}] "
                  f"return={log_dict.get('train/episode_return', float('nan')):.3f} "
                  f"policy_loss={update_info['policy_loss']:.4f}")

        # Periodic evaluation
        if total_steps % cfg.eval_interval < cfg.training.rollout_length:
            eval_metrics = evaluate_policy(
                policy, eval_env, cfg, device, total_steps, log_dir
            )
            wandb.log(eval_metrics, step=total_steps)
        
        # Checkpoint
        if total_steps % cfg.save_interval < cfg.training.rollout_length:
            ckpt = {
                "policy": policy.state_dict(),
                "value_fn": value_fn.state_dict(),
                "optimizer": optimizer.state_dict(),
                "total_steps": total_steps,
                "cfg": OmegaConf.to_container(cfg),
            }
            # Make a directory for checkpoints if it doesn't exist.
            os.makedirs(os.path.join("checkpoints", cfg.experiment.name), exist_ok=True)
            torch.save(ckpt, os.path.join("checkpoints", cfg.experiment.name, f"dense_ppo_{total_steps}.pth"))

    # Final evaluation
    print("[train] Running final evaluation...")
    final_eval_metrics = evaluate_policy(
        policy, eval_env, cfg, device, total_steps, log_dir
    )
    wandb.log(final_eval_metrics, step=total_steps)

    # Final save
    torch.save({"policy": policy.state_dict(), "cfg": OmegaConf.to_container(cfg)},
               os.path.join("checkpoints", cfg.experiment.name, "dense_ppo_final.pth"))
    env.close()
    eval_env.close()
    wandb.finish()


if __name__ == "__main__":
    main()
