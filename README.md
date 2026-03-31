# HW3 Submission — Making a Generalist Robotics Policy with Reinforcement Learning

## Repository layout

```
HW3_submission/
├── data/
│   ├── hw3_dense_ppo_seed0/       # Part 1 — Dense PPO seed 0
│   ├── hw3_dense_ppo_seed1/       # Part 1 — Dense PPO seed 1
│   ├── hw3_dense_ppo_seed2/       # Part 1 — Dense PPO seed 2
│   ├── hw3_transformer_grpo/      # Part 2c — Transformer GRPO (ground-truth resets)
│   └── hw3_transformer_ppo/       # Part 2a — Transformer PPO (placeholder)
└── mini-grp/                      # All training and evaluation scripts
    ├── train_dense_rl.py
    ├── train_transformer_rl.py
    ├── train_dagger.py
    ├── grp_model.py
    ├── networks.py
    ├── dreamerV3.py
    ├── libero_env_fast.py
    └── sim_eval.py
```

Each `data/<run>/` folder contains:
- `.hydra/` — the exact Hydra config and CLI overrides used for that run
- `train_*.log` — stdout/stderr from training
- `videos/` — periodic evaluation rollout recordings
- `wandb/` — W&B run data

---

## Environment setup

All scripts must be run from the `mini-grp/` directory (or an equivalent copy of it) with the `hw2` conda environment activated.

```bash
conda activate hw2
cd mini-grp
```

Required environment variables (MuJoCo headless rendering):

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

---

## Part 1 — Dense PPO (LIBERO-Spatial task 9)

Three independent seeds were trained to 2 M environment steps on **LIBERO-Spatial task 9** using an MLP actor-critic.

### Reproduce seed 0

```bash
python train_dense_rl.py \
    experiment.name=hw3_dense_ppo_seed0 \
    r_seed=0 \
    sim.task_set=libero_spatial \
    sim.eval_tasks=[9] \
    sim.reward_version=standard \
    sim.reward_scale=0.1 \
    training.total_env_steps=2000000 \
    training.rollout_length=4096 \
    training.ppo_epochs=10 \
    training.minibatch_size=256 \
    training.learning_rate=3e-4
```

### Reproduce seed 1

```bash
python train_dense_rl.py \
    experiment.name=hw3_dense_ppo_seed1 \
    r_seed=1 \
    sim.task_set=libero_spatial \
    sim.eval_tasks=[9] \
    sim.reward_scale=0.1 \
    training.total_env_steps=2000000 \
    training.rollout_length=4096 \
    training.ppo_epochs=10 \
    training.minibatch_size=256 \
    training.learning_rate=3e-4
```

### Reproduce seed 2

```bash
python train_dense_rl.py \
    experiment.name=hw3_dense_ppo_seed2 \
    r_seed=2 \
    sim.task_set=libero_spatial \
    sim.eval_tasks=[9] \
    sim.reward_scale=0.1 \
    training.total_env_steps=2000000 \
    training.rollout_length=4096 \
    training.ppo_epochs=10 \
    training.minibatch_size=256 \
    training.learning_rate=3e-4
```

### Key hyperparameters (Part 1)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `sim.reward_scale` | `0.1` | Scales raw LIBERO rewards |
| `training.rollout_length` | `4096` | Steps per PPO update |
| `training.ppo_epochs` | `10` | Gradient passes per rollout |
| `training.clip_eps` | `0.2` | PPO clipping ε |
| `training.value_clip_eps` | `10.0` | Critic clip range |
| `training.entropy_coeff` | `0.001` | Entropy bonus |
| `training.target_kl` | `0` | Early stopping disabled for dense policy |
| `training.learning_rate` | `3e-4` | Adam LR |
| `policy.obs_dim` | `13` | 7 proprio + 6 relative object pose |
| `policy.hidden_dim` | `256` | MLP width |
| `policy.n_layers` | `3` | MLP depth |

### Checkpoint format

Checkpoints are written to `outputs/<date>/<time>/checkpoints/dense_ppo_step<N>.pth`.  
Each file contains the keys: `policy`, `value`, `optimizer`, `step`, `cfg`.

---

## Part 2 — Transformer RL Fine-tuning (LIBERO-Spatial task 0)

All Part 2 runs initialise from the HW1 GRP checkpoint (`checkpoints/grp-ver2/miniGRP.pth`) and target **LIBERO-Spatial task 0**.

### Part 2a — PPO fine-tuning

> **Note:** The `data/hw3_transformer_ppo/` folder is a placeholder. The PPO fine-tuning run was cut short due to training instability (policy collapse at ~90 k steps). Fixes are in progress.

To reproduce:

```bash
python train_transformer_rl.py \
    experiment.name=hw3_transformer_ppo_seed0 \
    r_seed=0 \
    init_checkpoint=checkpoints/grp-ver2/miniGRP.pth \
    rl.algorithm=ppo \
    sim.task_set=libero_spatial \
    sim.eval_tasks=[0] \
    sim.reward_scale=0.1 \
    training.total_env_steps=200000 \
    training.rollout_length=4096 \
    training.ppo_epochs=5 \
    training.minibatch_size=128
```

### Part 2c — GRPO with ground-truth resets

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train_transformer_rl.py \
    experiment.name=hw3_transformer_grpo_seed0 \
    r_seed=0 \
    init_checkpoint=checkpoints/grp-ver2/miniGRP.pth \
    rl.algorithm=grpo \
    sim.task_set=libero_spatial \
    sim.eval_tasks=[0] \
    sim.reward_scale=0.1 \
    training.total_env_steps=200000 \
    grpo.chunk_size=32 \
    grpo.num_groups=4 \
    grpo.group_size=8
```

> **VRAM note:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is required for GRPO.  
> If you see OOM, reduce `grpo.chunk_size` (e.g. `16`) or `grpo.group_size`.

### Key hyperparameters (Part 2)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `training.learning_rate` | `3e-5` | Small LR for fine-tuning a pretrained transformer |
| `training.clip_eps` | `0.1` | Tighter PPO clip to protect HW1 representations |
| `training.entropy_coeff` | `0.0` | Disabled — non-zero nudges `log_std` up and causes collapse |
| `training.target_kl` | `0.01` | Early-stop PPO epochs when KL exceeds this |
| `training.value_clip_eps` | `10.0` | Separate, larger clip for the value head |
| `training.anneal_lr` | `true` | Linear LR decay over `total_env_steps` |
| `grpo.chunk_size` | `32` | Max steps per backward chunk (reduce to `16` if OOM) |
| `grpo.kl_coef` | `0.04` | β for KL(π_θ ‖ π_ref) penalty in GRPO |
| `grpo.num_groups` | `4` | Number of episode groups per GRPO update |
| `grpo.group_size` | `8` | Episodes per group |

### Checkpoint format

PPO transformer checkpoints are saved to `checkpoints/<experiment.name>/transformer_ppo_<step>.pth`  
and contain: `policy`, `value_fn`, `optimizer`, `total_steps`, `cfg`.

GRPO checkpoints are saved to `checkpoints/<experiment.name>/transformer_grpo_<step>.pth`  
and contain: `policy`, `optimizer`, `total_steps`, `cfg`.

---

## Standalone Evaluation

`sim_eval.py` accepts any checkpoint in HW1-style raw pickle format **or** an HW3 dict with a `policy` key.

```bash
# HW1-style raw pickle
python sim_eval.py \
    simEval=[libero_fast] \
    sim.eval_tasks=[9]

# HW3-style PPO/GRPO checkpoint
python sim_eval.py \
    checkpoint=/path/to/transformer_grpo_200000.pth \
    simEval=[libero_fast] \
    sim.eval_tasks=[0]
```

