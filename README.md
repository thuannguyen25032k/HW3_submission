# HW3 Submission — Making a Generalist Robotics Policy with Reinforcement Learning

## Repository layout

```
HW3_submission/
├── data/
│   ├── hw3_dense_ppo_seed0/       # Part 1 — Dense PPO seed 0
│   ├── hw3_dense_ppo_seed1/       # Part 1 — Dense PPO seed 1
│   ├── hw3_dense_ppo_seed2/       # Part 1 — Dense PPO seed 2
│   ├── hw3_transformer_grpo/      # Part 2c — Transformer GRPO (ground-truth resets)
│   ├── hw3_transformer_ppo/       # Part 2a — Transformer PPO (placeholder)
│   └── hw3_dagger_seed0/          # Part 3 — DAgger distillation
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
conda activate roble
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

## Part 3 — DAgger (LIBERO-Spatial task 9)

The transformer student is initialised from the HW1 GRP checkpoint and distilled from the best Part 1 dense PPO teacher (`hw3_dense_ppo_seed2`) on **LIBERO-Spatial task 9** using DAgger.

### Reproduce

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
python train_dagger.py \
    experiment.name=hw3_dagger_seed0 \
    r_seed=0 \
    teacher_checkpoint="checkpoints/hw3_dense_ppo_seed2/dense_ppo_final.pth" \
    student_init_checkpoint="checkpoints/grp-ver2/miniGRP.pth" \
    sim.task_set=libero_spatial \
    sim.eval_tasks=[9] \
    sim.episode_length=300 \
    dagger.num_rounds=120 \
    dagger.rollouts_per_round=120 \
    dagger.bc_epochs_per_round=80 \
    dagger.beta_schedule=linear \
    dagger.beta_init=1.0 \
    dagger.dataset_save_dir="dagger_datasets/seed0" \
    training.learning_rate=1e-4 \
    training.minibatch_size=64 \
    training.max_grad_norm=0.5 \
    eval_interval=1 \
    save_interval=5
```

### Key hyperparameters (Part 3)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `sim.fast_env_image_size` | `64` | Image obs size fed to the student (must match checkpoint config) |
| `sim.video_render_size` | `256` | Resolution for evaluation video rendering (independent of obs size) |
| `dagger.num_rounds` | `120` | DAgger iterations |
| `dagger.rollouts_per_round` | `120` | Rollouts collected per round |
| `dagger.bc_epochs_per_round` | `80` | Supervised BC epochs over aggregated dataset per round |
| `dagger.beta_schedule` | `linear` | Beta decays from `1.0` (pure teacher) → `0.0` (pure student) |
| `dagger.beta_init` | `1.0` | Initial teacher mixing coefficient |
| `training.learning_rate` | `1e-4` | Adam LR for BC updates |
| `training.minibatch_size` | `64` | Mini-batch size for BC |
| `training.max_grad_norm` | `0.5` | Gradient clipping |

### Seed table

| Experiment name | `r_seed` |
|-----------------|----------|
| `hw3_dagger_seed0` | 0 |

### Checkpoint format

The final student checkpoint is saved to `<hydra_output_dir>/checkpoints/dagger_student_final.pth`  
and contains: `student`, `cfg`.

Intermediate checkpoints are saved every 5 rounds as `dagger_student_round<NNN>.pth`.

---

## Standalone Evaluation (`sim_eval.py`)

`sim_eval.py` evaluates any saved checkpoint via the unified `conf/sim_eval.yaml` config.
Set `model_type` to match the checkpoint:

| `model_type` | Checkpoint format | Parts |
|---|---|---|
| `dense_policy` | HW3 dict with `policy` key | Part 1 |
| `transformer_policy` | Raw dill-pickled GRP model (HW1 format) | Parts 2 & 3 init |

> **Note:** `TransformerPolicyWrapper` expects the raw HW1 dill pickle, **not** an HW3 state-dict dict.
> DAgger and PPO fine-tuned transformer checkpoints use the HW3 dict format and are best evaluated
> via the built-in `evaluate_policy` in `train_transformer_rl.py`.

### Dense policy (Part 1)

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python sim_eval.py \
    experiment.name=hw3_dense_ppo_seed0_evaluation \
    checkpoint=checkpoints/hw3_dense_ppo_seed0/dense_ppo_final.pth \
    model_type=dense_policy \
    simEval=[libero_fast] \
    sim.eval_episodes=20 \
    sim.eval_tasks=[9] \
    testing=false
```

### Transformer policy — HW1-format pickle (Parts 2 & 3 init)

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python sim_eval.py \
    experiment.name=hw3_transformer_grpo_evaluation \
    checkpoint=checkpoints/grp-ver2/miniGRP.pth \
    model_type=transformer_policy \
    simEval=[libero_fast] \
    sim.eval_episodes=20 \
    sim.eval_tasks=[0] \
    testing=false
```

Add `testing=true` to skip W&B logging (dry-run / CI mode).

### Key `conf/sim_eval.yaml` parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `model_type` | `dense_policy` | `"dense_policy"` \| `"transformer_policy"` |
| `simEval` | `[libero_fast]` | Evaluators; `libero` requires `model_type=transformer_policy` |
| `sim.eval_episodes` | `10` | Number of evaluation episodes |
| `sim.eval_tasks` | `[9]` | LIBERO-Spatial task IDs |
| `sim.episode_length` | `300` | Max steps per episode |
| `testing` | `false` | `true` = skip W&B logging |
| `dense_policy.obs_dim` | `13` | 7 proprio + 6 relative object pose |
| `transformer_policy.fast_env_image_size` | `64` | Must match the checkpoint's baked-in image shape |
| `transformer_policy.use_pose` | `true` | Whether to feed the pose token to the transformer |

