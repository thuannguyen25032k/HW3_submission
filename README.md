# HW3 Submission — Making a Generalist Robotics Policy with Reinforcement Learning

## Repository layout

```
HW3_submission/
├── data/
│   ├── hw3_dense_ppo_seed0/          # Part 1 — Dense PPO seed 0
│   ├── hw3_dense_ppo_seed1/          # Part 1 — Dense PPO seed 1
│   ├── hw3_dense_ppo_seed2/          # Part 1 — Dense PPO seed 2
│   ├── hw3_transformer_ppo/          # Part 2a — Transformer PPO
│   ├── hw3_transformer_grpo/         # Part 2b — Transformer GRPO (ground-truth resets)
│   ├── hw3_transformer_grpo_dreamer/ # Part 2c — Transformer GRPO + DreamerV3 world model
│   └── hw3_dagger_seed0/             # Part 3  — DAgger distillation (optional)
└── mini-grp/                         # All training and evaluation scripts
    ├── train_dense_rl.py
    ├── train_transformer_rl.py
    ├── train_dagger.py
    ├── grp_model.py
    ├── networks.py
    ├── dreamerV3.py
    ├── libero_env_fast.py
    ├── sim_eval.py
    └── conf/
        ├── dense_ppo.yaml
        ├── transformer_rl.yaml
        ├── dagger.yaml
        └── sim_eval.yaml
```

Each `data/<run>/` folder contains:
- `.hydra/` — the exact Hydra config and CLI overrides used for that run
- `train_*.log` — stdout/stderr from training
- `videos/` — periodic evaluation rollout recordings
- `wandb/` — W&B run data (where available)

---

## Environment setup

All scripts must be run from the `mini-grp/` directory with the `hw2` conda environment activated.

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

Three independent seeds were trained to **2 M environment steps** on LIBERO-Spatial task 9
using an MLP actor-critic with PPO.

### Reproduce seed 0

```bash
python train_dense_rl.py \
    experiment.name=hw3_dense_ppo_seed0 \
    r_seed=0 \
    sim.task_set=libero_spatial \
    sim.eval_tasks=[9] \
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

Saved to `outputs/<date>/<time>/checkpoints/dense_ppo_<step>.pth`.
Keys: `policy`, `value`, `optimizer`, `step`, `cfg`.

---

## Part 2 — Transformer RL Fine-tuning (LIBERO-Spatial task 0)

All Part 2 runs initialise from the HW1 GRP checkpoint (`checkpoints/grp-ver2/miniGRP.pth`)
and target **LIBERO-Spatial task 0**.

---

### Part 2a — PPO fine-tuning

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
    training.rollout_length=2000 \
    training.ppo_epochs=10 \
    training.minibatch_size=128
```

### Key hyperparameters (Part 2a)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `training.learning_rate` | `3e-5` | Small LR for fine-tuning a pretrained transformer |
| `training.clip_eps` | `0.1` | Tighter PPO clip to protect HW1 representations |
| `training.entropy_coeff` | `0.001` | Small entropy bonus |
| `training.target_kl` | `0.01` | Early-stop PPO epochs when KL exceeds this |
| `training.anneal_lr` | `true` | Linear LR decay over `total_env_steps` |

### Checkpoint format (PPO)

Saved to `checkpoints/<experiment.name>/transformer_ppo_<step>.pth`.
Keys: `policy`, `value_fn`, `optimizer`, `total_steps`, `cfg`.

---

### Part 2b — GRPO with ground-truth resets

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

### Key hyperparameters (Part 2b)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `grpo.chunk_size` | `32` | Max steps per backward chunk (reduce to `16` if OOM) |
| `grpo.kl_coef` | `0.04` | β for KL(π_θ ‖ π_ref) penalty |
| `grpo.num_groups` | `4` | Number of episode groups per GRPO update |
| `grpo.group_size` | `8` | Episodes per group |
| `training.entropy_coeff` | `0.0` | Disabled — non-zero causes log_std drift and collapse |
| `training.clip_eps` | `0.1` | Tighter clip for fine-tuning |

### Checkpoint format (GRPO)

Saved to `checkpoints/<experiment.name>/transformer_grpo_<step>.pth`.
Keys: `policy`, `optimizer`, `total_steps`, `cfg`.

---

### Part 2c — GRPO with DreamerV3 world model

Uses imagined rollouts from a pre-trained DreamerV3 world model
(`checkpoints/dreamer/world_model.pth`) to supplement real environment experience.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train_transformer_rl.py \
    experiment.name=hw3_transformer_grpo_dreamer_seed0 \
    r_seed=0 \
    init_checkpoint=checkpoints/grp-ver2/miniGRP.pth \
    rl.algorithm=grpo_worldmodel \
    sim.task_set=libero_spatial \
    sim.eval_tasks=[0] \
    sim.reward_scale=1.0 \
    sim.episode_length=200 \
    training.total_env_steps=200000 \
    training.learning_rate=3e-5 \
    training.entropy_coeff=0.0 \
    training.max_grad_norm=0.5 \
    training.clip_eps=0.1 \
    grpo.group_size=8 \
    grpo.wm_horizon=15 \
    grpo.wm_update_every=4 \
    grpo.chunk_size=32 \
    grpo.kl_coef=0.0 \
    world_model.checkpoint=checkpoints/dreamer/world_model.pth \
    world_model.type=dreamer \
    world_model.obs_shape=[3,64,64] \
    world_model.action_dim=7 \
    world_model.stoch_dim=32 \
    world_model.discrete_dim=32 \
    world_model.deter_dim=512 \
    world_model.hidden_dim=512
```

### Key hyperparameters (Part 2c)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `rl.algorithm` | `grpo_worldmodel` | Enables DreamerV3-augmented GRPO |
| `sim.reward_scale` | `1.0` | Higher scale to match imagined reward magnitudes |
| `grpo.wm_horizon` | `15` | Imagination rollout length per group |
| `grpo.wm_update_every` | `4` | Real env steps between world-model GRPO updates |
| `grpo.kl_coef` | `0.0` | KL penalty disabled (world-model rollouts handle regularisation) |
| `world_model.stoch_dim` | `32` | DreamerV3 stochastic latent dimension |
| `world_model.deter_dim` | `512` | DreamerV3 deterministic state dimension |

### Checkpoint format (GRPO + world model)

Same as GRPO: `checkpoints/<experiment.name>/transformer_grpo_<step>.pth`.
Keys: `policy`, `optimizer`, `total_steps`, `cfg`.

---

## Part 3 — DAgger (LIBERO-Spatial task 9, optional)

The transformer student is initialised from the HW1 GRP checkpoint and distilled from the
best Part 1 dense PPO teacher (`hw3_dense_ppo_seed2`) on LIBERO-Spatial task 9.

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
python train_dagger.py \
    experiment.name=hw3_dagger_seed0 \
    r_seed=0 \
    teacher_checkpoint=checkpoints/hw3_dense_ppo_seed2/dense_ppo_final.pth \
    student_init_checkpoint=checkpoints/grp-ver2/miniGRP.pth \
    sim.task_set=libero_spatial \
    sim.eval_tasks=[9] \
    sim.episode_length=300 \
    dagger.num_rounds=120 \
    dagger.rollouts_per_round=120 \
    dagger.bc_epochs_per_round=80 \
    dagger.beta_schedule=linear \
    dagger.beta_init=1.0 \
    dagger.dataset_save_dir=dagger_datasets/seed0 \
    training.learning_rate=1e-4 \
    training.minibatch_size=64 \
    training.max_grad_norm=0.5 \
    eval_interval=1 \
    save_interval=5
```

### Key hyperparameters (Part 3)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `dagger.num_rounds` | `120` | DAgger iterations |
| `dagger.rollouts_per_round` | `120` | Rollouts collected per round |
| `dagger.bc_epochs_per_round` | `80` | Supervised BC epochs over aggregated dataset per round |
| `dagger.beta_schedule` | `linear` | β decays from `1.0` (pure teacher) → `0.0` (pure student) |
| `training.learning_rate` | `1e-4` | Adam LR for BC updates |
| `training.minibatch_size` | `64` | Mini-batch size for BC |

### Checkpoint format (DAgger)

Final student: `<hydra_output_dir>/checkpoints/dagger_student_final.pth`.
Keys: `student`, `cfg`.
Intermediate: `dagger_student_round<NNN>.pth` every 5 rounds.

---

## Standalone Evaluation (`sim_eval.py`)

`sim_eval.py` evaluates any saved checkpoint via the unified `conf/sim_eval.yaml` config.
Set `model_type` to match the checkpoint:

| `model_type` | Checkpoint type | Applicable parts |
|---|---|---|
| `dense_policy` | HW3 dict with `policy` key | Part 1 |
| `transformer_policy` | Raw dill-pickled GRP model (HW1 format) | Parts 2 & 3 init |

> **Note:** `TransformerPolicyWrapper` expects the raw HW1 dill pickle, **not** an HW3 state-dict.
> HW3 transformer checkpoints (PPO / GRPO fine-tuned) are best evaluated via the built-in
> `evaluate_policy` call inside `train_transformer_rl.py`.

### Dense policy (Part 1)

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python sim_eval.py \
    experiment.name=hw3_dense_ppo_seed2_eval \
    checkpoint=checkpoints/hw3_dense_ppo_seed2/dense_ppo_final.pth \
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
    experiment.name=hw3_transformer_grpo_eval \
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
| `simEval` | `[libero_fast]` | `[libero_fast]` \| `[libero]` (latter requires transformer_policy) |
| `sim.eval_episodes` | `10` | Number of evaluation episodes |
| `sim.eval_tasks` | `[9]` | LIBERO-Spatial task IDs |
| `sim.episode_length` | `300` | Max steps per episode |
| `testing` | `false` | `true` = skip W&B logging |
| `dense_policy.obs_dim` | `13` | 7 proprio + 6 relative object pose |
| `transformer_policy.fast_env_image_size` | `64` | Must match the checkpoint's baked-in image shape |
| `transformer_policy.use_pose` | `true` | Whether to feed the pose token to the transformer |

