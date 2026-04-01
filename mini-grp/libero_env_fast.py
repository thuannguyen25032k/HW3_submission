"""
Fast LIBERO environment - bypasses Robosuite wrapper for ~6000 SPS
"""
import hydra
import numpy as np
import gymnasium as gym
import os
import sys
from typing import Any, Mapping, cast

# Add LIBERO path
# sys.path.insert(0, '/teamspace/studios/this_studio/LIBERO')

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import DenseRewardEnv, OffScreenRenderEnv


class FastLIBEROEnv:
    """
    Fast LIBERO using raw MuJoCo for stepping.
    Observation: robot proprio + simple object positions.
    Reward: dense distance-based reward.
    """

    def __init__(self, benchmark_name='libero_spatial', task_id=0, max_episode_steps=300,
                 num_sim_steps=1, action_repeat=1, render_mode=None, cfg=None,
                 output_image_obs=False, image_size=64, image_camera='agentview'):
        self.benchmark_name = benchmark_name
        self.task_id = task_id
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.prev_bowl_pos = None
        self.prev_bowl_rel = None
        # Persistent "holding" estimate used by shaped reward to avoid gating flicker.
        self._hold_conf = 0.0
        self._prev_bowl_pos_world = None
        self._prev_gripper_to_bowl_dist = None
        self._prev_bowl_to_plate_xy_dist = None
        self._init_bowl_height = None
        self._init_plate_height = None
        self.render_mode = render_mode
        self._num_settle_steps = 10

        sim_cfg = getattr(cfg, "sim", None)
        cfg_output_image = bool(getattr(
            sim_cfg, "fast_env_output_image", False)) if sim_cfg is not None else False
        cfg_image_size = int(getattr(sim_cfg, "fast_env_image_size",
                             image_size)) if sim_cfg is not None else image_size
        cfg_image_camera = str(getattr(sim_cfg, "fast_env_image_camera",
                               image_camera)) if sim_cfg is not None else image_camera
        # reward_scale is applied to ver2 and ver3 to keep episode returns in a range
        # the critic can track with value_clip_eps. Defaults to 1.0 (no scaling).
        self.reward_scale = float(
            getattr(sim_cfg, "reward_scale", 1.0)) if sim_cfg is not None else 1.0
        self.output_image_obs = bool(output_image_obs or cfg_output_image)
        self.image_size = int(cfg_image_size)
        self.image_camera = cfg_image_camera

        # Get task
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite_name = cfg.sim.task_set if cfg is not None else benchmark_name
        self.benchmark_name = task_suite_name
        task_suite = benchmark_dict[task_suite_name]()
        task = task_suite.get_task(task_id)
        self.instruction = task.language
        task_bddl_file = os.path.join(get_libero_path(
            "bddl_files"), task.problem_folder, task.bddl_file)

        # Create base env
        enable_offscreen = (
            render_mode == 'rgb_array') or self.output_image_obs
        camera_size = self.image_size if self.output_image_obs else 256
        self.env = DenseRewardEnv(
            bddl_file_name=task_bddl_file,
            use_camera_obs=enable_offscreen,
            has_offscreen_renderer=enable_offscreen,
            has_renderer=False,
            camera_heights=camera_size,
            camera_widths=camera_size
            # control_freq=5,  # Control frequency (Hz)
            # controller="OSC_POSE",
        )

        # Action space: 7DOF arm
        self._action_dim = 7
        self._num_sim_steps = num_sim_steps
        self._action_repeat = action_repeat

        # Get robot and object info
        self._setup()

        # Observation space: either proprio/object state vector or RGB image.
        self.obs_dim = 7 + 3 * len(self.target_objects)  # 7 + 3*2 = 13
        if self.output_image_obs:
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(self.image_size, self.image_size, 3), dtype=np.uint8
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=-10, high=10, shape=(self.obs_dim,), dtype=np.float32
            )

        # Action space: 7DOF arm + 1 gripper (use 7 for now)
        self._action_dim = 7
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self._action_dim,), dtype=np.float32
        )

    def set_init_state(self, num_states):
        self.env.set_init_state(num_states)

    def _setup(self):
        """Extract object names and control range"""
        # Get inner env
        inner = self.env.env if hasattr(self.env, 'env') else self.env

        # Get robot
        self.robot = inner.robots[0]

        # Get target objects (first 2)
        self.target_objects = []
        # for name in inner.objects_dict.keys():
        #     if len(self.target_objects) >= 2:
        #         break
        self.target_objects.append("akita_black_bowl_1_main")
        self.target_objects.append("plate_1_main")

        # Get goal position (from task)
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[self.benchmark_name]()
        task = task_suite.get_task(self.task_id)
        self.goal_description = task.language

        # Control range
        self.ctrlrange = self.env.sim.model.actuator_ctrlrange[:self._action_dim]

    def _get_state_obs(self, obs_dict=None):
        """State observation: [eef_pos(3), eef_quat_xyz(3), gripper_qpos(1), rel_obj_offsets(6)]."""
        if obs_dict is None:
            obs_getter = getattr(self.env, "_get_observations", None)
            if callable(obs_getter):
                obs_dict = obs_getter()
            else:
                raise RuntimeError(
                    "FastLIBEROEnv requires an observation dict from env.reset/env.step")

        obs_dict = cast(Mapping[str, Any], obs_dict)

        eef_pos = np.asarray(obs_dict["robot0_eef_pos"], dtype=np.float32)
        eef_quat_xyz = np.asarray(
            obs_dict["robot0_eef_quat"], dtype=np.float32)[:3]
        gripper_qpos = np.asarray(
            [obs_dict["robot0_gripper_qpos"][0]], dtype=np.float32)

        sim = self.env.sim
        rel_obj_pos = []
        for name in self.target_objects:
            body_id = sim.model.body_name2id(name)
            obj_pos = sim.data.body_xpos[body_id]
            rel_obj_pos.append((obj_pos - eef_pos).astype(np.float32))

        state = np.concatenate(
            [eef_pos, eef_quat_xyz, gripper_qpos] + rel_obj_pos).astype(np.float32)
        return state

    def _get_image_obs(self):
        """RGB image observation from simulator camera."""
        frame = self.render(camera_name=self.image_camera,
                            width=self.image_size, height=self.image_size)
        if frame is None:
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        frame = np.asarray(frame)
        if frame.ndim == 2:
            frame = np.repeat(frame[:, :, None], 3, axis=2)
        if frame.shape[-1] > 3:
            frame = frame[:, :, :3]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame

    def _get_obs(self, obs_dict=None):
        if self.output_image_obs:
            return self._get_image_obs()
        return self._get_state_obs(obs_dict)

    def _reward(self, state, action):
        # 1. Extract Positions
        bowl_rel = state[7:10]  # bowl relative to gripper
        plate_rel = state[10:13]  # plate relative to gripper

        dist_gripper_bowl = np.linalg.norm(
            bowl_rel)    # gripper to bowl distance

        # bowl to plate relative position
        bowl_plate_rel = (state[7:10] - state[10:13])
        # horizontal distance between bowl and plate
        dist_bowl_plate_xy = float(
            np.linalg.norm(bowl_rel[:2] - plate_rel[:2]))
        # vertical distance between bowl and plate
        dist_bowl_plate_z = float(abs(bowl_rel[2] - plate_rel[2]))

        reward = 0.0
        reward -= dist_gripper_bowl  # Encourage reaching towards bowl

        is_grasping = dist_gripper_bowl < 0.04  # Grasping threshold (4cm)

        if is_grasping:
            reward += 1.0  # Bonus for grasping bowl
            reward += - 2 * dist_bowl_plate_xy  # Encourage moving bowl towards plate

        is_reaching_plate = dist_bowl_plate_xy < 0.05
        if is_grasping and is_reaching_plate:
            reward += 2.0  # Bonus for reaching above plate
            reward += -4*dist_bowl_plate_z  # Encourage lowering bowl onto plate

        # --- Sparse completion reward logic (for reference, not used in training) ---
        # Success criteria: bowl close to plate (XY < 5cm, similar height) and above plate
        height_close = abs(bowl_plate_rel[2]) < 0.05
        is_success = (dist_bowl_plate_xy <
                      0.05 and height_close and bowl_rel[2] > plate_rel[2])

        if is_success:
            reward += 40.0  # Completion reward

        reward *= self.reward_scale

        return reward, {
            'reward': reward,
            'is_grasping': is_grasping,
            'success_placed': is_success
        }

    def _compute_reward(self, action, state=None):
        """Dense reward computed from the same vector returned by _get_obs."""
        if state is None:
            raise RuntimeError(
                "_compute_reward requires an explicit state when using FastLIBEROEnv")
        reward, reward_info = self._reward(state, action)

        return reward, reward_info

    def _compute_init_distance(self):
        """Compute initial distance between bowl and plate"""
        sim = self.env.sim
        try:
            init_bowl = sim.data.body_xpos[sim.model.body_name2id(
                self.target_objects[0])]
            init_plate = sim.data.body_xpos[sim.model.body_name2id(
                self.target_objects[1])]
            self.init_bowl_plate_dist = np.linalg.norm(init_bowl - init_plate)
        except:
            self.init_bowl_plate_dist = 0.1  # default

    def reset(self, seed=None, options=None):
        self.current_step = 0
        obs_dict = self.env.reset()

        init_state = None
        if options is not None:
            init_state = options.get("init_state", None)
        if init_state is not None:
            obs_dict = self.env.set_init_state(init_state)

        # Compute initial distance
        self._compute_init_distance()

        # Warmup: apply no-op actions to let objects settle
        for _ in range(self._num_settle_steps):
            obs_dict, _, done, _ = self.env.step([0, 0, 0, 0, 0, 0, -1])
            if done:
                break

        state = self._get_state_obs(obs_dict)
        # Update previous bowl-relative position for next step
        self.prev_bowl_rel = state[7:10].copy()

        # Reset hold tracking.
        self._hold_conf = 0.0
        self._prev_gripper_to_bowl_dist = None
        self._prev_bowl_to_plate_xy_dist = None
        try:
            sim = self.env.sim
            bowl_pos = sim.data.body_xpos[sim.model.body_name2id(
                self.target_objects[0])]
            self._prev_bowl_pos_world = np.asarray(
                bowl_pos, dtype=np.float32).copy()
            self._init_bowl_height = float(bowl_pos[2])
        except Exception:
            self._prev_bowl_pos_world = None
            self._init_bowl_height = None
        try:
            sim = self.env.sim
            plate_pos = sim.data.body_xpos[sim.model.body_name2id(
                self.target_objects[1])]
            self._init_plate_height = float(plate_pos[2])
        except Exception:
            self._init_plate_height = None

        obs = self._get_image_obs() if self.output_image_obs else state
        info = {"state_obs": state.copy()}
        return obs, info

    def step(self, action):
        action = np.clip(action, -1, 1)
        obs_dict = None
        done = False
        info_env = {}
        for _ in range(max(1, self._num_sim_steps)):
            obs_dict, _, done, info_env = self.env.step(action.tolist())
            if done:
                break

        # Compute reward
        state = self._get_state_obs(obs_dict)
        reward, reward_info = self._reward(state, action)

        # Check success metrics (regardless of reward function)
        sim = self.env.sim
        gripper_pos = state[:3]

        try:
            bowl_pos = sim.data.body_xpos[sim.model.body_name2id(
                self.target_objects[0])]
        except:
            bowl_pos = np.zeros(3)

        try:
            plate_pos = sim.data.body_xpos[sim.model.body_name2id(
                self.target_objects[1])]
        except:
            plate_pos = np.zeros(3)

        # Success 1: Gripper close to bowl (grasping) - threshold 4cm
        dist_gripper_bowl = np.linalg.norm(gripper_pos - bowl_pos)
        success_grasping = 1.0 if dist_gripper_bowl < 0.04 else 0.0

        # Success 2: Bowl on plate - threshold 5cm horizontal, similar height
        dist_bowl_plate = np.linalg.norm(
            bowl_pos[:2] - plate_pos[:2])  # x,y only
        height_close = abs(bowl_pos[2] - plate_pos[2]) < 0.05
        success_placed = 1.0 if (
            dist_bowl_plate < 0.05 and height_close and bowl_pos[2] > plate_pos[2]) else 0.0

        # Add to info
        reward_info['success_grasping'] = float(success_grasping)
        reward_info['success_placed'] = float(success_placed)

        # Check termination
        self.current_step += 1

        done = False
        truncated = self.current_step >= self.max_episode_steps
        if truncated:
            done = True

        info = dict(info_env)
        info.update(reward_info)
        info['state_obs'] = state.copy()

        obs = self._get_image_obs() if self.output_image_obs else state
        return obs, reward, done, truncated, info

    def close(self):
        self.env.close()

    def render(self, camera_name='agentview', width=256, height=256):
        if not (self.render_mode == 'rgb_array' or self.output_image_obs):
            return None

        try:
            frame = self.env.sim.render(
                camera_name=camera_name, width=width, height=height)
            if frame is None:
                return None
            frame = np.asarray(frame)
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            return frame[::-1, :, :]
        except Exception:
            return None

    @property
    def unwrapped(self):
        return self


if __name__ == '__main__':
    import time

    env = FastLIBEROEnv()
    obs, _ = env.reset()
    print(f"Obs shape: {obs.shape}")
    print(f"Action shape: {env.action_space.shape}")

    # Test speed
    start = time.time()
    episodes = 0
    steps = 0
    for i in range(500):
        action = env.action_space.sample()
        obs, reward, done, truncated, _ = env.step(action)
        steps += 1
        if done:
            env.reset()
            episodes += 1
    elapsed = time.time() - start

    print(f"FastLIBERO: {steps/elapsed:.0f} SPS, {episodes} episodes")
    env.close()
