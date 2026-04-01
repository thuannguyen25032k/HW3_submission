
import dill
import h5py
import numpy as np
import torch


def _as_action_sequence(decoded_action, cfg):
    """Convert decoded action output to shape (action_stacking, action_dim)."""
    k = int(cfg.policy.action_stacking)
    d = int(cfg.action_dim)
    arr = np.asarray(decoded_action)

    if arr.ndim == 2 and arr.shape == (k, d):
        return arr
    if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[1:] == (k, d):
        return arr[0]
    raise ValueError(f"Expected decoded action shape ({k}, {d}) or (1, {k}, {d}), got {arr.shape}")

def get_text_tokens(cfg, tokenizer, text_model, goal, model=None):
    """
    Get the text tokens/embeddings for the goal.
    If a `model` with `encode_text_goal` is provided, use it so callers don't need a buffer.
    """
    if model is not None:
        return model.encode_text_goal(goal, tokenizer=tokenizer, text_model=text_model)
    # fallback to legacy behaviour
    if cfg.dataset.encode_with_t5:
        goal_ = np.zeros((cfg.max_block_size, cfg.n_embd), dtype=np.float32)
        input_ids = tokenizer(goal, return_tensors="pt").input_ids
        goal_t = text_model.encoder(input_ids).last_hidden_state.detach().cpu().numpy() ## Get the goal embedding
        goal_[:len(goal_t[0]), :] = goal_t[0][:cfg.max_block_size] ## Overwrite just the zeros up to the size of this vector, smaller vectors will have < max_block_size
    else:
        goal_ = " " * cfg.max_block_size
        goal_ = goal[:cfg.max_block_size] + goal_[len(goal):cfg.max_block_size]
        # legacy buffer-based encoding is not available here
        raise RuntimeError("Text encoding without model requires a buffer; pass model into get_text_tokens")
    return np.expand_dims(goal_, axis=0)

def get_blocked_mask(cfg, targets=None, T=0):
    ## Compute blocked masks
    c=192 ## Number of patches/channels in the image
    mask = torch.ones((1 + (c * cfg.policy.obs_stacking) + T + c, ), device=cfg.device) ## (1, T)
    if targets is None:
        pass
    elif (torch.rand(1)[0] > 0.66):  
        mask[1 + (c * cfg.policy.obs_stacking): 1 + (c * cfg.policy.obs_stacking) + T] = torch.zeros((1,T), device=cfg.device) ## Mask goal string
    elif (torch.rand(1)[0] > 0.33):
        mask[1 + (c * cfg.policy.obs_stacking) + T: 1 + (c * cfg.policy.obs_stacking) + T + c] = torch.zeros((1,c), device=cfg.device) ## Mask goal image

def eval_model_in_sim(cfg, model, device, log_dir, env, env_unwrapped,
                      wandb, iter_, tokenizer=None, text_model=None):
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    print("Evaluating model in sim environment")
    from collections import deque
    from einops import rearrange

    rewards = []
    for j in range(cfg.sim.eval_episodes): ## Better to eval over a few different goal configurations
        obs, reset_info = env.reset()
        obs_ = get_image_from_maniskill2_obs_dict(env_unwrapped, obs)[:,:,:3]
        obs_hist = deque(maxlen=cfg.policy.obs_stacking)
        last_action = np.zeros(cfg.action_dim)  # Track last action taken
        for _ in range(cfg.policy.obs_stacking):
            obs_hist.append(obs_)
        instruction = env_unwrapped.get_language_instruction()
        # print("Reset info", reset_info)
        print("Instruction", instruction)
        frames = []
        obs_list = []
        poses_list = []
        actions_list = []
        done, truncated, timeLimit, t = False, False, 100, 0
        txt_goal = get_text_tokens(cfg, tokenizer, text_model, instruction, model=model)
        # obs_hist.append(image) ## Add the new observation to the history buffer
        while not (done or truncated or (t > timeLimit)):
            # action[:3]: delta xyz; action[3:6]: delta rotation in axis-angle representation;
            # action[6:7]: gripper (the meaning of open / close depends on robot URDF)
            # obs = [obs_["image"] for obs_ in obs] # obs is a list of dicts
            image = np.stack(obs_hist, axis=-1)  # stack along the last dimension
            image = rearrange(image, 'h w c t -> h w (c t)')  # add batch dimension

            obs_state = torch.tensor(model.preprocess_state(image), dtype=torch.float32)
            goal_state = torch.tensor(model.preprocess_goal_image(image[:,:,:3]), dtype=torch.float32)
            
            # Prepare last_action tensor if available
            last_action_tensor = None
            if last_action is not None:
                last_action_tensor = torch.tensor(last_action, dtype=torch.float32).unsqueeze(0).to(device)
            
            action, loss = model.forward(torch.tensor(obs_state.unsqueeze(0), dtype=torch.float32).to(device)
                                ,torch.tensor(txt_goal).to(device)
                                ,torch.tensor(goal_state.unsqueeze(0), dtype=torch.float32).to(device),
                                mask_=True, ## Masks goal image
                                pose=torch.tensor([[obs["extra"]["tcp_pose"]]], dtype=torch.float32).to(device),
                                last_action=last_action_tensor,
                                )

            decoded = model.decode_action(action[0]).cpu().detach().numpy() ## Add in the gripper close action
            action_seq = _as_action_sequence(decoded, cfg)
            if t == 0:
                print(f"action_seq shape: {action_seq.shape}")
            assert action_seq.shape == (cfg.policy.action_stacking, cfg.action_dim), (
                f"Expected action_seq shape {(cfg.policy.action_stacking, cfg.action_dim)}, got {action_seq.shape}"
            )
            last_action = action_seq[-1].copy()  # Store last primitive action for next iteration
            for step_ in range(cfg.policy.action_stacking):
                act_ = action_seq[step_]
                obs, reward, done, truncated, info = env.step(act_)
                image = get_image_from_maniskill2_obs_dict(env_unwrapped, obs)
                image = image[:,:,:3] ## Remove last dimension of image color
                # Store the original image for video before stacking/processing
                frames.append(image)
                obs_list.append(obs)
                poses_list.append(obs["extra"]["tcp_pose"])
                actions_list.append(act_)
                reward = -(np.linalg.norm(info["eof_to_obj1_diff"]) + np.linalg.norm(info["eof_to_obj1_diff"])) ## Use a shaped reward as distance between gripper and objects
                rewards.append(reward)
                t=t+1   
                if done or truncated:
                    break
        
    
    episode_stats = info.get('episode_stats', {})
    episode_stats['rewards'] = np.mean(rewards)
    episode_stats['observations'] = obs_list
    episode_stats['poses'] = poses_list
    episode_stats['actions'] = actions_list
    # print("Episode stats", episode_stats)
    print(f"avg reward {np.mean(episode_stats['rewards']):.8f}")
    if not cfg.testing:
        wandb.log({"avg reward": np.mean(rewards)})
    
    import os
    path_ = os.path.join(log_dir, f"simple-env-{iter_}.mp4")
    import imageio
    imageio.mimsave(path_, frames, fps=20)
    episode_stats['video_url'] = path_

    if not cfg.testing:
        try:
            wandb.log({"example": wandb.Video(path_)})
        except Exception as e:
            print(f"Warning: failed to log video to wandb: {e}")

    return episode_stats

import gymnasium as gym
# --- History Stacking Wrapper ---
class DictWrapper(gym.ObservationWrapper):
    # from gymnasium.spaces import Box
    """
    A wrapper that grabs the observation from a specific key in the dictionary.
    """
    def __init__(self, env, obs_key=""):
        # gym.Wrapper.__init__(self, env)
        self.env = env
        self.observation_space = gym.spaces.Box( 
            low=0,
            high=255,
            shape=(256,256,3),  # Assuming the observation is an image of size 256x256 with 3 color channels
            dtype=np.uint8)
        self._obs_key = obs_key

    def observation(self, observation):
        """
        This method is called by the gym.ObservationWrapper after the environment's
        step or reset methods return an observation.
        """
        # Add the new observation to the history buffer
        return observation[self._obs_key]
    
    def step(self, action):
        """
        Step the environment and return the observation from the specified key.
        """
        obs, reward, done, info = self.env.step(action) ## LIBERO does not return truncated
        return obs[self._obs_key][::-1, :, :], reward, done, False, obs ## Not sure why the image was upside down.

    def reset(self, **kwargs):
        """
        Reset the environment and return the observation from the specified key.
        """
        obs = self.env.reset()
        return obs[self._obs_key][::-1, :, :], obs

def eval_libero(model, device, cfg, iter_=0, log_dir="./", 
                tokenizer=None, text_model=None, wandb=None,
                render=True):
        # cfg, model, device, log_dir, env, env_unwrapped, buffer,
        #               wandb, iter_, tokenizer=None, text_model=None):
    
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv, DenseRewardEnv
    import os
    from libero.libero.utils import get_libero_path
    from gymnasium.wrappers import FrameStackObservation
    from einops import rearrange
    import cv2

    target_object_names = ["akita_black_bowl_1_main", "plate_1_main"]
    from libero_env_fast import FastLIBEROEnv
    rw_func = FastLIBEROEnv(
            max_episode_steps=cfg.sim.episode_length
        )._reward

    def get_relative_object_offsets(sim_env, eef_pos):
        rel_offsets = []
        for obj_name in target_object_names:
            try:
                body_id = sim_env.sim.model.body_name2id(obj_name)
                obj_pos = sim_env.sim.data.body_xpos[body_id]
                rel_offsets.append(obj_pos - eef_pos)
            except Exception:
                continue
        if len(rel_offsets) == 0:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(rel_offsets, axis=-1).astype(np.float32)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_name = cfg.sim.task_set # can also choose libero_spatial, libero_object, etc.
    task_suite = benchmark_dict[task_suite_name]()
    
    # Load initial states and goal images from Hugging Face dataset if provided
    init_states_dataset = None
    if hasattr(cfg.sim, 'libero_init_state_hf_repo') and cfg.sim.libero_init_state_hf_repo:
        print(f"Loading initial states from Hugging Face: {cfg.sim.libero_init_state_hf_repo}")
        from datasets import load_dataset
        init_states_dataset = load_dataset(cfg.sim.libero_init_state_hf_repo, split='train')
        print(f"Loaded dataset with {len(init_states_dataset)} entries")
    elif hasattr(cfg.sim, 'libero_init_state_file') and cfg.sim.libero_init_state_file:
        print(f"Loading initial states from HDF5: {cfg.sim.libero_init_state_file}")
        init_states_dataset = h5py.File(hydra.utils.get_original_cwd()+cfg.sim.libero_init_state_file, 'r')

    trajectory_data = []
    # retrieve a specific task
    tasks = cfg.sim.eval_tasks
    for task_id in tasks:
        task = task_suite.get_task(task_id)
        task_name = task.name
        instruction = task.language
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
            f"language instruction is {instruction}, and the bddl file is {task_bddl_file}")

        # step over the environment
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": 256,
            "camera_widths": 256
        }

        env = DenseRewardEnv(**env_args) # env = OffScreenRenderEnv(**env_args)
        env.seed(0)
        
        # Load initial states from dataset if available, otherwise use default
        task_description = instruction.replace(" ", "_")
        task_demos = None
        if init_states_dataset is not None:
            task_demos = [item for item in init_states_dataset if item.get('task_description') == task_description]
            num_init_states = len(task_demos)
            if num_init_states > 0:
                print(f"Loaded {num_init_states} initial states from HF dataset for task: {task_description}")
            else:
                init_states = task_suite.get_task_init_states(task_id)
                num_init_states = len(init_states)
                print(f"Using default initial states for task: {task_description}")
        else:
            init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
            num_init_states = len(init_states)
            print(f"Using default initial states for task: {task_description}")
        
        # for init_state_id in range(len(init_states)):
        for init_state_id in range(min(1, num_init_states)):  ## Just do a couple different initializations for eval
            # Load init_state and goal_img from dataset or use default
            if init_states_dataset is not None:
                # Hugging Face dataset format
                if task_demos and init_state_id < len(task_demos):
                    demo = task_demos[init_state_id]
                    init_state = np.array(demo['init_state'])
                    goal_img = np.array(demo['goal_img']) if 'goal_img' in demo and demo['goal_img'] is not None else None
                    print(f"Loaded init_state and goal_img from HF dataset for demo {init_state_id}")
                else:
                    init_state = init_states[init_state_id]
                    goal_img = None
            else:
                init_state = init_states[init_state_id]
                goal_img = None
            

            env.reset()
            env.set_init_state(init_state)
            env_ = FrameStackObservation(DictWrapper(env, obs_key="agentview_image"), cfg.policy.obs_stacking) ## Stacking the observations
            obs, info = env_.reset()

            mask = get_blocked_mask(cfg, targets=None, T=0) ## Get the blocked mask
            
            txt_goal = get_text_tokens(cfg, tokenizer, text_model, instruction, model=model)
            
            # Use goal image from HDF5 if available, otherwise use first observation
            if goal_img is not None:
                image_goal = goal_img
                print(f"Using goal image from HDF5, shape: {image_goal.shape}")
            else:
                image_goal = np.zeros((256, 256, 3*cfg.policy.obs_stacking))[:,:,:3]
                print("Using first observation as goal image")
            frames = []
            rewards = []
            infos = []
            obs_list = []
            poses_list = []
            actions_list = []
            last_action = np.zeros(cfg.action_dim)  # Track last action taken
            done, truncated, timeLimit, t, wait_steps = False, False, cfg.sim.episode_length, 0, 0

            while not (done or truncated or (t > (timeLimit + wait_steps))):
                ## Reshape the image to the correct size and stack the hostory on the last channel dimension
                # image = obs[0]
                if t < wait_steps: ## let object stabalize before acting.
                    obs, reward, done, truncated, info = env_.step([0,0,0,0,0,0,0])
                    t += 1
                    continue
                # obs = obs.reshape((128, 128, 3*cfg.policy.obs_stacking)) ## Assuming the observation is an image of size 128x128 with 3 color channels  
                obs = rearrange(obs, 't h w c -> h w (t c)', c=3, t=cfg.policy.obs_stacking) ## Rearranging the image to have the stacked history in the last channel dimension
                # image = obs[:,:,:3] ## Remove the last dimension of the image color
                obs_state = model.preprocess_state(obs)
                goal_state = model.preprocess_goal_image(image_goal)
                rel_obj_offsets = get_relative_object_offsets(env, info["robot0_eef_pos"])
                pose_vec = np.concatenate(
                    (
                        info["robot0_eef_pos"],
                        info["robot0_eef_quat"][:3],
                        [(info["robot0_gripper_qpos"][0])],
                        rel_obj_offsets,
                    ),
                    axis=-1,
                )
                pose_ = torch.tensor([[pose_vec]], 
                                    dtype=torch.float32).to(device)
 
                # Prepare last_action tensor if available  
                last_action_tensor = None
                if last_action is not None:
                    last_action_tensor = model.encode_action(torch.tensor([[last_action]], dtype=torch.float32)).to(device)
                
                out = model.forward(
                    observations=torch.tensor(np.array([[obs_state]])).to(device),
                    text_goal=torch.tensor(txt_goal).to(device),
                    goal_image=torch.tensor(np.array([goal_state])).to(device), 
                    mask_=True,
                    pose=pose_,
                    prev_actions=last_action_tensor,
                )

                decoded = model.decode_action(out['actions']).cpu().detach().numpy()
                action_seq = _as_action_sequence(decoded, cfg)
                if t == 0:
                    print(f"action_seq shape: {action_seq.shape}")
                assert action_seq.shape == (cfg.policy.action_stacking, cfg.action_dim), (
                    f"Expected action_seq shape {(cfg.policy.action_stacking, cfg.action_dim)}, got {action_seq.shape}"
                )
                last_action = action_seq[-1].copy()
                ## If the actions are stacked into a longer vector execute the sequence of actions
                for step_ in range(cfg.policy.action_stacking):
                    act_ = action_seq[step_]
                    ## Resize image for data
                    if obs.ndim == 4 and obs.shape[0] == 1:
                        obs = obs[0]
                    image = obs  # Resize to 128x128
                    frames.append(image)
                    rel_obj_offsets = get_relative_object_offsets(env, info["robot0_eef_pos"])
                    pose_data = np.concatenate(
                        (
                            info["robot0_eef_pos"],
                            info["robot0_eef_quat"][:3],
                            [(info["robot0_gripper_qpos"][0])],
                            rel_obj_offsets,
                        ),
                        axis=-1,
                    )
                    obs_list.append(cv2.resize(obs, (64, 64)))

                    obs, reward, done, truncated, info = env_.step(act_)
                    pose_data_ = np.concatenate(
                        (
                            info["robot0_eef_pos"],
                            info["robot0_eef_quat"][:3],
                            [(info["robot0_gripper_qpos"][0])],
                            rel_obj_offsets,
                        ),
                        axis=-1,
                    )
                    reward = rw_func(pose_data_, act_)[0] ## Use the reward function from the environment for more accurate rewards
                    # Store the original image for video before stacking/processing
                    poses_list.append(pose_data)
                    actions_list.append(act_)
                    # reward = -(np.linalg.norm(info["eof_to_obj1_diff"]) + np.linalg.norm(info["eof_to_obj1_diff"])) ## Use a shaped reward as distance between gripper and objects
                    rewards.append(reward)
                    infos.append(info)
                    t=t+1   
                    # print(f"Step {t}, reward: {reward:.4f}, done: {done}, truncated: {truncated}")
                    if done or truncated:
                        print("Episode finished with success after {} timesteps".format(step_))
                        break
                if done:
                    print("Episode finished with success after {} timesteps".format(step_))
                    break
            
            trajectory_data.append({
                'task_id': task_id,
                'init_state_id': init_state_id,
                'rewards': rewards,
                'infos': infos,
                'observations': obs_list,
                'poses': poses_list,
                'actions': actions_list,
            })
            import os
            path_ = os.path.join(log_dir, f"libero-{iter_}-task-id-{task_id}-init-id-{init_state_id}.mp4")
            import imageio
            imageio.mimsave(path_, frames, fps=20)
    episode_stats = info.get('episode_stats', {})
    episode_stats['rewards'] = np.mean([np.mean(traj['rewards']) for traj in trajectory_data])
    episode_stats['video_url'] = path_
    episode_stats['traj'] = trajectory_data
    print(f"avg reward {np.mean([np.mean(traj['rewards']) for traj in trajectory_data]):.8f}")
    if not cfg.testing:
        wandb.log({"avg reward_"+str(task_id): np.mean([np.mean(traj['rewards']) for traj in trajectory_data])})
    if not cfg.testing:
        wandb.log({"example": wandb.Video(path_)})
    env.close()
    
    # Close HDF5 file if it was opened
    if init_states_dataset is not None and isinstance(init_states_dataset, h5py.File):
        init_states_dataset.close()
        print("Closed HDF5 file")
    
    return episode_stats


def eval_libero_fast(model, device, cfg, iter_=0, log_dir="./",
                     tokenizer=None, text_model=None, wandb=None,
                     render=False):
    """Evaluate a DensePolicy on FastLIBEROEnv.

    Uses ``info["state_obs"]`` (the privileged state vector) as input,
    matching exactly how DensePolicy was trained in ``train_dense_rl.py``.
    One action is produced per step via ``model.get_action(obs_t, deterministic=True)``.

    Args:
        model:   DensePolicy instance (already on ``device``, in eval() mode).
        device:  torch.device.
        cfg:     Hydra DictConfig.  Uses ``sim.*`` keys.
        iter_:   Iteration index used for video file naming.
        log_dir: Directory to save mp4 videos.
        render:  Whether to save a video for each episode.
        wandb:   Optional wandb run handle for logging.

    Returns:
        dict with keys: rewards, success_rate, traj, video_url.
    """
    from libero.libero import benchmark
    from libero_env_fast import FastLIBEROEnv
    import os
    import imageio

    benchmark_dict  = benchmark.get_benchmark_dict()
    task_suite_name = cfg.sim.task_set
    task_suite      = benchmark_dict[task_suite_name]()

    trajectory_data = []
    last_video_path = None
    total_successes = 0
    total_episodes  = 0

    for task_id in cfg.sim.eval_tasks:
        env = FastLIBEROEnv(
            benchmark_name=task_suite_name,
            task_id=int(task_id),
            max_episode_steps=cfg.sim.episode_length,
            render_mode="rgb_array" if render else None,
            cfg=cfg,
        )
        task        = task_suite.get_task(int(task_id))
        instruction = task.language
        print(f"[info] fast eval task {task_id} from suite {task_suite_name}: {instruction}")

        init_states     = task_suite.get_task_init_states(int(task_id))
        episodes_to_run = min(cfg.sim.eval_episodes, len(init_states))

        for init_state_id in range(episodes_to_run):
            env.reset()
            env.set_init_state(init_states[init_state_id])
            obs, info = env.reset()

            rewards      = []
            infos        = []
            poses_list   = []
            actions_list = []
            obs_list     = []
            frames       = []
            done = truncated = False
            t    = 0

            if render:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            with torch.no_grad():
                while not (done or truncated or t >= cfg.sim.episode_length):
                    # Privileged state obs — same input used during training
                    state_obs = np.asarray(info["state_obs"], dtype=np.float32)
                    obs_t     = torch.tensor(state_obs, dtype=torch.float32,
                                             device=device).unsqueeze(0)   # (1, obs_dim)

                    # Single deterministic action from the policy
                    action_np, _, _, _ = model.get_action(obs_t, deterministic=True)

                    poses_list.append(state_obs.copy())
                    obs_list.append(state_obs.copy())

                    obs, reward, done, truncated, info = env.step(action_np)

                    actions_list.append(action_np.copy())
                    rewards.append(float(reward))
                    infos.append(info)

                    if render:
                        frame = env.render()
                        if frame is not None:
                            frames.append(frame)
                    t += 1
                    if done or truncated:
                        break

            success = float(info.get("success_placed", 0.0))
            total_successes += int(success)
            total_episodes  += 1

            avg_ep_reward = float(np.mean(rewards)) if rewards else 0.0
            ep_return     = float(sum(rewards))
            print(f"  ep {init_state_id}: return={ep_return:.3f}  "
                  f"avg_reward={avg_ep_reward:.4f}  success={success:.0f}")

            path_ = None
            if render and frames:
                path_ = os.path.join(
                    log_dir,
                    f"libero-fast-{iter_}-task-id-{task_id}-init-id-{init_state_id}.mp4"
                )
                imageio.mimsave(path_, frames, fps=20)
                last_video_path = path_

            # ---- per-episode W&B logging ----
            if not cfg.testing and wandb is not None:
                ep_log = {
                    "eval/episode":        total_episodes,
                    "eval/task_id":        int(task_id),
                    "eval/ep_return":      ep_return,
                    "eval/avg_reward":     avg_ep_reward,
                    "eval/success":        success,
                    "eval/episode_length": t,
                }
                if path_ is not None:
                    try:
                        ep_log["eval/video"] = wandb.Video(path_, fps=20, format="mp4")
                    except Exception as e:
                        print(f"Warning: failed to attach episode video to wandb: {e}")
                wandb.log(ep_log)

            trajectory_data.append({
                "task_id":       task_id,
                "init_state_id": init_state_id,
                "rewards":       rewards,
                "infos":         infos,
                "observations":  obs_list,
                "poses":         poses_list,
                "actions":       actions_list,
                "success":       success,
                "video_url":     path_,
            })

        env.close()

    avg_reward   = float(np.mean([np.mean(traj["rewards"]) for traj in trajectory_data])) \
                   if trajectory_data else 0.0
    success_rate = float(total_successes / max(1, total_episodes))

    print(f"\n[eval summary]  avg_reward={avg_reward:.4f}  "
          f"success_rate={success_rate:.2f}  ({total_successes}/{total_episodes})")

    episode_stats = {
        "rewards":      avg_reward,
        "success_rate": success_rate,
        "traj":         trajectory_data,
        "video_url":    last_video_path,
    }

    # ---- summary W&B logging ----
    # if not cfg.testing and wandb is not None:
    #     wandb.log({
    #         "eval/avg_reward_summary":   avg_reward,
    #         "eval/success_rate_summary": success_rate,
    #         "eval/total_episodes":       total_episodes,
    #         "eval/total_successes":      total_successes,
    #     })
    #     # summary table: one row per episode
    #     columns = ["task_id", "init_state_id", "ep_return", "success"]
    #     rows    = [
    #         [t["task_id"], t["init_state_id"],
    #          float(sum(t["rewards"])), t["success"]]
    #         for t in trajectory_data
    #     ]
    #     wandb.log({"eval/episode_table": wandb.Table(columns=columns, data=rows)})

    return episode_stats

import hydra
import os
from omegaconf import DictConfig, OmegaConf


# ---------------------------------------------------------------------------
# Transformer policy eval loop  (image obs → TransformerPolicyWrapper)
# ---------------------------------------------------------------------------

def eval_libero_fast_transformer(model, device, cfg, iter_=0, log_dir="./",
                                 wandb=None, render=False):
    """Evaluate a TransformerPolicyWrapper on FastLIBEROEnv.

    Uses the image observation returned by FastLIBEROEnv (when
    ``fast_env_output_image=True``) together with the language instruction and
    an optional pose token, matching the API used in ``train_transformer_rl.py``.

    Args:
        model:   TransformerPolicyWrapper instance (on ``device``, eval mode).
        device:  torch.device.
        cfg:     Hydra DictConfig.  Uses ``sim.*`` keys.
        iter_:   Iteration index used for video file naming.
        log_dir: Directory to save mp4 videos.
        render:  Whether to save a video per episode.
        wandb:   Optional wandb run handle.

    Returns:
        dict with keys: rewards, success_rate, traj, video_url.
    """
    from libero.libero import benchmark
    from libero_env_fast import FastLIBEROEnv
    from train_transformer_rl import _extract_pose_from_info
    import imageio

    benchmark_dict  = benchmark.get_benchmark_dict()
    task_suite_name = cfg.sim.task_set
    task_suite      = benchmark_dict[task_suite_name]()

    use_pose        = model.model._cfg.policy.use_pose_data

    trajectory_data = []
    last_video_path = None
    total_successes = 0
    total_episodes  = 0

    for task_id in cfg.sim.eval_tasks:
        # FastLIBEROEnv must output images for the transformer
        env = FastLIBEROEnv(
            benchmark_name=task_suite_name,
            task_id=int(task_id),
            max_episode_steps=cfg.sim.episode_length,
            render_mode="rgb_array" if render else None,
            cfg=OmegaConf.merge(cfg, OmegaConf.create({
                "sim": {
                    "fast_env_output_image": True,
                    "fast_env_image_size": int(cfg.transformer_policy.fast_env_image_size),
                }
            })),
        )
        task        = task_suite.get_task(int(task_id))
        instruction = task.language
        print(f"[info] transformer eval task {task_id} | {task_suite_name}: {instruction}")

        init_states     = task_suite.get_task_init_states(int(task_id))
        episodes_to_run = min(cfg.sim.eval_episodes, len(init_states))

        for init_state_id in range(episodes_to_run):
            env.reset()
            env.set_init_state(init_states[init_state_id])
            obs, info = env.reset()
            obs = np.ascontiguousarray(obs)   # (H, W, C) uint8

            # Encode goal once per episode
            txt_goal, goal_state = model.encode_goals(obs, instruction)
            pose = _extract_pose_from_info(info, model, device) if use_pose else None

            rewards      = []
            infos        = []
            actions_list = []
            obs_list     = []
            frames       = []
            done = truncated = False
            t    = 0

            if render:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            with torch.no_grad():
                while not (done or truncated or t >= cfg.sim.episode_length):
                    obs_t = torch.from_numpy(obs).float().to(device).unsqueeze(0)  # (1,H,W,C)

                    action_np, _, _, _ = model.get_action(
                        obs_t, txt_goal, goal_state, pose, deterministic=True
                    )

                    obs_list.append(obs.copy())
                    obs, reward, done, truncated, info = env.step(action_np)
                    obs = np.ascontiguousarray(obs)

                    pose = _extract_pose_from_info(info, model, device) if use_pose else None

                    actions_list.append(action_np.copy())
                    rewards.append(float(reward))
                    infos.append(info)

                    if render:
                        frame = env.render()
                        if frame is not None:
                            frames.append(frame)
                    t += 1
                    if done or truncated:
                        break

            success = float(info.get("success_placed", 0.0))
            total_successes += int(success)
            total_episodes  += 1

            avg_ep_reward = float(np.mean(rewards)) if rewards else 0.0
            ep_return     = float(sum(rewards))
            print(f"  ep {init_state_id}: return={ep_return:.3f}  "
                  f"avg_reward={avg_ep_reward:.4f}  success={success:.0f}")

            path_ = None
            if render and frames:
                path_ = os.path.join(
                    log_dir,
                    f"libero-fast-transformer-{iter_}-task-id-{task_id}-init-id-{init_state_id}.mp4"
                )
                imageio.mimsave(path_, frames, fps=20)
                last_video_path = path_

            # ---- per-episode W&B logging ----
            if not cfg.testing and wandb is not None:
                ep_log = {
                    "eval/episode":        total_episodes,
                    "eval/task_id":        int(task_id),
                    "eval/ep_return":      ep_return,
                    "eval/avg_reward":     avg_ep_reward,
                    "eval/success":        success,
                    "eval/episode_length": t,
                }
                if path_ is not None:
                    try:
                        ep_log["eval/video"] = wandb.Video(path_, fps=20, format="mp4")
                    except Exception as e:
                        print(f"Warning: failed to attach episode video to wandb: {e}")
                wandb.log(ep_log)

            trajectory_data.append({
                "task_id":       task_id,
                "init_state_id": init_state_id,
                "rewards":       rewards,
                "infos":         infos,
                "observations":  obs_list,
                "actions":       actions_list,
                "success":       success,
                "video_url":     path_,
            })

        env.close()

    avg_reward   = float(np.mean([np.mean(traj["rewards"]) for traj in trajectory_data])) \
                   if trajectory_data else 0.0
    success_rate = float(total_successes / max(1, total_episodes))

    print(f"\n[eval summary]  avg_reward={avg_reward:.4f}  "
          f"success_rate={success_rate:.2f}  ({total_successes}/{total_episodes})")

    episode_stats = {
        "rewards":      avg_reward,
        "success_rate": success_rate,
        "traj":         trajectory_data,
        "video_url":    last_video_path,
    }

    # ---- summary W&B logging ----
    if not cfg.testing and wandb is not None:
        wandb.log({
            "eval/avg_reward_summary":   avg_reward,
            "eval/success_rate_summary": success_rate,
            "eval/total_episodes":       total_episodes,
            "eval/total_successes":      total_successes,
        })
        columns = ["task_id", "init_state_id", "ep_return", "success"]
        rows    = [
            [t["task_id"], t["init_state_id"],
             float(sum(t["rewards"])), t["success"]]
            for t in trajectory_data
        ]
        wandb.log({"eval/episode_table": wandb.Table(columns=columns, data=rows)})

    return episode_stats


# ---------------------------------------------------------------------------
# Entry point — dispatches to the correct model loader and eval loop
# ---------------------------------------------------------------------------

@hydra.main(config_path="conf", config_name="sim_eval", version_base=None)
def my_main(cfg: DictConfig):
    import wandb as wandb_lib
    from omegaconf import OmegaConf
    print(OmegaConf.to_yaml(cfg))

    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = hydra.utils.to_absolute_path(cfg.checkpoint)
    model_type      = str(cfg.model_type).lower()

    print(f"[sim_eval] model_type={model_type}  checkpoint={checkpoint_path}")

    # ------------------------------------------------------------------ #
    # 0. Init W&B (skipped in dry-run / testing mode)                     #
    # ------------------------------------------------------------------ #
    run = None
    if not cfg.testing:
        run = wandb_lib.init(
            project=cfg.experiment.project,
            name=cfg.experiment.name,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=[model_type, cfg.sim.task_set],
        )
        print(f"[wandb] run initialised: {run.name}  url: {run.url}")

    # ------------------------------------------------------------------ #
    # 1. Load model                                                        #
    # ------------------------------------------------------------------ #
    if model_type == "dense_policy":
        from train_dense_rl import DensePolicy
        ckpt       = torch.load(checkpoint_path, map_location=device)
        obs_dim    = int(cfg.dense_policy.obs_dim)
        action_dim = int(cfg.dense_policy.action_dim)
        hidden_dim = int(cfg.dense_policy.hidden_dim)
        n_layers   = int(cfg.dense_policy.n_layers)
        policy = DensePolicy(obs_dim, action_dim, hidden_dim, n_layers).to(device)
        policy.load_state_dict(ckpt["policy"])
        policy.eval()
        print(f"Loaded DensePolicy  obs_dim={obs_dim}  action_dim={action_dim}")

    elif model_type == "transformer_policy":
        from train_transformer_rl import TransformerPolicyWrapper
        policy = TransformerPolicyWrapper(checkpoint_path, device, cfg)
        policy.eval()
        print(f"Loaded TransformerPolicyWrapper from {checkpoint_path}")

    else:
        raise ValueError(
            f"Unknown model_type='{model_type}'. "
            "Choose 'dense_policy' or 'transformer_policy'."
        )

    # ------------------------------------------------------------------ #
    # 2. Run evaluators                                                    #
    # ------------------------------------------------------------------ #
    all_results = {}

    try:
        if "libero_fast" in cfg.simEval:
            if model_type == "dense_policy":
                results = eval_libero_fast(
                    policy, device, cfg,
                    iter_=0, log_dir=log_dir, render=True,
                    wandb=wandb_lib if run is not None else None,
                )
            else:  # transformer_policy
                results = eval_libero_fast_transformer(
                    policy, device, cfg,
                    iter_=0, log_dir=log_dir, render=True,
                    wandb=wandb_lib if run is not None else None,
                )
            print(f"[libero_fast]  avg_reward={results['rewards']:.4f}  "
                  f"success_rate={results['success_rate']:.2f}")
            all_results["libero_fast"] = results

        if "libero" in cfg.simEval and "libero_fast" not in cfg.simEval:
            if model_type != "transformer_policy":
                raise NotImplementedError(
                    "The 'libero' evaluator uses the GRP model API and requires "
                    "model_type=transformer_policy.  Use simEval=[libero_fast] for "
                    "dense_policy evaluation."
                )
            results = eval_libero(
                policy, device, cfg,
                iter_=0, log_dir=log_dir, render=True,
            )
            print(f"[libero]  avg_reward={results['rewards']:.4f}")
            all_results["libero"] = results

    finally:
        if run is not None:
            run.finish()

    return all_results


if __name__ == "__main__":
    results = my_main()
    print("results:", results)