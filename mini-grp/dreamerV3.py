import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical, Normal, Bernoulli, Independent, OneHotCategoricalStraightThrough, kl_divergence
from torch.distributions.utils import probs_to_logits
import numpy as np
try:
    # When imported as a package module: `from hw2.dreamerV3 import DreamerV3`
    from .networks import (
        EncoderConv,
        DecoderConv,
        RecurrentModel,
        PriorNet,
        PosteriorNet,
        RewardPredictor,
        ContinuePredictor,
        ActorNet,
        CriticNet,
    )
except ImportError:
    # When executed or imported with cwd=hw2/: `from dreamerV3 import DreamerV3`
    from networks import (
        EncoderConv,
        DecoderConv,
        RecurrentModel,
        PriorNet,
        PosteriorNet,
        RewardPredictor,
        ContinuePredictor,
        ActorNet,
        CriticNet,
    )

def symlog(x):
    """
    Symmetric log transformation.
    Squashes large values while preserving sign and small values.
    y = sign(x) * ln(|x| + 1)
    """
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    """Inverse of symlog.

    If y = symlog(x) = sign(x) * log(|x|+1), then x = symexp(y) = sign(y) * (exp(|y|)-1).
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

class GRPBase(nn.Module):
    """Base class for GRP models"""
    def __init__(self, cfg):
        super(GRPBase, self).__init__()
        self._cfg = cfg
        self._action_mean = torch.tensor(self._cfg.dataset.action_mean, dtype=torch.float32, device=self._cfg.device)
        self._action_std = torch.tensor(self._cfg.dataset.action_std, dtype=torch.float32, device=self._cfg.device)
        self._stacking_action_mean = torch.tensor(np.repeat([self._cfg.dataset.action_mean], self._cfg.policy.action_stacking, axis=0).flatten(), 
                                                 dtype=torch.float32, device=self._cfg.device)
        self._stacking_action_std = torch.tensor(np.repeat([self._cfg.dataset.action_std], self._cfg.policy.action_stacking, axis=0).flatten(), 
                                                dtype=torch.float32, device=self._cfg.device)
        self._pose_mean = torch.tensor(self._cfg.dataset.pose_mean, dtype=torch.float32, device=self._cfg.device)
        self._pose_std = torch.tensor(self._cfg.dataset.pose_std, dtype=torch.float32, device=self._cfg.device)

    def encode_text_goal(self, goal, tokenizer=None, text_model=None):
        import numpy as _np
        import torch as _torch
        if self._cfg.dataset.encode_with_t5:
            if tokenizer is None or text_model is None:
                raise ValueError("tokenizer and text_model must be provided when using T5 encoding")
            # TODO:    
            ## Provide the logic converting text goal to T5 embedding tensor
            pass
        else:
            pad = " " * self._cfg.max_block_size
            goal_ = goal[:self._cfg.max_block_size] + pad[len(goal):self._cfg.max_block_size]
            try:
                stoi = {c: i for i, c in enumerate(self._cfg.dataset.chars_list)}
                ids = [stoi.get(c, 0) for c in goal_]
            except Exception:
                ids = [0] * self._cfg.max_block_size
            return _torch.tensor(_np.expand_dims(_np.array(ids, dtype=_np.int64), axis=0), dtype=_torch.long, device=self._cfg.device)

    def process_text_embedding_for_buffer(self, goal, tokenizer=None, text_model=None):
        """
        Process text goal embedding for storing in the circular buffer.
        Returns a numpy array of shape (max_block_size, n_embd) without batch dimension.
        """
        import numpy as _np
        if tokenizer is None or text_model is None:
            raise ValueError("tokenizer and text_model must be provided when using T5 encoding")
        
        goal_ = _np.zeros((self._cfg.max_block_size, self._cfg.n_embd), dtype=_np.float32)
        input_ids = tokenizer(goal, return_tensors="pt").input_ids
        goal_t = text_model.encoder(input_ids).last_hidden_state.detach().cpu().numpy()
        goal_[:len(goal_t[0]), :] = goal_t[0][:self._cfg.max_block_size]
        return goal_


    def resize_image(self, image):
        """Resize image to match model input size"""
        import cv2
        import numpy as _np
        img = _np.array(image, dtype=_np.float32)
        img = cv2.resize(img, (self._cfg.image_shape[0], self._cfg.image_shape[1]))
        return img

    def normalize_state(self, image):
        """Normalize image to [-1, 1] range"""
        enc = ((image / 255.0) * 2.0) - 1.0
        return enc
    
    def preprocess_state(self, image):
        """Preprocess observation image"""
        img = self.resize_image(image)
        img = self.normalize_state(img)
        ## Change numpy array from channel-last to channel-first
        img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        # img = img.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        return img

    def preprocess_goal_image(self, image):
        """Preprocess goal image"""
        return self.preprocess_state(image)

    def decode_action(self, action_tensor):
        """Decode normalized actions to original action space"""
        import torch as _torch
        action_mean = self._stacking_action_mean
        action_std = self._stacking_action_std
        return (action_tensor * (action_std)) + action_mean

    def encode_action(self, action_float):
        """Encode actions to normalized space [-1, 1]"""
        import torch as _torch
        ## If the action_float has length greater than action_dim then use stacking otherwise just use normal standardiaztion vectors
        if action_float.shape[1] == len(self._cfg.dataset.action_mean):
            action_mean = self._action_mean
            action_std = self._action_std
            return (action_float - action_mean) / (action_std)  

        action_mean = self._stacking_action_mean
        action_std = self._stacking_action_std
        return (action_float - action_mean) / (action_std)
    
    def decode_pose(self, pose_tensor):
        """
        Docstring for decode_pose
        
        :param self: Description
        :param pose_tensor: Description
        self._decode_state = lambda sinN: (sinN * state_std) + state_mean  # Undo mapping to [-1, 1]
        """
        import torch as _torch
        pose_mean = self._pose_mean
        pose_std = self._pose_std
        return (pose_tensor * (pose_std)) + pose_mean
    
    def encode_pose(self, pose_float):
        """
        Docstring for encode_pose
        
        :param self: Description
        :param pose_float: Description
        self._encode_pose = lambda pf:   (pf - pose_mean)/(pose_std) # encoder: take a float, output an integer
        """
        import torch as _torch
        pose_mean = self._pose_mean
        pose_std = self._pose_std
        return (pose_float - pose_mean) / (pose_std)

class DreamerV3(GRPBase):
    def __init__(self, 
                 obs_shape=(3, 128, 128),  # Updated default to match your error
                 action_dim=6, 
                 stoch_dim=32, 
                 discrete_dim=32, 
                 deter_dim=512, 
                 hidden_dim=512, cfg=None):
        """DreamerV3 model implementation.
        
        Args:
            obs_shape: Shape of input observations (C, H, W)
            action_dim: Dimension of action space
            stoch_dim: Dimension of stochastic latent state
            discrete_dim: Number of discrete categories for stochastic state
            deter_dim: Dimension of deterministic state
            hidden_dim: Dimension of hidden layers in encoder/decoder
            cfg: Configuration object for model hyperparameters and settings
        """
        # TODO: Part 3.1 - Initialize DreamerV3 architecture
        ## Define encoder, RSSM components (GRU, prior/posterior nets), and decoder heads
        super(DreamerV3, self).__init__(cfg)
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.stoch_dim = stoch_dim  # Latent dimension for stochastic state
        self.discrete_dim = discrete_dim    # Latent dimension for discrete representation predicted by a sequence model
        self.deter_dim = deter_dim  # Recurrent dimension for deterministic state
        self.hidden_dim = hidden_dim    # Hidden dimension for encoder/decoder networks
        self.encodedObsSize = hidden_dim + deter_dim  # Output dimension of the encoder
        # Define encoder for images
        self.encoder = EncoderConv(input_dim=obs_shape, output_dim=hidden_dim)
        
        # GRU for deterministic state update
        self.recurrent_net = RecurrentModel(recurrent_dim=deter_dim, latent_dim=stoch_dim*discrete_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        # Prior and posterior networks for stochastic state
        self.prior_net = PriorNet(input_dim=deter_dim, latent_dim=stoch_dim, latent_classes=discrete_dim, hidden_dim=hidden_dim)
        self.post_net = PosteriorNet(input_dim=self.encodedObsSize, latent_dim=stoch_dim, latent_classes=discrete_dim, hidden_dim=hidden_dim)
        # Decoder heads for reconstruction, reward, and continue prediction
        self.decoder = DecoderConv(input_dim=deter_dim + stoch_dim * discrete_dim, output_dim=obs_shape) # Reconstruct image from combined deterministic and stochastic state
        self.reward_head = RewardPredictor(input_dim=deter_dim + stoch_dim * discrete_dim, hidden_dim=hidden_dim)  # Predict reward from combined state
        self.continue_head = ContinuePredictor(input_dim=deter_dim + stoch_dim * discrete_dim, hidden_dim=hidden_dim)  # Predict continue flag from combined state

        # Dreamer-style actor/critic (not used by the current trainer yet).
        # These operate on RSSM features feat = concat([h,z]).
        feat_dim = deter_dim + stoch_dim * discrete_dim
        self.actor = ActorNet(input_dim=feat_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        self.critic = CriticNet(input_dim=feat_dim, hidden_dim=hidden_dim)

        self.type = 'dreamerV3'
        self.device = self._cfg.device
        self.to(self.device)

    # ... [Helper methods same as before] ...

    def get_initial_state(self, batch_size, device):
        return {
            'h': torch.zeros(batch_size, self.deter_dim, device=device),
            'z': torch.zeros(batch_size, self.stoch_dim * self.discrete_dim, device=device),
            'z_probs': torch.zeros(batch_size, self.stoch_dim, self.discrete_dim, device=device)
        }

    def sample_stochastic(self, logits, training=True):
        # TODO: Part 3.1 - Implement stochastic sampling
        ## Sample from discrete categorical distribution using logits
        # Expected logits shape: (B, stoch_dim, discrete_dim) or (B, stoch_dim * discrete_dim)
        if logits.dim() == 2:
            B = logits.shape[0]
            logits = logits.view(B, self.stoch_dim, self.discrete_dim)
        elif logits.dim() != 3:
            raise ValueError(
                f"logits must have shape (B, stoch_dim, discrete_dim) or (B, stoch_dim*discrete_dim); got {tuple(logits.shape)}"
            )

        # Sample directly from the raw network logits.
        # IMPORTANT: do NOT do a probs_to_logits(softmax(logits)) round-trip here.
        # That conversion introduces numerical error that accumulates across time steps
        # and makes the KL (rep_loss / dyn_loss) measure a slightly different distribution
        # from what was actually sampled, causing rep_loss to diverge.

        # Unimix smoothing (DreamerV3 paper §A): blend categorical probs with uniform
        # to ensure no class has p=0, which would make KL = inf.
        unimix = 0.01
        probs = torch.softmax(logits, dim=-1)
        uniform = torch.ones_like(probs) / self.discrete_dim
        probs = (1.0 - unimix) * probs + unimix * uniform
        # Convert smoothed probs back to logits for sampling
        smooth_logits = probs_to_logits(probs)

        if training:
            # Straight-through one-hot categorical sampling so gradients can flow.
            z = Independent(OneHotCategoricalStraightThrough(logits=smooth_logits), 1).rsample()  # (B, stoch_dim, discrete_dim)
        else:
            # Deterministic mode for evaluation/planning.
            idx = torch.argmax(smooth_logits, dim=-1)  # (B, stoch_dim)
            z = torch.nn.functional.one_hot(idx, num_classes=self.discrete_dim).to(logits.dtype)

        # Flatten to match the rest of the model which expects (B, stoch_dim * discrete_dim)
        z_flat = z.view(z.shape[0], self.stoch_dim * self.discrete_dim)
        # Return probs for state bookkeeping (z_probs is only used for get_initial_state / logging).
        return z_flat, probs, smooth_logits

    def rssm_step(self, prev_state, action, embed=None):
        # TODO: Part 3.1 - Implement RSSM step
        ## Update deterministic state (h) with GRU, compute prior and posterior distributions
        # prev_state: dict with keys {'h', 'z', 'z_probs'}
        # action: (B, action_dim)
        # embed: encoded observation features (B, hidden_dim) or None
        if action.dim() > 2:
            # If a (B, T, A) slipped through, we only support one step here.
            action = action[:, 0]

        h_prev = prev_state['h']  # (B, deter_dim)
        z_prev = prev_state['z']  # (B, stoch_dim * discrete_dim)

        # 1) Deterministic state update via GRU
        h = self.recurrent_net(h_prev, z_prev, action)  # (B, deter_dim): the new deterministic state after observing the previous state and action

        # 2) Prior over stochastic state from new deterministic state
        prior_logits = self.prior_net(h)  # (B, stoch_dim, discrete_dim): the prior distribution over the stochastic state based on the new deterministic state
        z, z_probs, smooth_prior_logits = self.sample_stochastic(prior_logits, training=self.training)    # Sample from the prior during imagination/rollout, and use argmax during evaluation
        # 3) If we have an observation embedding, compute posterior and sample from it.
        if embed is not None:
            # Posterior conditions on [h, embed]
            if embed.dim() > 2:
                embed = embed[:, 0]
            post_in = torch.cat([h, embed], dim=-1) # (B, deter_dim + hidden_dim): combine deterministic state and observation embedding for posterior computation
            post_logits = self.post_net(post_in)  # (B, stoch_dim, discrete_dim): the posterior distribution over the stochastic state based on the new deterministic state and the current observation embedding
            z, z_probs, smooth_post_logits = self.sample_stochastic(post_logits, training=self.training)    # Sample from the posterior during training, and use argmax during evaluation
            return {
                'h': h,
                'z': z,
                'z_probs': z_probs,
                'prior_logits': smooth_prior_logits,
                'post_logits': smooth_post_logits,
            }

        # 4) Otherwise, sample from the prior (imagination / rollout)
        return {
            'h': h,
            'z': z,
            'z_probs': z_probs,
            'prior_logits': smooth_prior_logits,  # Return the smoothed prior logits for KL computation in compute_loss
            'post_logits': None,
        }

    def forward(self, normalized_observations, prev_actions=None, prev_state=None,
                mask_=True, pose=None, last_action=None,
                text_goal=None, goal_image=None):
        # TODO: Part 3.2 - Implement DreamerV3 forward pass
        ## Encode images, unroll RSSM, and compute reconstructions and heads
        # normalized_observations: (B, T, C, H, W)
        # prev_actions: (B, T, action_dim)
        if normalized_observations is None:
            raise ValueError("DreamerV3.forward requires `normalized_observations` (B, T, C, H, W)")
        if prev_actions is None:
            raise ValueError("DreamerV3.forward requires `prev_actions` (B, T, action_dim)")

        if normalized_observations.dim() != 5:
            raise ValueError(f"normalized_observations must have shape (B, T, C, H, W); got {tuple(normalized_observations.shape)}")
        if prev_actions.dim() != 3:
            raise ValueError(f"prev_actions must have shape (B, T, A); got {tuple(prev_actions.shape)}")

        B, T, C, H, W = normalized_observations.shape
        device = normalized_observations.device

        # Initialize RSSM state
        state = prev_state if prev_state is not None else self.get_initial_state(B, device=device)

        # Encode normalized_observations frame-wise.
        # Normalization to [-1,1] is expected to have been done by the caller
        # (e.g. dreamer_model_trainer passes normalized_images).
        # Do NOT call preprocess_state() here: it operates on numpy arrays via
        # cv2 and would crash / corrupt a GPU tensor.
        obs_flat = normalized_observations.reshape(B * T, C, H, W)  # (B*T, C, H, W)
        embed_flat = self.encoder(obs_flat)  # (B*T, hidden_dim)
        embed = embed_flat.view(B, T, -1)    # (B, T, hidden_dim)

        hs, zs, z_probs_list = [], [], []   # hs: hidden states, zs: sampled stochastic states, z_probs_list: categorical probabilities for KL computation
        priors_logits, posts_logits = [], []    # priors_logits/posts_logits: for KL divergence losses in compute_loss
        rewards, continues = [], []     # rewards: predicted rewards, continues: predicted continue logits
        features = []   # features: combined [h,z] for each time step, used for actor/critic heads if needed

        # The RSSM convention: h_t is computed from (h_{t-1}, z_{t-1}, a_{t-1}).
        # So at step t we feed the *previous* action a_{t-1}.
        # At t=0 the previous action is zeros (already encoded in the initial state).
        zero_action = torch.zeros(B, self.action_dim, device=device, dtype=prev_actions.dtype)

        for t in range(T):
            # Previous action: zeros at t=0, else prev_actions[:, t-1, :]
            a_t = zero_action if t == 0 else prev_actions[:, t - 1, :]   # (B, action_dim): action taken *before* observing step t
            e_t = embed[:, t, :]    # (B, hidden_dim): the encoded observation at time t

            step_out = self.rssm_step(state, a_t, embed=e_t)    # (B, deter_dim), (B, stoch_dim*discrete_dim), (B, stoch_dim, discrete_dim), (B, stoch_dim, discrete_dim)
            state = {
                'h': step_out['h'],  # (B, deter_dim): the new deterministic state
                'z': step_out['z'],  # (B, stoch_dim*discrete_dim): the sampled stochastic state (flattened)
                'z_probs': step_out['z_probs'], # (B, stoch_dim, discrete_dim): the probabilities of the categorical distribution for KL divergence computation
            }

            h_t = step_out['h']     # (B, deter_dim): the new deterministic state
            z_t = step_out['z']     # (B, stoch_dim*discrete_dim): the sampled stochastic state (flattened)
            # feat_t = torch.cat([h_t, z_t], dim=-1)  # (B, deter_dim + stoch_dim*discrete_dim): the combined feature vector for decoding and heads

            hs.append(h_t) 
            zs.append(z_t)
            z_probs_list.append(step_out['z_probs'])
            priors_logits.append(step_out['prior_logits'])
            posts_logits.append(step_out['post_logits'])
            
            # features.append(feat_t)
        # Stack temporal lists into tensors: (B, T, ...)
        recurrentStates = torch.stack(hs, dim=1)  # (B, T, deter_dim)
        posteriors = torch.stack(zs, dim=1)  # (B, T, stoch_dim*discrete_dim)
        priors_stacked = torch.stack(priors_logits, dim=1)   # (B, T, stoch_dim, discrete_dim)
        posts_stacked = torch.stack(posts_logits, dim=1)  # (B, T, stoch_dim, discrete_dim)
        # print(f"recurrentStates shape: {recurrentStates.shape}, posteriors shape: {posteriors.shape}")
        full_state = torch.cat((recurrentStates, posteriors), dim=-1)  # (B, T, deter_dim + stoch_dim*discrete_dim)
        # Heads
        reconstructions = self.decoder(full_state.view(B*T, -1)).view(B, T, *self.obs_shape)  # (B, T, C, H, W)
        rewards_pred = self.reward_head(full_state.view(B*T, -1)).view(B, T)  # (B, T)
        continues_logits = self.continue_head(full_state.view(B*T, -1)).view(B, T)  # (B, T)

        # Stack time dimension
        out = {
            'reconstructions': reconstructions,         # (B, T, C, H, W)
            'rewards': rewards_pred,                         # (B, T)
            'continues': continues_logits,              # (B, T) (logits)
            'priors_logits': priors_stacked,            # (B, T, stoch_dim, discrete_dim)
            'posts_logits': posts_stacked,              # (B, T, stoch_dim, discrete_dim)
            'h': torch.stack(hs, dim=1),
            'z': torch.stack(zs, dim=1),
            'z_probs': torch.stack(z_probs_list, dim=1),
            'final_state': state,
        }
        return out
    
    def compute_loss(self, output, normalized_images, rewards, dones, device):
        """
        Compute the total loss for DreamerV3 model training.
        
        Args:
            output: Dictionary containing model outputs (reconstructions, rewards, continues, priors_logits, posts_logits)
            normalized_images: Ground truth normalized_images tensor
            rewards: Ground truth rewards tensor
            dones: Ground truth done flags tensor
            device: Device to perform computations on
            pred_coeff: Coefficient for prediction losses (reconstruction + reward + continue)
            dyn_coeff: Coefficient for dynamics loss
            rep_coeff: Coefficient for representation loss
        
        Returns:
            Dictionary containing:
                - total_loss: Combined weighted loss
                - recon_loss: Reconstruction loss
                - reward_loss: Reward prediction loss
                - continue_loss: Continue prediction loss
                - dyn_loss: Dynamics loss (KL divergence)
                - rep_loss: Representation loss (KL divergence)
        """
        # TODO: Part 3.2 - Implement DreamerV3 loss computation
        ## Compute reconstruction, reward, KL divergence losses and combine them
        # Shapes we expect:
        # - normalized_images: (B, T, C, H, W)
        # - rewards: (B, T) or (B, T, 1)
        # - dones: (B, T) or (B, T, 1)
        # - output['reconstructions']: (B, T, C, H, W)
        # - output['rewards']: (B, T)
        # - output['continues']: (B, T) (logits)
        # - output['priors_logits']: (B, T, Z, C)
        # - output['posts_logits']: (B, T, Z, C)

        pred_coeff = float(getattr(getattr(self._cfg, 'loss_coeffs', {}), 'pred_coeff', 1.0))
        dyn_coeff = float(getattr(getattr(self._cfg, 'loss_coeffs', {}), 'dyn_coeff', 1.0))
        rep_coeff = float(getattr(getattr(self._cfg, 'loss_coeffs', {}), 'rep_coeff', 0.1))
        # Optional KL free nats
        free_nats = 1.0

        reconstructions = output['reconstructions']
        rewards_pred = output['rewards']
        continues_pred = output['continues']
        priors_logits = output['priors_logits']
        posts_logits = output['posts_logits']

        if rewards.dim() == 3 and rewards.shape[-1] == 1:
            rewards = rewards.squeeze(-1)
        if dones.dim() == 3 and dones.shape[-1] == 1:
            dones = dones.squeeze(-1)

        # --- Reconstruction loss (pixel MSE in normalized space) ---
        # Image Reconstruction Loss (MSE)
        recon_loss = F.mse_loss(reconstructions, normalized_images)

        # --- Reward prediction loss ---
        # DreamerV3 commonly models reward/value in symlog-space for stability,
        # especially when rewards are heavy-tailed or mostly negative (cost-style).
        reward_target = symlog(rewards)
        reward_loss = F.mse_loss(rewards_pred, reward_target)

        # Continue Predictor Loss (Binary Cross Entropy)
        # Model predicts whether the episode continues (1) or is done (0)
        continues_target = 1.0 - dones.float().squeeze(-1)
        continue_loss = F.binary_cross_entropy_with_logits(continues_pred, continues_target)

        # --- KL losses between posterior and prior categorical latents ---
        # Use raw network logits directly — no softmax/probs_to_logits round-trip.
        # The round-trip would introduce numerical error that makes the KL measure
        # a different distribution from what was sampled, causing rep_loss to diverge.
        prior_distribution    = Independent(OneHotCategoricalStraightThrough(logits=priors_logits), 1)
        prior_distribution_sg = Independent(OneHotCategoricalStraightThrough(logits=priors_logits.detach()), 1)
        post_distribution     = Independent(OneHotCategoricalStraightThrough(logits=posts_logits), 1)
        post_distribution_sg  = Independent(OneHotCategoricalStraightThrough(logits=posts_logits.detach()), 1)

        # dyn_loss: trains the prior  → KL(sg(post) ‖ prior)
        # rep_loss: trains the posterior → KL(post ‖ sg(prior))
        # Apply free_nats AFTER averaging so the floor is on the scalar loss,
        # not per-element (per-element flooring permanently hides improvement).
        dyn_loss = torch.maximum(
            kl_divergence(post_distribution_sg, prior_distribution),
            torch.tensor(free_nats, device=device)
        ).mean()
        rep_loss = torch.maximum(
            kl_divergence(post_distribution, prior_distribution_sg),
            torch.tensor(free_nats, device=device)
        ).mean()


        pred_loss = recon_loss + reward_loss + continue_loss
        total_loss = pred_coeff * pred_loss + dyn_coeff * dyn_loss + rep_coeff * rep_loss

        losses = {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'reward_loss': reward_loss,
            'continue_loss': continue_loss,
            'dyn_loss': dyn_loss,
            'rep_loss': rep_loss,
        }

        return losses

