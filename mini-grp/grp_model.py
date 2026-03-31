import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange


def get_patches_fast(images, cfg):
    batch_size, height, width, channels = images.shape
    patch_size = cfg.patch_size ## n_patches = 8

    patches = rearrange(images[:,:,:,:3], 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size)
    if channels > 3:
        ## History stacking in the channel dimension for observations only, not goal images.
        patches = rearrange(images, 'b (h p1) (w p2) (c hs) -> b (h w hs) (p1 p2 c)', p1 = patch_size, p2 = patch_size, hs=cfg.policy.obs_stacking) ## Stack the history in the channel dimension
    return patches


def calc_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B,T,C = x.shape
        if mask == None:
            mask = torch.ones((T, ), device=x.device) ## (1, T)
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(mask == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd=n_embd, dropout=dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        with torch.profiler.record_function("Self-Attention"):
            out = torch.cat([h(x, mask) for h in self.heads], dim=-1)
            out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd=n_embd, dropout=dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, mask=None):
        x = x + self.sa(self.ln1(x), mask)
        x = x + self.ffwd(self.ln2(x))
        return x


class GRP(nn.Module):
    def __init__(self, cfg, mlp_ratio=4):
        super(GRP, self).__init__()
        self._cfg = cfg
        chars = cfg.dataset.chars_list
        cfg.vocab_size = len(chars)
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.patch_size = (
            cfg.image_shape[0] // cfg.n_patches,
            cfg.image_shape[1] // cfg.n_patches,
        )
        cfg.patch_size = self.patch_size[0]

        # Positional embedding - compute sequence length from all token types
        text_block_size = self._cfg.max_block_size
        num_obs_tokens = (self._cfg.n_patches ** 2) * self._cfg.policy.obs_stacking
        num_goal_img_tokens = self._cfg.n_patches ** 2
        num_pose_tokens = 1 if self._cfg.policy.use_pose_data else 0
        seq_len = 1 + num_obs_tokens + num_pose_tokens + text_block_size + num_goal_img_tokens
        self.register_buffer(
            "positional_embeddings",
            calc_positional_embeddings(seq_len, cfg.n_embd),
            persistent=False,
        )

        # Learnable readout token for classification/action prediction
        self.readout_tokens = nn.Parameter(torch.randn(1, 1, cfg.n_embd) * 0.02)

        # Linear projection from patch space to embedding space
        self.input_d = int(self._cfg.image_shape[2] * self.patch_size[0] * self.patch_size[1])
        self.lin_map = nn.Linear(self.input_d, self._cfg.n_embd, bias=False)
        
        # Optional pose encoder
        pose_input_dim = (
            len(self._cfg.dataset.action_std) if hasattr(self._cfg.dataset, "action_std") else self._cfg.action_dim
        )
        if self._cfg.policy.use_pose_data:
            self.lin_map_pose = nn.Linear(pose_input_dim, self._cfg.n_embd, bias=False)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [Block(self._cfg.n_embd, self._cfg.n_head, dropout=self._cfg.dropout) for _ in range(self._cfg.n_blocks)]
        )
        self.ln_f = nn.LayerNorm(self._cfg.n_embd)

        # Action head - discrete (classification) or continuous (regression)
        self.is_discrete = hasattr(cfg, "action_bins") and cfg.action_bins is not None
        stacked_action_dim = cfg.action_dim * cfg.policy.action_stacking
        if self.is_discrete:
            self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, cfg.action_bins * stacked_action_dim),
            nn.LayerNorm(cfg.action_bins * stacked_action_dim))
        else:
            self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, stacked_action_dim),
            )


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, images, goals_txt, goal_imgs, targets=None, pose=None, mask_=False, last_action=None):
        n, c, h, w = images.shape
        block_size = self._cfg.max_block_size
        n_goal_img_tokens = self._cfg.n_patches ** 2
        obs_patches = get_patches_fast(images, self._cfg)
        patches_g = get_patches_fast(goal_imgs, self._cfg)
        out_obs = self.lin_map(obs_patches)

        has_goal_txt = goals_txt is not None
        has_goal_img = goal_imgs is not None
        
        # Process goal image patches
        if patches_g is not None:
            out_g = self.lin_map(patches_g)
        else:
            out_g = torch.zeros(n, n_goal_img_tokens, self._cfg.n_embd, device=out_obs.device)

        # Process text goal embeddings
        if goals_txt is None:
            goals_e = torch.zeros(n, block_size, self._cfg.n_embd, device=out_obs.device)
        else:
            # Convert to embeddings if needed
            if self._cfg.dataset.encode_with_t5:
                # T5 embeddings are already in embedding space
                if goals_txt.size(-1) == self._cfg.n_embd:
                    # Already embedded
                    goals_e = goals_txt.unsqueeze(0) if goals_txt.dim() == 2 else goals_txt
                else:
                    # Token IDs that need embedding
                    goals_e = goals_txt.squeeze(1) if goals_txt.dim() == 3 else goals_txt
                    goals_e = self.token_embedding_table(goals_e.long())
            else:
                # Token IDs need to be embedded
                goals_e = goals_txt.squeeze(1) if goals_txt.dim() == 3 else goals_txt
                goals_e = self.token_embedding_table(goals_e.long())
            
            # Adjust sequence length to match block_size
            current_len = goals_e.size(1)
            if current_len < block_size:
                pad = torch.zeros(n, block_size - current_len, self._cfg.n_embd, device=goals_e.device)
                goals_e = torch.cat((goals_e, pad), dim=1)
            elif current_len > block_size:
                goals_e = goals_e[:, :block_size, :]

        # Concatenate all token types
        pose_tokens = None
        if self._cfg.policy.use_pose_data:
            if pose is None:
                pose_tokens = torch.zeros(n, 1, self._cfg.n_embd, device=out_obs.device)
            else:
                pose_tokens = self.lin_map_pose(pose)
                # Ensure pose_tokens has shape (batch, 1, n_embd)
                if pose_tokens.dim() == 2:
                    pose_tokens = pose_tokens.unsqueeze(1)
        
        # Build token sequence
        if pose_tokens is not None:
            out = torch.cat((self.readout_tokens.expand(n, 1, -1), out_obs, pose_tokens, goals_e, out_g), dim=1)
        else:
            out = torch.cat((self.readout_tokens.expand(n, 1, -1), out_obs, goals_e, out_g), dim=1)

        # Adding positional embedding
        out = out + self.positional_embeddings[: out.size(1)].unsqueeze(0)

        ## Compute blocked masks
        mask = None
        if targets is not None or mask_ is True or (not has_goal_txt) or (not has_goal_img):
            seq_len = out.size(1)
            key_mask = torch.ones(seq_len, device=out.device, dtype=torch.bool)
            obs_len = out_obs.size(1)
            pose_len = 1 if self._cfg.policy.use_pose_data else 0
            text_start = 1 + obs_len + pose_len
            text_end = text_start + block_size
            img_start = text_end
            img_end = img_start + n_goal_img_tokens

            ## Randomly mask out goal modalities during training
            if targets is not None:
                rand_val = torch.rand(1, device=out.device).item()
                # equitable masking probabilities
                if rand_val > 0.9:
                    key_mask[text_start:text_end] = False
                elif rand_val > 0.1:
                    key_mask[img_start:img_end] = False

            if not has_goal_txt:
                key_mask[text_start:text_end] = False
            if not has_goal_img or mask_ is True:
                key_mask[img_start:img_end] = False

            mask = key_mask[None, None, :].expand(n, 1, seq_len)

        # Transformer Blocks
        for block in self.blocks:
            out = block(out, mask)
        out = self.ln_f(out)

        # Getting the readout token only
        out = out[:, 0]
        logits = self.mlp(out)

        # Compute output and loss
        if targets is None:
            loss = None
        else:
            if targets.dim() == 3:
                targets = targets.reshape(targets.shape[0], -1)
            B, C = targets.shape
            if self.is_discrete:
                logits = logits.view(B, self._cfg.action_bins, C)
                loss = F.cross_entropy(logits, targets)
            else:
                loss = F.mse_loss(logits, targets)
        return (logits, loss)
    
    def resize_image(self, image):
        """
        Docstring for resize_image
        
        :param self: Description
        :param image: Description
        self._resize_state = lambda sf:   cv2.resize(np.array(sf, dtype=np.float32), (cfg.image_shape[0], cfg.image_shape[1]))  # resize state
        """
        import cv2
        import numpy as _np
        img = _np.array(image, dtype=_np.float32)
        img = cv2.resize(img, (self._cfg.image_shape[0], self._cfg.image_shape[1]))
        return img

    def normalize_state(self, image):
        """
        Docstring for preprocess_state
        
        :param self: Description
        :param image: Description
        self._encode_state = lambda af:   ((af/(255.0)*2.0)-1.0) # encoder: take a float, output an integer
        self._resize_state = lambda sf:   cv2.resize(np.array(sf, dtype=np.float32), (cfg.image_shape[0], cfg.image_shape[1]))  # resize state
        """
        # img = _np.array(image, dtype=_np.float32)
        # img = cv2.resize(img, (self._cfg.image_shape[0], self._cfg.image_shape[1]))
        enc = ((image / 255.0) * 2.0) - 1.0
        # t = _torch.tensor(enc, dtype=_torch.float32, device=self._cfg.device)
        return enc
    
    def preprocess_state(self, image):
        img = self.resize_image(image)
        img = self.normalize_state(img)
        return img

    def preprocess_goal_image(self, image):
        return self.preprocess_state(image)
    
    def reset(self):
        """
        Reset the model's internal state if needed.
        """
        return None

    def encode_text_goal(self, goal, tokenizer=None, text_model=None):
        import numpy as _np
        import torch as _torch
        if self._cfg.dataset.encode_with_t5:
            if tokenizer is None or text_model is None:
                raise ValueError("tokenizer and text_model must be provided when using T5 encoding")
            goal_ = _np.zeros((self._cfg.max_block_size, self._cfg.n_embd), dtype=_np.float32)
            with _torch.no_grad():
                input_ids = tokenizer(goal, return_tensors="pt").input_ids.to(text_model.device)
                goal_t = text_model.encoder(input_ids).last_hidden_state.detach().cpu().numpy()
            goal_[: len(goal_t[0]), :] = goal_t[0][: self._cfg.max_block_size]
            return _torch.tensor(goal_, dtype=_torch.float32, device=self._cfg.device)

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

    def decode_action(self, action_tensor):
        """Decode normalized actions to original action space"""
        import torch as _torch
        action_mean = _torch.tensor(np.repeat([self._cfg.dataset.action_mean], self._cfg.policy.action_stacking, axis=0).flatten(), 
                                   dtype=action_tensor.dtype, device=action_tensor.device)
        action_std = _torch.tensor(np.repeat([self._cfg.dataset.action_std], self._cfg.policy.action_stacking, axis=0).flatten(), 
                                  dtype=action_tensor.dtype, device=action_tensor.device)
        return (action_tensor * (action_std)) + action_mean

    def encode_action(self, action_float):
        """Encode actions to normalized space [-1, 1]"""
        import torch as _torch
        ## If the action_float has length greater than action_dim then use stacking otherwise just use normal standardiaztion vectors
        if action_float.shape[1] == len(self._cfg.dataset.action_mean):
            action_mean = _torch.tensor(self._cfg.dataset.action_mean, dtype=action_float.dtype, device=action_float.device)
            action_std = _torch.tensor(self._cfg.dataset.action_std, dtype=action_float.dtype, device=action_float.device)
            return (action_float - action_mean) / (action_std)  

        action_mean = _torch.tensor(np.repeat([self._cfg.dataset.action_mean], self._cfg.policy.action_stacking, axis=0).flatten(), 
                                   dtype=action_float.dtype, device=action_float.device)
        action_std = _torch.tensor(np.repeat([self._cfg.dataset.action_std], self._cfg.policy.action_stacking, axis=0).flatten(), 
                                  dtype=action_float.dtype, device=action_float.device)
        return (action_float - action_mean) / (action_std)

    def decode_pose(self, pose_tensor):
        """Decode normalized pose to original pose space."""
        import torch as _torch
        pose_mean = _torch.tensor(self._cfg.dataset.pose_mean, dtype=pose_tensor.dtype, device=pose_tensor.device)
        pose_std = _torch.tensor(self._cfg.dataset.pose_std, dtype=pose_tensor.dtype, device=pose_tensor.device)
        return (pose_tensor * pose_std) + pose_mean

    def encode_pose(self, pose_float):
        """Encode pose to normalized space."""
        import torch as _torch
        pose_mean = _torch.tensor(self._cfg.dataset.pose_mean, dtype=pose_float.dtype, device=pose_float.device)
        pose_std = _torch.tensor(self._cfg.dataset.pose_std, dtype=pose_float.dtype, device=pose_float.device)
        return (pose_float - pose_mean) / pose_std
    
    def decode_state(self, state_tensor):
        """
        Docstring for decode_state
        
        :param self: Description
        :param state_tensor: Description
        self._decode_state = lambda sinN: (sinN * state_std) + state_mean  # Undo mapping to [-1, 1]
        """
        import torch as _torch
        state_mean = _torch.tensor(self._cfg.dataset.state_mean, dtype=state_tensor.dtype, device=state_tensor.device)
        state_std = _torch.tensor(self._cfg.dataset.state_std, dtype=state_tensor.dtype, device=state_tensor.device)
        return (state_tensor * (state_std)) + state_mean
    
    def encode_state(self, state_float):
        """
        Docstring for encode_state
        
        :param self: Description
        :param state_float: Description
        self._encode_state = lambda sf:   (sf - state_mean)/(state_std) # encoder: take a float, output an integer
        """
        import torch as _torch
        state_mean = _torch.tensor(self._cfg.dataset.state_mean, dtype=state_float.dtype, device=state_float.device)
        state_std = _torch.tensor(self._cfg.dataset.state_std, dtype=state_float.dtype, device=state_float.device)
        return (state_float - state_mean) / (state_std)


@torch.no_grad()
def estimate_loss(model, dataset):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(model._cfg.eval_iters)
        for k in range(model._cfg.eval_iters):
            X, x_pose, x_goal, x_goal_img, Y, last_action = dataset.get_batch_grp(split, model._cfg, model._cfg.batch_size)
            logits, loss = model(X, x_goal, x_goal_img, Y, pose=x_pose, last_action=last_action)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out