import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli, Independent, OneHotCategoricalStraightThrough
from torch.distributions.utils import probs_to_logits


class EncoderConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EncoderConv, self).__init__()
        self.input_dim = input_dim
        channels, height, width = input_dim
        self.output_dim = output_dim
        self.conv_net = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1),  # (B, 32, H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (B, 64, H/4, W/4)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (B, 128, H/8, W/8)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # (B, 256, H/16, W/16)
            nn.ReLU(),
            nn.Flatten(),  # (B, 256 * H/16 * W/16)
            nn.Linear(256 * (height // 16) * (width // 16), output_dim),
            # No activation here: the embedding should be unbounded so the
            # posterior net (PosteriorNet) receives the full-range representation.
            # A final ReLU would zero-clamp half the features and starve the
            # posterior of information, causing recon_loss to increase.
        )

    def forward(self, x):
        return self.conv_net(x).view(x.size(0), self.output_dim)

class DecoderConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DecoderConv, self).__init__()
        self.input_dim = input_dim
        channels, height, width = output_dim
        self.deconv_net = nn.Sequential(
            nn.Linear(input_dim, 256 * (height // 16) * (width // 16)),
            nn.ReLU(),
            nn.Unflatten(1, (256, height // 16, width // 16)),  # (B, 256, H/16, W/16)
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 128, H/8, W/8)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 64, H/4, W/4)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 32, H/2, W/2)
            nn.ReLU(),
            nn.ConvTranspose2d(32, channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, channels, H, W)
            # nn.Tanh()  # Assuming output is normalized to [-1, 1]
        )

    def forward(self, x):
        return self.deconv_net(x)

class RecurrentModel(nn.Module):
    def __init__(self, recurrent_dim, latent_dim, action_dim, hidden_dim):
        """Recurrent model architecture for processing sequences of features and actions.
       
        Args:
            recurrent_dim: Dimensionality of the input features to the RNN (e.g., concatenated pose and action features)
            latent_dim: Dimensionality of the latent state representation
            action_dim: Dimensionality of the action space (output dimension of the model)
            hidden_dim: Dimensionality of the hidden state in the RNN
        """
        super(RecurrentModel, self).__init__()
        self.recurrent_dim = recurrent_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.activation = nn.ReLU()
        self.linear = nn.Linear(latent_dim + action_dim, hidden_dim)
        self.recurrent = nn.GRUCell(hidden_dim, recurrent_dim)

    def forward(self, recurrent_state, latent_state, action):
        """Forward pass for the recurrent model.
        
        Args:
            recurrent_state: Current hidden state of the RNN (B, recurrent_dim)
            latent_state: Current latent state representation (B, latent_dim)
            action: Current action input (B, action_dim)

        """
        # Concatenate latent state and action input
        x = torch.cat([latent_state, action], dim=-1)
        x = self.linear(x)
        x = self.activation(x)
        hidden = self.recurrent(x, recurrent_state)
        return hidden


class PriorNet(nn.Module):
    def __init__(self, input_dim, latent_dim, latent_classes, hidden_dim):
        super(PriorNet, self).__init__()
        self.latent_dim = latent_dim
        self.latent_classes = latent_classes
        self.latent_size = latent_dim * latent_classes
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.latent_size)
        )

    def forward(self, x):
        """Return logits for the prior categorical latents.

        NOTE: Sampling is handled by the caller (e.g., DreamerV3.sample_stochastic)
        to avoid duplicated sampling logic and keep gradients consistent.
        """
        raw_logits = self.net(x)  # (B, latent_dim * latent_classes)
        logits = raw_logits.view(-1, self.latent_dim, self.latent_classes)  # (B, Z, C)
        # Clamp logits to prevent extreme values that make KL(post||prior) → inf.
        # Without clamping, after ~1000 steps unbounded logits cause dyn_loss to explode.
        return logits

class PosteriorNet(nn.Module):
    def __init__(self, input_dim, latent_dim, latent_classes, hidden_dim):
        super(PosteriorNet, self).__init__()
        self.latent_dim = latent_dim
        self.latent_classes = latent_classes
        self.latent_size = latent_dim * latent_classes
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.latent_size)
        )

    def forward(self, x):
        """Return logits for the posterior categorical latents.

        Sampling is handled by the caller (e.g., DreamerV3.sample_stochastic).
        """
        raw_logits = self.net(x)  # (B, latent_dim * latent_classes)
        logits = raw_logits.view(-1, self.latent_dim, self.latent_classes)  # (B, Z, C)
        # Clamp logits to keep KL finite. Matches the clamp in PriorNet.
        return logits

class RewardPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RewardPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

class ContinuePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ContinuePredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        logits = self.net(x).squeeze(-1)
        return logits

class ActorNet(nn.Module):
    """Simple Gaussian actor head for Dreamer.

    Takes Dreamer features (concat([h,z])) and outputs a Normal distribution over actions.
    We clamp log_std for stability.
    """

    def __init__(self, input_dim, action_dim, hidden_dim, actionLow = -1.0, actionHigh = 1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2),
        )
        # Keep these buffers device-agnostic; they will move with `.to(device)`.
        # Support either scalars or per-dimension bounds.
        low = torch.as_tensor(actionLow, dtype=torch.float32)
        high = torch.as_tensor(actionHigh, dtype=torch.float32)
        if low.numel() == 1:
            low = low.repeat(action_dim)
        if high.numel() == 1:
            high = high.repeat(action_dim)
        self.register_buffer("actionScale", (high - low) / 2.0)
        self.register_buffer("actionBias", (high + low) / 2.0)

    def forward(self, x, training=False):
        logStdMin, logStdMax = -5, 2
        mean, logStd = self.net(x).chunk(2, dim=-1)
        logStd = logStdMin + (logStdMax - logStdMin)/2*(torch.tanh(logStd) + 1) # (-1, 1) to (min, max)
        std = torch.exp(logStd)

        distribution = Normal(mean, std)
        sample = distribution.sample()
        sampleTanh = torch.tanh(sample)
        action = sampleTanh*self.actionScale + self.actionBias
        if training:
            logprobs = distribution.log_prob(sample)
            logprobs -= torch.log(self.actionScale*(1 - sampleTanh.pow(2)) + 1e-6)
            entropy = distribution.entropy()
            return action, logprobs.sum(-1), entropy.sum(-1)
        else:
            return action

class CriticNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CriticNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        mean, logStd = self.net(x).chunk(2, dim=-1)
        return Normal(mean.squeeze(-1), torch.exp(logStd).squeeze(-1))