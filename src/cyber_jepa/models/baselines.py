"""
Required Control Baselines for Cyber-JEPA Evaluation.

Includes:
- Blue observation persistence baseline (predicts O_t as O_{t+k})
- Latent persistence baseline (predicts z_t as z_{t+k})
- Randomly initialized frozen encoder baseline
- Raw flat action-conditioned MLP future prediction baseline
"""

import torch
import torch.nn as nn
from cyber_jepa.models.predictor import ActionEncoder


class ObservationPersistenceBaseline:
    """Predicts future observation as identical to current observation (z_{t+k} = z_t)."""

    def predict(self, current_obs: torch.Tensor, horizon: int) -> torch.Tensor:
        """Returns current_obs as prediction for horizon k."""
        return current_obs.clone()


class LatentPersistenceBaseline:
    """Predicts future latent z_{t+k} as current latent z_t."""

    def predict(self, current_latent: torch.Tensor, horizon: int) -> torch.Tensor:
        """Returns current_latent as prediction for horizon k."""
        return current_latent.clone()


class RawActionConditionedMLP(nn.Module):
    """Raw flat action-conditioned MLP baseline (non-JEPA baseline)."""

    def __init__(
        self,
        obs_dim: int = 52,
        hidden_dim: int = 64,
        max_horizon: int = 16,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        self.action_encoder = ActionEncoder(hidden_dim=hidden_dim, max_horizon=max_horizon)
        self.obs_proj = nn.Linear(obs_dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, obs_dim),
        )

    def forward(self, history_obs: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        """
        Input: history_obs [B, T_hist, 52], action_seq [B, k]
        Output: predicted raw flat observation [B, 52]
        """
        B, T_hist, D = history_obs.shape
        K = action_seq.shape[1]

        # Use latest observation timestep t
        latest_obs = history_obs[:, -1, :]              # [B, 52]
        h_obs = self.obs_proj(latest_obs)               # [B, D]

        act_tokens = self.action_encoder(action_seq)    # [B, k, D]
        h_act = act_tokens.mean(dim=1)                  # [B, D] pooled action representation

        cat_in = torch.cat([h_obs, h_act], dim=-1)      # [B, 2 * D]
        pred_obs = self.mlp(cat_in)                     # [B, 52]
        return pred_obs
