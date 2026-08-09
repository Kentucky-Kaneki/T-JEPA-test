"""
Flat Representation Controlled Ablations for Cyber-JEPA.

Contains:
1. FlatShuffledTimeRepresentation: Permutes temporal history order per sample.
2. FlatShuffledFeaturesRepresentation: Applies deterministic feature order permutation.
3. FlatCurrentOnlyRepresentation: Encodes only instantaneous current frame x_t (h=1).
4. FlatVariableHistoryRepresentation: Configurable history length h in {1, 2, 4, 8}.
"""

import torch
import torch.nn as nn
from cyber_jepa.representations.flat import FlatVectorRepresentation


class FlatShuffledTimeRepresentation(nn.Module):
    """Ablation 2A: Randomly permutes temporal history before encoding."""

    def __init__(
        self,
        obs_dim: int = 52,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        ffn_dim: int = 256,
        history_len: int = 4,
        seed: int = 42,
    ):
        super().__init__()
        self.history_len = history_len
        self.seed = seed
        self.base_encoder = FlatVectorRepresentation(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            history_len=history_len,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: x of shape [B, T_hist, 52]
        Shuffles temporal dimension T_hist independently per sample.
        """
        B, T_hist, D = x.shape
        device = x.device

        # Deterministic generator seeded for reproducibility
        gen = torch.Generator(device="cpu").manual_seed(self.seed)

        shuffled_x = torch.zeros_like(x)
        for b in range(B):
            perm = torch.randperm(T_hist, generator=gen).to(device)
            shuffled_x[b] = x[b, perm, :]

        return self.base_encoder(shuffled_x)


class FlatShuffledFeaturesRepresentation(nn.Module):
    """Ablation 2A: Consistently permutes feature order within observations."""

    def __init__(
        self,
        obs_dim: int = 52,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        ffn_dim: int = 256,
        history_len: int = 4,
        seed: int = 42,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.history_len = history_len
        self.seed = seed

        # Generate fixed feature permutation
        gen = torch.Generator(device="cpu").manual_seed(seed)
        perm = torch.randperm(obs_dim, generator=gen)
        self.register_buffer("feature_perm", perm)

        self.base_encoder = FlatVectorRepresentation(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            history_len=history_len,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: x of shape [B, T_hist, 52]
        Permutes feature dimension according to feature_perm.
        """
        shuffled_x = x[:, :, self.feature_perm]
        return self.base_encoder(shuffled_x)


class FlatCurrentOnlyRepresentation(nn.Module):
    """Ablation 2C: Receives only current instantaneous state x_t (h=1)."""

    def __init__(
        self,
        obs_dim: int = 52,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        ffn_dim: int = 256,
        history_len: int = 1,
    ):
        super().__init__()
        self.history_len = 1
        self.base_encoder = FlatVectorRepresentation(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            history_len=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: x of shape [B, T_hist, 52] or [B, 1, 52]
        Takes only the last timestep x_t = x[:, -1:, :].
        """
        x_current = x[:, -1:, :]  # [B, 1, 52]
        return self.base_encoder(x_current)


class FlatVariableHistoryRepresentation(nn.Module):
    """Ablation 2B: Flat representation with variable history length h in {1, 2, 4, 8}."""

    def __init__(
        self,
        history_len: int = 4,
        obs_dim: int = 52,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        ffn_dim: int = 256,
    ):
        super().__init__()
        self.history_len = history_len
        self.base_encoder = FlatVectorRepresentation(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            history_len=history_len,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: x of shape [B, T_hist, 52]
        Truncates or takes last history_len steps: x[:, -self.history_len:, :].
        """
        if x.shape[1] > self.history_len:
            x_sliced = x[:, -self.history_len:, :]
        else:
            x_sliced = x
        return self.base_encoder(x_sliced)
