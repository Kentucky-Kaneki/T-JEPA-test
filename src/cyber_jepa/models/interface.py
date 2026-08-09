"""
Public Defender World Model Protocol Interface.

Exposes typed prediction interface P(z_{t+k} | O_{t-h+1:t}^{Blue}, a_{t:t+k-1}^{Blue})
for downstream planners without embedding policy selection or action optimization logic.
"""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable
import torch


@dataclass
class TargetSpec:
    """Target specification for world model latent prediction."""
    granularity: str                         # 'feature', 'host', 'subnet', 'network'
    target_id: str                           # entity identifier
    query_indices: list[int]


@dataclass
class LatentContext:
    """Encoded history representation context."""
    context_tokens: torch.Tensor             # [B, N_tokens, D] or [B, D]
    history_len: int
    metadata: dict[str, Any]


@dataclass
class LatentPrediction:
    """Predicted future latent representation."""
    predicted_latent: torch.Tensor           # [B, D]
    validity_mask: torch.Tensor              # [B, N_tokens]
    target: TargetSpec
    horizon: int


@runtime_checkable
class DefenderWorldModel(Protocol):
    """Public Protocol Interface for Defender-Oriented Action-Conditioned World Model."""

    def encode_history(
        self,
        observations: torch.Tensor,          # [B, T_hist, 52]
    ) -> LatentContext:
        ...

    def predict_future(
        self,
        context: LatentContext,
        actions: torch.Tensor,               # [B, k]
        target: TargetSpec,
        horizon: int,
    ) -> LatentPrediction:
        ...
