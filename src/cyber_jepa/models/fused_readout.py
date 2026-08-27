"""
Fused Latent Readout Head - Phase 4, Test 2.

Replaces the bulk of `models.predictor.LatentPredictor` for the fused (Test 2)
variants: since context+action fusion now happens inside the encoder
(`flat_fused.py` / `feature_fused.py`), all that is left downstream is a thin
per-horizon readout over the encoder's own action-token outputs, plus the same
TargetSpec granularity/entity/horizon combination `LatentPredictor` used, kept for
parity so Test 1 vs Test 2 predictions are read out the same way.
"""

from typing import cast
import torch
import torch.nn as nn

from cyber_jepa.models.context import ContextTokens
from cyber_jepa.models.predictor import TargetSpec


class FusedLatentReadout(nn.Module):
    def __init__(self, hidden_dim: int = 64, max_horizon: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_horizon = max_horizon
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, hidden_dim)
        # Same granularity/entity embedding scheme as LatentPredictor, for parity.
        self.granularity_emb = nn.Embedding(4, hidden_dim)  # 0=feature,1=host,2=subnet,3=network
        self.entity_emb = nn.Embedding(16, hidden_dim)

    def forward(self, context: ContextTokens, target_spec: TargetSpec | None = None) -> torch.Tensor:
        action_start = context.metadata.get("action_start_idx")
        if action_start is None:
            raise ValueError("FusedLatentReadout requires ContextTokens produced with actions (no action tokens found)")

        action_out = context.tokens[:, action_start:, :]     # [B, K, D]
        K = action_out.shape[1]
        if K == 0:
            raise ValueError("FusedLatentReadout received zero action tokens - was `actions` passed to encode_context?")

        pred_latents = self.head(self.norm(action_out))      # [B, K, D]

        if target_spec is None:
            return pred_latents

        k_idx = min(max(target_spec.horizon, 1), K) - 1
        pred_k = pred_latents[:, k_idx, :]

        device = pred_k.device
        gran_map = {"feature": 0, "host": 1, "subnet": 2, "network": 3}
        g_id = gran_map.get(target_spec.granularity.lower(), 3)
        g_emb = self.granularity_emb(torch.tensor(g_id, device=device))
        e_emb = self.entity_emb(torch.tensor(min(target_spec.target_id, 15), device=device))
        return cast(torch.Tensor, pred_k + g_emb + e_emb)
