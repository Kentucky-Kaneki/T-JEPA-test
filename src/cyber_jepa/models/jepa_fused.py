"""
Fused Cyber-JEPA Model - Phase 4, Test 2.

Mirrors `models.jepa.CyberJEPA`'s training-loop contract (forward signature, EMA
target update, single-frame T=1 target encoding, LayerNorm+SmoothL1 JEPA loss)
exactly, so every downstream evaluation tool built for Test 1 -
`metrics.compute_action_degradation`, `probes.LinearProbeEvaluator`,
`diagnostics.*`, `selection.apply_preregistered_selection_rule` - works unchanged on
Test 2 runs. The only architectural difference is that there is no separate
ActionEncoder/predictor: `online_encoder` is a fused encoder
(`FlatFusedRepresentation` / `FeatureFusedRepresentation`) that accepts the action
sequence directly in `encode_context`, and `readout` (`FusedLatentReadout`) is a thin
head instead of a full predictor.

The EMA target encoder is a deep copy of the SAME fused encoder class, but is always
called with `actions=None` (context-only pass) - it only ever encodes the
single-frame next observation `O_{t+1}`, exactly as in Test 1, so target-branch
semantics are held fixed and are not part of this ablation.
"""

import copy
from typing import cast
import torch
import torch.nn as nn

from cyber_jepa.models.context import ContextTokens
from cyber_jepa.models.predictor import TargetSpec, compute_jepa_loss
from cyber_jepa.models.fused_readout import FusedLatentReadout


def count_fused_subsystem_parameters(model: "CyberJEPAFused") -> dict[str, int]:
    """Parameter breakdown for a fused model, in categories comparable to
    `models.jepa.count_subsystem_parameters` (see `capacity_matching.py` for how the
    two are reconciled for budget matching)."""
    fused_encoder_params = sum(p.numel() for p in model.online_encoder.parameters() if p.requires_grad)
    readout_params = sum(p.numel() for p in model.readout.parameters() if p.requires_grad)
    target_params = sum(p.numel() for p in model.target_encoder.parameters())
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "fused_tokenizer_encoder_and_action_pathway": fused_encoder_params,
        "readout_head": readout_params,
        "target_encoder_ema": target_params,
        "total_trainable": total_trainable,
        "total_non_trainable_ema": target_params,
    }


class CyberJEPAFused(nn.Module):
    """Action-fused (Test 2) Cyber-JEPA: no separate ActionEncoder."""

    def __init__(
        self,
        online_encoder: nn.Module,
        hidden_dim: int = 64,
        max_horizon: int = 16,
        ema_momentum_init: float = 0.996,
        ema_momentum_final: float = 1.000,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ema_momentum_init = ema_momentum_init
        self.ema_momentum_final = ema_momentum_final

        self.online_encoder = online_encoder

        self.target_encoder = copy.deepcopy(online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.target_encoder.eval()

        self.readout = FusedLatentReadout(hidden_dim=hidden_dim, max_horizon=max_horizon)

    def _encode_target_single_frame(self, target_obs: torch.Tensor) -> torch.Tensor:
        """Mirrors `models.jepa.CyberJEPA._encode_target_single_frame` exactly
        (context-only, no actions, T=1) - duplicated rather than imported so Test 1's
        frozen module is never touched; keep the two in sync if that method changes."""
        target_single = target_obs.unsqueeze(1) if target_obs.dim() == 2 else target_obs
        self.target_encoder.eval()
        target_ctx = self.target_encoder.encode_context(target_single, actions=None, return_context_tokens=True)
        if isinstance(target_ctx, ContextTokens):
            if target_ctx.global_token is not None:
                return target_ctx.global_token
            return target_ctx.tokens.mean(dim=1)
        return cast(torch.Tensor, target_ctx)

    def forward(
        self,
        history_obs: torch.Tensor,          # [B, T_hist, 52]
        action_seq: torch.Tensor,           # [B, K]
        target_obs: torch.Tensor,           # [B, 52] single-frame target
        target_spec: TargetSpec | None = None,
        host_known_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Identical (history_obs, action_seq, target_obs, ...) -> (loss, pred, target)
        contract as `CyberJEPA.forward`, so `metrics.compute_action_degradation` and
        the training loop work unmodified on Test 2 models."""
        context_out = self.online_encoder.encode_context(
            history_obs, actions=action_seq, host_known_mask=host_known_mask, return_context_tokens=True,
        )
        pred_latents = self.readout(context_out, target_spec=target_spec)
        if pred_latents.dim() == 3 and target_spec is None:
            pred_latent = pred_latents[:, -1, :]
        else:
            pred_latent = pred_latents

        with torch.no_grad():
            target_latent = self._encode_target_single_frame(target_obs).detach()

        loss = compute_jepa_loss(pred_latent, target_latent)
        return loss, pred_latent, target_latent

    @torch.no_grad()
    def update_target_encoder(self, step: int, total_steps: int) -> float:
        """Identical EMA schedule/formula to `CyberJEPA.update_target_encoder`:
        theta_bar <- m * theta_bar + (1-m) * theta."""
        if total_steps <= 1:
            m = self.ema_momentum_init
        else:
            m = self.ema_momentum_init + (self.ema_momentum_final - self.ema_momentum_init) * (step / total_steps)
        m = min(1.0, max(0.0, m))

        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(m).add_(param_q.data, alpha=1.0 - m)
        for buffer_q, buffer_k in zip(self.online_encoder.buffers(), self.target_encoder.buffers()):
            buffer_k.data.copy_(buffer_q.data)
        return m
