"""
Cyber-JEPA Model Core.

Maintains online context encoder f_theta, complete EMA target encoder f_bar_theta,
ActionEncoder e_a, and LatentPredictor g_phi with EMA momentum scheduling.
Handles HierarchicalOutput explicitly.
"""

import copy
from typing import Any
import torch
import torch.nn as nn

from cyber_jepa.models.predictor import ActionEncoder, LatentPredictor, TargetSpec
from cyber_jepa.representations.hierarchical import HierarchicalOutput


def compute_jepa_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Layer-Normalized Smooth L1 JEPA Loss."""
    norm_pred = nn.functional.layer_norm(pred, pred.shape[-1:])
    norm_target = nn.functional.layer_norm(target, target.shape[-1:])
    return nn.functional.smooth_l1_loss(norm_pred, norm_target)


class CyberJEPA(nn.Module):
    """Action-Conditioned Joint-Embedding Predictive Architecture for CybORG."""

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

        # 1. Online context encoder f_theta
        self.online_encoder = online_encoder

        # 2. Complete EMA target encoder f_bar_theta (copied & frozen)
        self.target_encoder = copy.deepcopy(online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.target_encoder.eval()

        # 3. Action encoder e_a
        self.action_encoder = ActionEncoder(hidden_dim=hidden_dim, max_horizon=max_horizon)

        # 4. Latent predictor g_phi
        self.predictor = LatentPredictor(hidden_dim=hidden_dim, max_horizon=max_horizon)

    def forward(
        self,
        history_obs: torch.Tensor,         # [B, T_hist, 52]
        action_seq: torch.Tensor,          # [B, k]
        target_obs: torch.Tensor,          # [B, 52] or target representation
        target_spec: TargetSpec | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass computing online predicted latent and target latent.

        Returns:
            loss: Layer-Normalized Smooth L1 loss scalar
            pred_latent: [B, hidden_dim]
            target_latent: [B, hidden_dim] (stop-gradient)
        """
        B = history_obs.shape[0]

        # 1. Encode context via online encoder
        context_out = self.online_encoder(history_obs)
        if isinstance(context_out, HierarchicalOutput):
            context_latents = context_out.global_token # [B, hidden_dim]
        elif context_out.dim() == 4:
            context_latents = context_out[:, -1, :, :].mean(dim=1)
        elif context_out.dim() == 3:
            context_latents = context_out[:, -1, :]
        else:
            context_latents = context_out

        # 2. Predict future representation
        pred_latent = self.predictor(context_latents, action_seq, target_spec=target_spec) # [B, hidden_dim] or [B, K, hidden_dim]
        if pred_latent.dim() == 3 and target_spec is None:
            pred_latent = pred_latent[:, -1, :] # Default to final horizon step

        # 3. Target representation via EMA target encoder (deterministic, no grad)
        with torch.no_grad():
            self.target_encoder.eval()
            if target_obs.dim() == 2:
                hist_len = getattr(self.online_encoder, "history_len", 4)
                target_in = target_obs.unsqueeze(1).expand(-1, hist_len, -1)
            else:
                target_in = target_obs

            target_out = self.target_encoder(target_in)
            if isinstance(target_out, HierarchicalOutput):
                target_latents = target_out.global_token
            elif target_out.dim() == 4:
                target_latents = target_out[:, -1, :, :].mean(dim=1)
            elif target_out.dim() == 3:
                target_latents = target_out[:, -1, :]
            else:
                target_latents = target_out

            target_latent = target_latents.detach()

        # 4. Layer-Normalized Smooth L1 JEPA Loss
        loss = compute_jepa_loss(pred_latent, target_latent)
        return loss, pred_latent, target_latent

    @torch.no_grad()
    def update_target_encoder(self, step: int, total_steps: int) -> float:
        """EMA update of target encoder parameters: theta_bar <- m * theta_bar + (1-m) * theta."""
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
