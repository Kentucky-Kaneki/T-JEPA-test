"""
Cyber-JEPA Model Core.

Maintains online context encoder f_theta, complete EMA target encoder f_bar_theta,
ActionEncoder e_a, LatentPredictor g_phi, and explicit ContextAggregator interventions
with single-frame (T=1) target encoding and parameter accounting.
"""

import copy
from typing import Any, cast
import torch
import torch.nn as nn

from cyber_jepa.models.context import ContextTokens
from cyber_jepa.models.aggregators import (
    ContextAggregator,
    LegacyLastStepMean,
    LearnedQueryPool,
    TokenPreservingAggregator,
)
from cyber_jepa.models.predictor import ActionEncoder, LatentPredictor, TargetSpec, compute_jepa_loss


def count_subsystem_parameters(model: Any) -> dict[str, int]:
    """Report detailed parameter breakdown across model subsystems."""
    counts: dict[str, int] = {}
    online_enc: nn.Module | None = getattr(model, "online_encoder", None)
    if online_enc is not None:
        counts["tokenizer_and_encoder"] = sum(p.numel() for p in online_enc.parameters() if p.requires_grad)

    agg: nn.Module | None = getattr(model, "aggregator", None)
    if agg is not None:
        counts["aggregator"] = sum(p.numel() for p in agg.parameters() if p.requires_grad)

    pred: LatentPredictor | None = getattr(model, "predictor", None)
    if pred is not None:
        counts["action_encoder"] = sum(p.numel() for p in pred.action_encoder.parameters() if p.requires_grad)
        counts["predictor_body"] = sum(p.numel() for p in pred.parameters() if p.requires_grad) - counts["action_encoder"]

    target_enc: nn.Module | None = getattr(model, "target_encoder", None)
    if target_enc is not None:
        counts["target_encoder_ema"] = sum(p.numel() for p in target_enc.parameters())

    model_module: nn.Module = cast(nn.Module, model)
    counts["total_trainable"] = sum(p.numel() for p in model_module.parameters() if p.requires_grad)
    if target_enc is not None:
        counts["total_non_trainable_ema"] = sum(p.numel() for p in target_enc.parameters())
    return counts


class CyberJEPA(nn.Module):
    """Action-Conditioned Joint-Embedding Predictive Architecture for CybORG."""

    def __init__(
        self,
        online_encoder: nn.Module,
        hidden_dim: int = 64,
        max_horizon: int = 16,
        aggregator_mode: str = "legacy_last_step_mean",
        ema_momentum_init: float = 0.996,
        ema_momentum_final: float = 1.000,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.aggregator_mode = aggregator_mode
        self.ema_momentum_init = ema_momentum_init
        self.ema_momentum_final = ema_momentum_final

        # 1. Online context encoder f_theta
        self.online_encoder = online_encoder

        # 2. Complete EMA target encoder f_bar_theta (copied & frozen)
        self.target_encoder = copy.deepcopy(online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.target_encoder.eval()

        # 3. Explicit Context Aggregator intervention
        self.aggregator_module: nn.Module
        if aggregator_mode == "legacy_last_step_mean":
            self.aggregator_module = LegacyLastStepMean(hidden_dim=hidden_dim)
        elif aggregator_mode == "learned_query_pool":
            self.aggregator_module = LearnedQueryPool(hidden_dim=hidden_dim)
        elif aggregator_mode == "token_preserving_predictor":
            self.aggregator_module = TokenPreservingAggregator(hidden_dim=hidden_dim)
        else:
            raise ValueError(f"Unknown aggregator_mode: {aggregator_mode}")

        self.aggregator = self.aggregator_module

        # 4. Latent predictor g_phi with action encoder
        self.predictor = LatentPredictor(hidden_dim=hidden_dim, max_horizon=max_horizon)

    @property
    def action_encoder(self) -> ActionEncoder:
        return self.predictor.action_encoder

    def _encode_target_single_frame(
        self,
        target_obs: torch.Tensor, # [B, 52] or [B, 1, 52]
    ) -> torch.Tensor:
        """
        Encode single-frame target observation (T=1) without artificial 4x repetition.
        Returns target latent [B, hidden_dim].
        """
        if target_obs.dim() == 2:
            target_single = target_obs.unsqueeze(1) # [B, 1, 52]
        else:
            target_single = target_obs

        self.target_encoder.eval()
        enc_fn = getattr(self.target_encoder, "encode_context", None)
        if callable(enc_fn):
            target_ctx = enc_fn(target_single, return_context_tokens=True)
            if isinstance(target_ctx, ContextTokens):
                if target_ctx.global_token is not None:
                    return target_ctx.global_token
                else:
                    return target_ctx.tokens.mean(dim=1)
            else:
                return cast(torch.Tensor, target_ctx)
        else:
            target_out = self.target_encoder(target_single)
            if hasattr(target_out, "global_token"):
                return cast(torch.Tensor, getattr(target_out, "global_token"))
            elif target_out.dim() == 4:
                return cast(torch.Tensor, target_out[:, -1, :, :].mean(dim=1))
            elif target_out.dim() == 3:
                return cast(torch.Tensor, target_out.mean(dim=1))
            else:
                return cast(torch.Tensor, target_out)

    def forward(
        self,
        history_obs: torch.Tensor,         # [B, T_hist, 52]
        action_seq: torch.Tensor,          # [B, k]
        target_obs: torch.Tensor,          # [B, 52] single-frame target
        target_spec: TargetSpec | None = None,
        host_known_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass computing online predicted latent and single-frame target latent.

        Returns:
            loss: Layer-Normalized Smooth L1 loss scalar
            pred_latent: [B, hidden_dim]
            target_latent: [B, hidden_dim] (stop-gradient)
        """
        B = history_obs.shape[0]

        # 1. Encode context via online encoder
        enc_fn = getattr(self.online_encoder, "encode_context", None)
        if callable(enc_fn):
            context_out = enc_fn(history_obs, host_known_mask=host_known_mask)
        else:
            context_raw = self.online_encoder(history_obs)
            if isinstance(context_raw, ContextTokens):
                context_out = context_raw
            else:
                dummy_tokens = context_raw if context_raw.dim() == 3 else context_raw.unsqueeze(1)
                context_out = ContextTokens(
                    tokens=dummy_tokens,
                    padding_mask=torch.zeros((B, dummy_tokens.shape[1]), device=history_obs.device, dtype=torch.bool),
                    visibility_mask=None,
                    time_ids=torch.zeros((B, dummy_tokens.shape[1]), device=history_obs.device, dtype=torch.long),
                    entity_ids=torch.zeros((B, dummy_tokens.shape[1]), device=history_obs.device, dtype=torch.long),
                    token_type_ids=torch.zeros((B, dummy_tokens.shape[1]), device=history_obs.device, dtype=torch.long),
                    global_token=context_raw if context_raw.dim() == 2 else None,
                )

        # 2. Aggregate context
        agg_out = self.aggregator_module(context_out)

        # 3. Predict future representation
        if self.aggregator_mode == "token_preserving_predictor":
            pred_latent = self.predictor(
                z_t=context_out,
                actions=action_seq,
                target_spec=target_spec,
            )
        else:
            pred_latent = self.predictor(
                z_t=agg_out.latent,
                actions=action_seq,
                target_spec=target_spec,
            )

        if pred_latent.dim() == 3 and target_spec is None:
            pred_latent = pred_latent[:, -1, :]

        # 4. Single-frame target representation via EMA target encoder (T=1, no grad)
        with torch.no_grad():
            target_latent = self._encode_target_single_frame(target_obs).detach()

        # 5. Layer-Normalized Smooth L1 JEPA Loss
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
