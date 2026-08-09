"""
Action Encoder and Latent Predictor for Cyber-JEPA.

ActionEncoder embeds defensive action sequences a_{t:t+k-1} across future offsets.
LatentPredictor predicts target latent representations given context tokens, action sequence,
and target query specification using Layer-Normalized Smooth L1 loss.
"""

from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionEncoder(nn.Module):
    """Embeds defensive actions (type, target host, target subnet, future offset)."""

    def __init__(
        self,
        num_action_types: int = 16,
        num_hosts: int = 13,
        num_subnets: int = 3,
        hidden_dim: int = 64,
        max_horizon: int = 16,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.action_type_emb = nn.Embedding(num_action_types, hidden_dim)
        self.target_host_emb = nn.Embedding(num_hosts + 1, hidden_dim)   # +1 for NONE
        self.target_subnet_emb = nn.Embedding(num_subnets + 1, hidden_dim) # +1 for NONE
        self.offset_emb = nn.Embedding(max_horizon, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(
        self,
        action_indices: torch.Tensor,       # [B, k] discrete action indices
        action_types: torch.Tensor | None = None, # [B, k] optional discrete action type IDs
        target_hosts: torch.Tensor | None = None, # [B, k] optional host IDs
        target_subnets: torch.Tensor | None = None,# [B, k] optional subnet IDs
    ) -> torch.Tensor:
        """
        Input: action_indices [B, k]
        Output: action tokens [B, k, hidden_dim]
        """
        B, K = action_indices.shape
        device = action_indices.device

        # Default fallback embeddings if specific field IDs not passed
        act_type_ids = action_types if action_types is not None else (action_indices % 16)
        host_ids = target_hosts if target_hosts is not None else torch.zeros((B, K), device=device, dtype=torch.long)
        sub_ids = target_subnets if target_subnets is not None else torch.zeros((B, K), device=device, dtype=torch.long)
        offset_ids = torch.arange(K, device=device).unsqueeze(0).expand(B, -1) # [B, k]

        e_act = self.action_type_emb(act_type_ids)
        e_host = self.target_host_emb(host_ids)
        e_sub = self.target_subnet_emb(sub_ids)
        e_off = self.offset_emb(offset_ids)

        cat_emb = torch.cat([e_act, e_host, e_sub, e_off], dim=-1) # [B, k, 4 * D]
        act_tokens = self.mlp(cat_emb)                             # [B, k, D]
        return act_tokens


class LatentPredictor(nn.Module):
    """Predicts target latent z_{t+k} from context tokens and future action sequence."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 256,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Target query token
        self.target_query_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Predictor Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.pred_head = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        context_tokens: torch.Tensor,       # [B, N_ctx, hidden_dim] or [B, hidden_dim]
        action_tokens: torch.Tensor,        # [B, k, hidden_dim]
    ) -> torch.Tensor:
        """
        Predicts future latent representation.
        Returns: predicted latent z_hat [B, hidden_dim]
        """
        B = action_tokens.shape[0]

        if context_tokens.dim() == 2:
            ctx = context_tokens.unsqueeze(1) # [B, 1, D]
        else:
            ctx = context_tokens             # [B, N_ctx, D]

        # Target query token
        q_token = self.target_query_token.expand(B, -1, -1) # [B, 1, D]

        # Combine context, action sequence, and query token
        seq = torch.cat([ctx, action_tokens, q_token], dim=1) # [B, N_ctx + k + 1, D]

        out = self.transformer(seq)
        out = self.norm(out)

        # Prediction from query token position
        pred = self.pred_head(out[:, -1, :]) # [B, D]
        return pred


def compute_jepa_loss(pred_latent: torch.Tensor, target_latent: torch.Tensor) -> torch.Tensor:
    """
    Computes Layer-Normalized Smooth L1 JEPA Loss:
    L = SmoothL1(LN(pred), stopgrad(LN(target)))
    """
    ln_pred = F.layer_norm(pred_latent, (pred_latent.shape[-1],))
    ln_target = F.layer_norm(target_latent.detach(), (target_latent.shape[-1],))
    return F.smooth_l1_loss(ln_pred, ln_target)
