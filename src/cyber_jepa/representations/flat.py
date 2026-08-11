"""
Flat Temporal Vector Representation for Cyber-JEPA.

Encodes 52-dim flat vector observations across history timesteps into network-level latents
using per-timestep MLP projection, relative time embeddings, and temporal Transformer attention.
"""

from typing import Any, cast
import torch
import torch.nn as nn

from cyber_jepa.models.context import ContextTokens, TokenType


class FlatVectorRepresentation(nn.Module):
    """Flat temporal vector observation tokenizer and encoder."""

    def __init__(
        self,
        obs_dim: int = 52,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        ffn_dim: int = 256,
        history_len: int = 4,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.history_len = history_len

        # Per-timestep MLP projection
        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim, ffn_dim),
            nn.LayerNorm(ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
        )

        # Relative time embeddings (-3, -2, -1, 0)
        self.time_emb = nn.Embedding(history_len, hidden_dim)

        # One functional regularization token
        self.reg_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Pre-LN Transformer Encoder
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

    def encode_context(
        self,
        x: torch.Tensor,
        host_known_mask: torch.Tensor | None = None,
        return_context_tokens: bool = True,
    ) -> ContextTokens | torch.Tensor:
        """
        Input: x of shape [B, T_hist, 52]
        Returns: ContextTokens dataclass containing tokens [B, T_hist, D] and global_token [B, D]
        """
        B, T_hist, D = x.shape
        device = x.device

        # Project timesteps to hidden_dim
        tokens = self.input_proj(x) # [B, T_hist, hidden_dim]

        # Add relative time embeddings
        time_ids = torch.arange(T_hist, device=device).unsqueeze(0).expand(B, -1)
        tokens = tokens + self.time_emb(time_ids)

        # Append functional regularization token
        reg_tokens = self.reg_token.expand(B, -1, -1) # [B, 1, hidden_dim]
        seq_tokens = torch.cat([tokens, reg_tokens], dim=1) # [B, T_hist + 1, hidden_dim]

        # Temporal Transformer
        out = self.transformer(seq_tokens)
        out = self.norm(out)

        global_latent = out[:, -1, :] # [B, hidden_dim]
        flat_tokens = out[:, :-1, :]  # [B, T_hist, hidden_dim]

        if not return_context_tokens:
            return cast(torch.Tensor, global_latent)

        padding_mask = torch.zeros((B, T_hist), device=device, dtype=torch.bool)
        entity_ids = torch.zeros((B, T_hist), device=device, dtype=torch.long)
        token_type_ids = torch.full((B, T_hist), TokenType.TEMPORAL_FLAT, device=device, dtype=torch.long)

        ctx = ContextTokens(
            tokens=flat_tokens,
            padding_mask=padding_mask,
            visibility_mask=None,
            time_ids=time_ids,
            entity_ids=entity_ids,
            token_type_ids=token_type_ids,
            global_token=global_latent,
            metadata={"history_len": T_hist, "obs_dim": D},
        )
        ctx.validate()
        return ctx

    def forward(
        self,
        x: torch.Tensor,
        host_known_mask: torch.Tensor | None = None,
    ) -> ContextTokens | torch.Tensor:
        return self.encode_context(x, host_known_mask=host_known_mask, return_context_tokens=True)
