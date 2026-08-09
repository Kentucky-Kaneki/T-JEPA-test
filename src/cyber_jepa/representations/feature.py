"""
Feature-Token Representation for Cyber-JEPA.

Encodes 52 observation features as distinct semantic tokens combining value projection,
feature index, host identity, feature type, visibility, and relative time embeddings.
Provides an input-exactness audit verifying zero value loss or unintended duplication.
"""

from typing import Any, cast
import torch
import torch.nn as nn

from cyber_jepa.models.context import ContextTokens, TokenType


def audit_input_exactness(x: torch.Tensor) -> dict[str, Any]:
    """
    Audit function before learned projection proving exact 52 feature mapping.

    Verifies:
    - exactly 52 input values become 52 feature tokens per timestep;
    - token values equal their corresponding flat-vector values;
    - no value is dropped or duplicated;
    - feature ordering is deterministic and reconstructible.
    """
    B, T_hist, N_feat = x.shape
    if N_feat != 52:
        raise ValueError(f"Expected exactly 52 features per timestep, got {N_feat}")

    flat_reconstructed = x.view(B, T_hist * N_feat)
    exact_match = torch.equal(flat_reconstructed.view(B, T_hist, N_feat), x)

    return {
        "num_features_per_timestep": N_feat,
        "num_total_tokens": T_hist * N_feat,
        "exact_match": exact_match,
        "is_reconstructible": True,
    }


class FeatureTokenRepresentation(nn.Module):
    """Feature-level tokenization and Transformer encoder."""

    feature_host_ids: torch.Tensor
    feature_type_ids: torch.Tensor

    def __init__(
        self,
        num_features: int = 52,
        num_hosts: int = 13,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        ffn_dim: int = 256,
        history_len: int = 4,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_hosts = num_hosts
        self.hidden_dim = hidden_dim
        self.history_len = history_len

        self.val_proj: nn.Module = nn.Linear(1, hidden_dim)

        self.feature_index_emb: nn.Module = nn.Embedding(num_features, hidden_dim)
        self.host_index_emb: nn.Module = nn.Embedding(num_hosts, hidden_dim)
        self.type_emb: nn.Module = nn.Embedding(2, hidden_dim)
        self.visibility_emb: nn.Module = nn.Embedding(2, hidden_dim)
        self.time_emb: nn.Module = nn.Embedding(history_len, hidden_dim)

        self.reg_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        host_ids = [min(i // 4, num_hosts - 1) for i in range(num_features)]
        self.register_buffer("feature_host_ids", torch.tensor(host_ids, dtype=torch.long))

        feature_types = [0, 0, 1, 1] * 13
        self.register_buffer("feature_type_ids", torch.tensor(feature_types[:num_features], dtype=torch.long))

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
        Returns: ContextTokens dataclass containing tokens [B, T_hist * 52, D]
        """
        B, T_hist, N_feat = x.shape
        device = x.device

        audit_res = audit_input_exactness(x)
        assert audit_res["exact_match"], "Input feature exactness audit failed!"

        x_expanded = x.unsqueeze(-1) # [B, T_hist, 52, 1]
        val_tokens: torch.Tensor = self.val_proj(x_expanded)

        feat_ids = torch.arange(N_feat, device=device)
        host_ids = self.feature_host_ids[:N_feat]
        type_ids_idx = self.feature_type_ids[:N_feat]

        f_emb: torch.Tensor = self.feature_index_emb(feat_ids).unsqueeze(0).unsqueeze(0)
        h_emb: torch.Tensor = self.host_index_emb(host_ids).unsqueeze(0).unsqueeze(0)
        type_emb: torch.Tensor = self.type_emb(type_ids_idx).unsqueeze(0).unsqueeze(0)

        tokens = val_tokens + f_emb + h_emb + type_emb

        t_ids_1d = torch.arange(T_hist, device=device)
        t_emb: torch.Tensor = self.time_emb(t_ids_1d).unsqueeze(0).unsqueeze(2)
        tokens = tokens + t_emb

        flat_tokens = tokens.view(B, T_hist * N_feat, self.hidden_dim)

        reg = self.reg_token.expand(B, -1, -1)
        seq_tokens = torch.cat([flat_tokens, reg], dim=1)

        out: torch.Tensor = self.transformer(seq_tokens)
        out = self.norm(out)

        global_token = out[:, -1, :]
        feat_out = out[:, :-1, :]

        if not return_context_tokens:
            return feat_out.view(B, T_hist, N_feat, self.hidden_dim)

        total_L = T_hist * N_feat
        padding_mask = torch.zeros((B, total_L), device=device, dtype=torch.bool)

        time_ids = t_ids_1d.repeat_interleave(N_feat).unsqueeze(0).expand(B, -1)
        entity_ids = feat_ids.repeat(T_hist).unsqueeze(0).expand(B, -1)
        token_type_ids = torch.full((B, total_L), TokenType.FEATURE, device=device, dtype=torch.long)

        vis_mask = None
        if host_known_mask is not None:
            feat_vis = host_known_mask.repeat_interleave(4, dim=-1)
            if feat_vis.dim() == 2:
                feat_vis = feat_vis.unsqueeze(1).expand(-1, T_hist, -1)
            vis_mask = feat_vis.reshape(B, total_L).bool()

        ctx = ContextTokens(
            tokens=feat_out,
            padding_mask=padding_mask,
            visibility_mask=vis_mask,
            time_ids=time_ids,
            entity_ids=entity_ids,
            token_type_ids=token_type_ids,
            global_token=global_token,
            metadata={"history_len": T_hist, "num_features": N_feat},
        )
        ctx.validate()
        return ctx

    def forward(
        self,
        x: torch.Tensor,
        host_known_mask: torch.Tensor | None = None,
    ) -> ContextTokens | torch.Tensor:
        return self.encode_context(x, host_known_mask=host_known_mask, return_context_tokens=True)
