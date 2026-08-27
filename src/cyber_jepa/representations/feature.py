"""
Feature Token Representation for Cyber-JEPA (Phase 3 baseline).

RECONSTRUCTION NOTE: `representations/feature.py` was not among the uploaded Phase 3
code files (only `flat.py` / `flat_ablations.py` were provided), but the Phase 4 fused
variant (`feature_fused.py`) and the Test 1 vs Test 2 comparison both need it as the
reference baseline. This module is reconstructed from the spec in
`evaluation_procedure.md` SS2.1 ("Feature Token Encoder") and SS4 ("Feature Token
Predictor Flow"): each of the 52 raw features gets its own token carrying a projected
value, feature index embedding, host identity embedding, feature type embedding, and
relative time embedding; history is flattened to T_hist * 52 tokens plus one
regularization token.

ASSUMPTION (flag for verification against the real Phase 3 file, if one exists): the
52 ChallengeWrapper features are treated as 13 hosts x 4 features/host (Scenario1b's
Blue table wrapper encodes each host as a short block - most commonly Activity +
Compromised level - of 4 values). `feature_idx // 4` gives the host slot (0-indexed)
and `feature_idx % 4` gives the intra-host feature slot. If the real Phase 3 mapping
differs, adjust `HOST_BLOCK_SIZE` / `_feature_to_host_and_slot` below; `feature_fused.py`
imports both and will pick up the change automatically.

Host indices follow the same 1..13 (0=NONE) convention as
`cyber_jepa.models.predictor.ActionEncoder`, so that `feature_fused.py` can share a
single host embedding table between state and action tokens.
"""

from typing import cast
import torch
import torch.nn as nn

from cyber_jepa.models.context import ContextTokens, TokenType

HOST_BLOCK_SIZE = 4     # features per host in the flat 52-dim vector (13 hosts * 4 = 52)
NUM_HOSTS_0IDX = 13


def _feature_to_host_and_slot(obs_dim: int = 52, host_block_size: int = HOST_BLOCK_SIZE) -> tuple[torch.Tensor, torch.Tensor]:
    """Map each of the `obs_dim` feature indices to (host_id in 1..13, intra-host feature slot)."""
    feature_idx = torch.arange(obs_dim, dtype=torch.long)
    host_slot_0idx = feature_idx // host_block_size          # 0..12
    feature_slot = feature_idx % host_block_size              # 0..3
    host_id_1idx = host_slot_0idx + 1                          # 1..13 (matches ActionEncoder host convention)
    return host_id_1idx, feature_slot


class FeatureTokenRepresentation(nn.Module):
    """Per-feature tokenized observation encoder (structured alternative to flat.py)."""

    host_id_lut: torch.Tensor
    feature_slot_lut: torch.Tensor

    def __init__(
        self,
        obs_dim: int = 52,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        ffn_dim: int = 256,
        history_len: int = 4,
        host_block_size: int = HOST_BLOCK_SIZE,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.history_len = history_len
        self.host_block_size = host_block_size

        self.value_proj = nn.Linear(1, hidden_dim)
        self.feature_index_emb = nn.Embedding(obs_dim, hidden_dim)
        self.host_index_emb = nn.Embedding(NUM_HOSTS_0IDX + 1, hidden_dim)   # 0=NONE (unused for state), 1..13
        self.feature_type_emb = nn.Embedding(host_block_size, hidden_dim)
        self.time_emb = nn.Embedding(history_len, hidden_dim)
        self.reg_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        host_id_lut, feature_slot_lut = _feature_to_host_and_slot(obs_dim, host_block_size)
        self.register_buffer("host_id_lut", host_id_lut)
        self.register_buffer("feature_slot_lut", feature_slot_lut)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def encode_context(
        self,
        x: torch.Tensor,                       # [B, T_hist, obs_dim]
        host_known_mask: torch.Tensor | None = None,
        return_context_tokens: bool = True,
    ) -> ContextTokens | torch.Tensor:
        B, T_hist, F = x.shape
        device = x.device
        assert F == self.obs_dim, f"expected {self.obs_dim} features, got {F}"

        values = x.reshape(B, T_hist * F, 1)                       # [B, T_hist*F, 1]
        val_tok = self.value_proj(values)                          # [B, T_hist*F, D]

        feature_idx = torch.arange(F, device=device)
        host_ids = self.host_id_lut.to(device)[feature_idx]        # [F]
        feat_slots = self.feature_slot_lut.to(device)[feature_idx] # [F]

        feat_idx_e = self.feature_index_emb(feature_idx).view(1, 1, F, -1).expand(B, T_hist, F, -1)
        host_e = self.host_index_emb(host_ids).view(1, 1, F, -1).expand(B, T_hist, F, -1)
        type_e = self.feature_type_emb(feat_slots).view(1, 1, F, -1).expand(B, T_hist, F, -1)

        time_ids_per_t = torch.arange(T_hist, device=device)
        time_e = self.time_emb(time_ids_per_t).view(1, T_hist, 1, -1).expand(B, T_hist, F, -1)

        tok = val_tok.view(B, T_hist, F, -1) + feat_idx_e + host_e + type_e + time_e
        tok = tok.reshape(B, T_hist * F, -1)                                    # [B, L, D], L = T_hist*F

        # Exactness audit (mirrors evaluation_procedure.md SS2.1/SS6): 52 features must
        # produce exactly 52 tokens per timestep, T_hist*52 tokens total.
        assert tok.shape[1] == T_hist * self.obs_dim, "feature tokenization exactness check failed"

        reg = self.reg_token.expand(B, -1, -1)
        seq = torch.cat([tok, reg], dim=1)
        out = self.transformer(seq)
        out = self.norm(out)

        global_token = out[:, -1, :]
        feat_out = out[:, :-1, :]

        if not return_context_tokens:
            return cast(torch.Tensor, global_token)

        L = T_hist * F
        entity_ids = host_ids.unsqueeze(0).expand(B, T_hist, -1).reshape(B, L)
        time_ids = time_ids_per_t.view(1, T_hist, 1).expand(B, T_hist, F).reshape(B, L)
        token_type_ids = torch.full((B, L), TokenType.FEATURE, device=device, dtype=torch.long)
        padding_mask = torch.zeros((B, L), device=device, dtype=torch.bool)

        ctx = ContextTokens(
            tokens=feat_out,
            padding_mask=padding_mask,
            visibility_mask=None,
            time_ids=time_ids,
            entity_ids=entity_ids,
            token_type_ids=token_type_ids,
            global_token=global_token,
            metadata={"history_len": T_hist, "obs_dim": F, "host_block_size": self.host_block_size},
        )
        ctx.validate()
        return ctx

    def forward(self, x: torch.Tensor, host_known_mask: torch.Tensor | None = None) -> ContextTokens | torch.Tensor:
        return self.encode_context(x, host_known_mask=host_known_mask, return_context_tokens=True)
