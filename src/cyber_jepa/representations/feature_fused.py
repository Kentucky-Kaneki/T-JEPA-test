"""
Feature Fused Representation - Phase 4, Test 2, Variant 2b.

Early-fusion counterpart to `representations/feature.py`: state tokens (52 features x
T_hist, per `feature.py`'s tokenization) and action tokens share ONE Transformer stack
and, critically, share the SAME host embedding table - `feature.py`'s per-feature host
identity embedding and the action's target-host embedding are the same
`nn.Embedding` instance here, unlike Test 1 where `ActionEncoder.host_emb` is a
separate, independently-trained table. This is the variant where shared-embedding
grounding has a natural mechanism to matter: an action token targeting Host 7 can
attend directly to Host 7's feature tokens through a shared representation space
within a single self-attention stack (permissive mode) or read them one-directionally
(strict mode - see `fusion_masking.py`).

See `fused_readout.py` for the downstream readout and `capacity_matching.py` for how
`num_layers`/`ffn_dim` are chosen to match Test 1's combined parameter budget.
"""

from typing import cast
import torch
import torch.nn as nn

from cyber_jepa.models.context import ContextTokens, TokenType
from cyber_jepa.models.action_semantics import build_scenario1b_action_lookup_buffers, NUM_ACTION_TYPES
from cyber_jepa.models.shared_semantic_embeddings import SharedEntityEmbeddings, UnifiedTimeEmbedding
from cyber_jepa.models.fusion_masking import build_fusion_attention_mask, STRICT
from cyber_jepa.representations.feature import _feature_to_host_and_slot, HOST_BLOCK_SIZE


class FeatureFusedRepresentation(nn.Module):
    type_map: torch.Tensor
    host_map: torch.Tensor
    subnet_map: torch.Tensor
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
        max_horizon: int = 16,
        host_block_size: int = HOST_BLOCK_SIZE,
        attention_mode: str = STRICT,
        shared_embeddings: SharedEntityEmbeddings | None = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.history_len = history_len
        self.max_horizon = max_horizon
        self.host_block_size = host_block_size
        self.attention_mode = attention_mode

        # --- state tokenization (mirrors feature.py) ---
        self.value_proj = nn.Linear(1, hidden_dim)
        self.feature_index_emb = nn.Embedding(obs_dim, hidden_dim)
        self.feature_type_emb = nn.Embedding(host_block_size, hidden_dim)
        self.reg_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        host_id_lut, feature_slot_lut = _feature_to_host_and_slot(obs_dim, host_block_size)
        self.register_buffer("host_id_lut", host_id_lut)
        self.register_buffer("feature_slot_lut", feature_slot_lut)

        # --- shared semantic fields: host embedding used by BOTH state and action tokens ---
        self.shared_embeddings = shared_embeddings or SharedEntityEmbeddings(hidden_dim=hidden_dim)
        self.time_axis = UnifiedTimeEmbedding(hidden_dim, history_len=history_len, max_horizon=max_horizon)

        # --- action-only semantic fields ---
        self.action_type_emb = nn.Embedding(NUM_ACTION_TYPES, hidden_dim)
        self.action_pos_emb = nn.Embedding(max_horizon, hidden_dim)
        self.action_valid_emb = nn.Embedding(2, hidden_dim)
        self.action_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        lut = build_scenario1b_action_lookup_buffers()
        self.register_buffer("type_map", lut["type_map"])
        self.register_buffer("host_map", lut["host_map"])
        self.register_buffer("subnet_map", lut["subnet_map"])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def _build_state_tokens(self, x: torch.Tensor) -> torch.Tensor:
        B, T_hist, F = x.shape
        device = x.device
        assert F == self.obs_dim, f"expected {self.obs_dim} features, got {F}"

        values = x.reshape(B, T_hist * F, 1)
        val_tok = self.value_proj(values)

        feature_idx = torch.arange(F, device=device)
        host_ids = self.host_id_lut.to(device)[feature_idx]
        feat_slots = self.feature_slot_lut.to(device)[feature_idx]

        feat_idx_e = self.feature_index_emb(feature_idx).view(1, 1, F, -1).expand(B, T_hist, F, -1)
        host_e = self.shared_embeddings.embed_host(host_ids).view(1, 1, F, -1).expand(B, T_hist, F, -1)
        type_e = self.feature_type_emb(feat_slots).view(1, 1, F, -1).expand(B, T_hist, F, -1)

        time_ids_per_t = self.time_axis.history_ids(B, device, length=T_hist)          # [B, T_hist]
        time_e = self.time_axis(time_ids_per_t).unsqueeze(2).expand(B, T_hist, F, -1)

        tok = val_tok.view(B, T_hist, F, -1) + feat_idx_e + host_e + type_e + time_e
        return tok.reshape(B, T_hist * F, -1)

    def _build_action_tokens(self, actions: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
        B, K = actions.shape
        device = actions.device
        act_idx = torch.clamp(actions, 0, self.type_map.shape[0] - 1)
        type_ids = self.type_map[act_idx]
        host_ids = self.host_map[act_idx]
        subnet_ids = self.subnet_map[act_idx]

        pos = torch.arange(K, device=device).unsqueeze(0).expand(B, K)
        valid_ids = torch.ones((B, K), device=device, dtype=torch.long) if valid_mask is None else valid_mask.to(torch.long)

        t_e = self.action_type_emb(type_ids)
        h_e = self.shared_embeddings.embed_host(host_ids)      # SAME table as state tokens
        s_e = self.shared_embeddings.embed_subnet(subnet_ids)
        p_e = self.action_pos_emb(pos)
        v_e = self.action_valid_emb(valid_ids)
        time_e = self.time_axis(self.time_axis.horizon_ids(B, K, device))

        token = t_e + h_e + s_e + p_e + v_e + time_e
        return cast(torch.Tensor, self.action_proj(token))

    def encode_context(
        self,
        x: torch.Tensor,
        actions: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        host_known_mask: torch.Tensor | None = None,
        return_context_tokens: bool = True,
    ) -> ContextTokens | torch.Tensor:
        B, T_hist, F = x.shape
        device = x.device

        state_tok = self._build_state_tokens(x)                    # [B, T_hist*F, D]
        reg_tok = self.reg_token.expand(B, -1, -1)
        state_seq = torch.cat([state_tok, reg_tok], dim=1)
        state_len = state_seq.shape[1]

        if actions is not None:
            action_tok = self._build_action_tokens(actions, valid_mask)
            action_len = action_tok.shape[1]
            seq = torch.cat([state_seq, action_tok], dim=1)
            mask = build_fusion_attention_mask(state_len, action_len, self.attention_mode, device=device)
        else:
            action_len = 0
            seq = state_seq
            mask = None

        out = self.transformer(seq, mask=mask)
        out = self.norm(out)

        L_state = T_hist * F
        global_token = out[:, L_state, :]
        feat_out = out[:, :L_state, :]
        action_out = out[:, state_len:, :] if action_len > 0 else None

        if not return_context_tokens:
            return cast(torch.Tensor, global_token)

        feature_idx = torch.arange(F, device=device)
        entity_ids_state = self.host_id_lut.to(device)[feature_idx].unsqueeze(0).expand(B, T_hist, -1).reshape(B, L_state)
        time_ids_state = self.time_axis.history_ids(B, device, length=T_hist).unsqueeze(-1).expand(B, T_hist, F).reshape(B, L_state)
        token_type_state = torch.full((B, L_state), TokenType.FEATURE, device=device, dtype=torch.long)

        if action_len > 0:
            act_idx = torch.clamp(actions, 0, self.type_map.shape[0] - 1)
            entity_ids_action = self.host_map[act_idx]
            time_ids_action = self.time_axis.horizon_ids(B, action_len, device)
            token_type_action = torch.full((B, action_len), TokenType.ACTION, device=device, dtype=torch.long)
            tokens = torch.cat([feat_out, action_out], dim=1)
            entity_ids = torch.cat([entity_ids_state, entity_ids_action], dim=1)
            time_ids = torch.cat([time_ids_state, time_ids_action], dim=1)
            token_type_ids = torch.cat([token_type_state, token_type_action], dim=1)
        else:
            tokens = feat_out
            entity_ids = entity_ids_state
            time_ids = time_ids_state
            token_type_ids = token_type_state

        L = tokens.shape[1]
        padding_mask = torch.zeros((B, L), device=device, dtype=torch.bool)

        ctx = ContextTokens(
            tokens=tokens,
            padding_mask=padding_mask,
            visibility_mask=None,
            time_ids=time_ids,
            entity_ids=entity_ids,
            token_type_ids=token_type_ids,
            global_token=global_token,
            metadata={
                "history_len": T_hist, "obs_dim": F, "action_len": action_len,
                "action_start_idx": L_state, "attention_mode": self.attention_mode,
                "fusion": "early", "representation": "feature_fused",
                "host_block_size": self.host_block_size,
            },
        )
        ctx.validate()
        return ctx

    def forward(
        self,
        x: torch.Tensor,
        actions: torch.Tensor | None = None,
        host_known_mask: torch.Tensor | None = None,
    ) -> ContextTokens | torch.Tensor:
        return self.encode_context(x, actions=actions, host_known_mask=host_known_mask, return_context_tokens=True)
