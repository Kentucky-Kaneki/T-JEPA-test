"""
Flat Fused Representation - Phase 4, Test 2, Variant 2a.

Same input as `representations/flat.py` (raw 52-dim ChallengeWrapper vector, T_hist
history), but with NO separate `ActionEncoder` module. Action semantics (type, target
host, target subnet, horizon position, validity) are embedded with the shared host
/subnet tables from `shared_semantic_embeddings.py`, and injected as extra tokens into
the SAME Transformer stack that encodes context - i.e. action fusion happens inside the
encoder, not downstream in a predictor.

Note on the shared-embedding benefit here vs `feature_fused.py`: flat tokens are
per-timestep aggregates over all 13 hosts, not per-host, so there is no state-side
token for the action's shared host embedding to attend against directly the way there
is in the feature-token representation. The action token still embeds its target host
through the shared table (for architectural parity and so the ablation isolates
"fusion point" rather than "fusion point + embedding sharing" as two confounded
variables), but this variant is expected to show a weaker grounding effect than
`feature_fused.py` if shared embeddings matter - that contrast is itself part of what
this round of testing is meant to reveal.

The "predictor" for this variant is just a thin readout head
(`cyber_jepa.models.fused_readout.FusedLatentReadout`) applied to the encoder's own
output at the action-token positions - see `jepa_fused.py`.

Capacity: `num_layers`/`ffn_dim` are expected to come from
`cyber_jepa.models.capacity_matching.search_matched_fused_config` so this encoder's
trainable parameter count matches Test 1's (flat encoder + ActionEncoder + predictor
body) combined budget. This class does not perform that search itself - it just
accepts whatever `num_layers`/`ffn_dim` it is given.
"""

from typing import cast
import torch
import torch.nn as nn

from cyber_jepa.models.context import ContextTokens, TokenType
from cyber_jepa.models.action_semantics import build_scenario1b_action_lookup_buffers, NUM_ACTION_TYPES
from cyber_jepa.models.shared_semantic_embeddings import SharedEntityEmbeddings, UnifiedTimeEmbedding
from cyber_jepa.models.fusion_masking import build_fusion_attention_mask, STRICT


class FlatFusedRepresentation(nn.Module):
    """Early-fusion flat encoder: state tokens + action tokens, one shared Transformer."""

    type_map: torch.Tensor
    host_map: torch.Tensor
    subnet_map: torch.Tensor

    def __init__(
        self,
        obs_dim: int = 52,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,          # set via capacity_matching for the real Test 2 runs
        ffn_dim: int = 256,           # set via capacity_matching for the real Test 2 runs
        history_len: int = 4,
        max_horizon: int = 16,
        attention_mode: str = STRICT,
        shared_embeddings: SharedEntityEmbeddings | None = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.history_len = history_len
        self.max_horizon = max_horizon
        self.attention_mode = attention_mode

        # --- state tokenization (identical to flat.py's input_proj) ---
        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim, ffn_dim),
            nn.LayerNorm(ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.reg_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # --- shared semantic fields (host/subnet tables shared across state+action) ---
        self.shared_embeddings = shared_embeddings or SharedEntityEmbeddings(hidden_dim=hidden_dim)
        self.time_axis = UnifiedTimeEmbedding(hidden_dim, history_len=history_len, max_horizon=max_horizon)

        # --- action-only semantic fields (no state-side analogue in the flat representation) ---
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
        h_e = self.shared_embeddings.embed_host(host_ids)      # SHARED table
        s_e = self.shared_embeddings.embed_subnet(subnet_ids)  # SHARED table
        p_e = self.action_pos_emb(pos)
        v_e = self.action_valid_emb(valid_ids)
        time_e = self.time_axis(self.time_axis.horizon_ids(B, K, device))

        token = t_e + h_e + s_e + p_e + v_e + time_e
        return cast(torch.Tensor, self.action_proj(token))     # [B, K, D]

    def encode_context(
        self,
        x: torch.Tensor,                          # [B, T_hist, obs_dim]
        actions: torch.Tensor | None = None,       # [B, K] or None (None -> context-only pass)
        valid_mask: torch.Tensor | None = None,
        host_known_mask: torch.Tensor | None = None,
        return_context_tokens: bool = True,
    ) -> ContextTokens | torch.Tensor:
        B, T_hist, D_in = x.shape
        device = x.device

        state_tok = self.input_proj(x)                                   # [B, T_hist, D]
        state_tok = state_tok + self.time_axis(self.time_axis.history_ids(B, device, length=T_hist))
        reg_tok = self.reg_token.expand(B, -1, -1)                       # [B, 1, D]
        state_seq = torch.cat([state_tok, reg_tok], dim=1)               # [B, T_hist+1, D]
        state_len = state_seq.shape[1]

        if actions is not None:
            action_tok = self._build_action_tokens(actions, valid_mask)  # [B, K, D]
            action_len = action_tok.shape[1]
            seq = torch.cat([state_seq, action_tok], dim=1)              # [B, T_hist+1+K, D]
            mask = build_fusion_attention_mask(state_len, action_len, self.attention_mode, device=device)
        else:
            action_len = 0
            seq = state_seq
            mask = None

        out = self.transformer(seq, mask=mask)
        out = self.norm(out)

        global_token = out[:, T_hist, :]            # the reg token's position
        flat_tokens = out[:, :T_hist, :]
        action_out = out[:, state_len:, :] if action_len > 0 else None

        if not return_context_tokens:
            return cast(torch.Tensor, global_token)

        if action_len > 0:
            act_idx = torch.clamp(actions, 0, self.type_map.shape[0] - 1)
            entity_ids_action = self.host_map[act_idx]
            time_ids_action = self.time_axis.horizon_ids(B, action_len, device)
            token_type_action = torch.full((B, action_len), TokenType.ACTION, device=device, dtype=torch.long)
            tokens = torch.cat([flat_tokens, action_out], dim=1)
        else:
            entity_ids_action = torch.zeros((B, 0), device=device, dtype=torch.long)
            time_ids_action = torch.zeros((B, 0), device=device, dtype=torch.long)
            token_type_action = torch.zeros((B, 0), device=device, dtype=torch.long)
            tokens = flat_tokens

        entity_ids_state = torch.zeros((B, T_hist), device=device, dtype=torch.long)
        time_ids_state = self.time_axis.history_ids(B, device, length=T_hist)
        token_type_state = torch.full((B, T_hist), TokenType.TEMPORAL_FLAT, device=device, dtype=torch.long)

        entity_ids = torch.cat([entity_ids_state, entity_ids_action], dim=1)
        time_ids = torch.cat([time_ids_state, time_ids_action], dim=1)
        token_type_ids = torch.cat([token_type_state, token_type_action], dim=1)

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
                "history_len": T_hist, "obs_dim": D_in, "action_len": action_len,
                "action_start_idx": T_hist, "attention_mode": self.attention_mode,
                "fusion": "early", "representation": "flat_fused",
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
