"""
Action-Conditioned Transformer Predictor & Target Query Granularity Interface for Cyber-JEPA.

Takes encoded context representation z_t (vector or token memory) and K sequence of action tokens,
using cross-attention or causal self-attention over horizon sequence [1..K].
Exposes TargetSpec query interface for predicting feature, host, subnet, or network targets.
"""

from dataclasses import dataclass
from typing import Any, cast
import torch
import torch.nn as nn

from cyber_jepa.models.context import ContextTokens


def compute_jepa_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Layer-Normalized Smooth L1 JEPA Loss."""
    norm_pred = nn.functional.layer_norm(pred, pred.shape[-1:])
    norm_target = nn.functional.layer_norm(target, target.shape[-1:])
    return nn.functional.smooth_l1_loss(norm_pred, norm_target)


@dataclass
class TargetSpec:
    """Target Granularity Query Specification (Phase 5 Contract)."""
    horizon: int                             # k in 1..K
    granularity: str                         # 'feature', 'host', 'subnet', 'network'
    target_id: int = 0                      # Entity index (0..12 for host, 0..2 for subnet, 0 for network)
    valid_mask: torch.Tensor | None = None   # Boolean validity mask


class ActionEncoder(nn.Module):
    """Categorical & Structural Action Encoder (Phase 6 Contract)."""

    type_map: torch.Tensor
    host_map: torch.Tensor
    subnet_map: torch.Tensor

    def __init__(
        self,
        num_action_types: int = 16,
        num_hosts: int = 14,                # 0=NONE, 1..13=hosts
        num_subnets: int = 4,               # 0=NONE, 1..3=subnets
        max_horizon: int = 16,
        hidden_dim: int = 64,
        projection_depth: int = 2,
    ):
        super().__init__()
        self.num_action_types = num_action_types
        self.num_hosts = num_hosts
        self.num_subnets = num_subnets
        self.max_horizon = max_horizon
        self.hidden_dim = hidden_dim

        self.type_emb = nn.Embedding(num_action_types, hidden_dim)
        self.host_emb = nn.Embedding(num_hosts, hidden_dim)
        self.subnet_emb = nn.Embedding(num_subnets, hidden_dim)
        self.pos_emb = nn.Embedding(max_horizon, hidden_dim)
        self.valid_emb = nn.Embedding(2, hidden_dim)

        if projection_depth < 1:
            raise ValueError("projection_depth must be at least one")
        projection: list[nn.Module] = []
        for _ in range(projection_depth - 1):
            projection.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        projection.append(nn.Linear(hidden_dim, hidden_dim))
        self.proj = nn.Sequential(*projection)

        # Build 66-element Scenario1b action mapping lookup buffers
        type_ids, host_ids, subnet_ids = self._build_scenario1b_action_tables()
        self.register_buffer("type_map", torch.tensor(type_ids, dtype=torch.long))
        self.register_buffer("host_map", torch.tensor(host_ids, dtype=torch.long))
        self.register_buffer("subnet_map", torch.tensor(subnet_ids, dtype=torch.long))

    def _build_scenario1b_action_tables(self) -> tuple[list[int], list[int], list[int]]:
        """Build deterministic lookup tables for discrete action indices 0..65."""
        types = [0] * 66
        hosts = [0] * 66
        subnets = [0] * 66

        types[0] = 0; hosts[0] = 0; subnets[0] = 0 # Sleep
        types[1] = 1; hosts[1] = 0; subnets[1] = 0 # Monitor

        # 2..17 Analyse
        for idx in range(2, 18):
            types[idx] = 2
            if idx <= 14:
                hosts[idx] = (idx - 2) + 1 # 1..13
            else:
                hosts[idx] = 0 # Router

        # 18..33 Remove
        for idx in range(18, 34):
            types[idx] = 3
            if idx <= 30:
                hosts[idx] = (idx - 18) + 1
            else:
                hosts[idx] = 0

        # 34..49 Misinform / Decoy
        for idx in range(34, 50):
            types[idx] = 0
            if idx <= 46:
                hosts[idx] = (idx - 34) + 1
            else:
                hosts[idx] = 0

        # 50..65 Restore
        for idx in range(50, 66):
            types[idx] = 4
            if idx <= 62:
                hosts[idx] = (idx - 50) + 1
            else:
                hosts[idx] = 0

        return types, hosts, subnets

    def forward(
        self,
        actions: torch.Tensor,               # [B, K] discrete indices OR [B, K, 3] categorical IDs
        valid_mask: torch.Tensor | None = None, # [B, K] bool
    ) -> torch.Tensor:
        """Returns action tokens [B, K, hidden_dim]."""
        B, K = actions.shape[0], actions.shape[1]
        device = actions.device

        if actions.dim() == 2:
            act_idx = torch.clamp(actions, 0, 65)
            type_ids = self.type_map[act_idx]
            host_ids = self.host_map[act_idx]
            subnet_ids = self.subnet_map[act_idx]
        elif actions.dim() == 3 and actions.shape[2] >= 3:
            type_ids = actions[:, :, 0]
            host_ids = actions[:, :, 1]
            subnet_ids = actions[:, :, 2]
        else:
            raise ValueError(f"Invalid actions shape: {actions.shape}")

        pos = torch.arange(K, device=device).unsqueeze(0).expand(B, K)
        if valid_mask is None:
            valid_ids = torch.ones((B, K), device=device, dtype=torch.long)
        else:
            valid_ids = valid_mask.to(torch.long)

        t_e = self.type_emb(type_ids)
        h_e = self.host_emb(host_ids)
        s_e = self.subnet_emb(subnet_ids)
        p_e = self.pos_emb(pos)
        v_e = self.valid_emb(valid_ids)

        token = t_e + h_e + s_e + p_e + v_e
        return cast(torch.Tensor, self.proj(token)) # [B, K, hidden_dim]


class LatentPredictor(nn.Module):
    """Transformer Predictor mapping context and action sequence to target queries z_{t+k}."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        ffn_dim: int = 256,
        max_horizon: int = 16,
        action_projection_depth: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_horizon = max_horizon

        self.action_encoder = ActionEncoder(
            hidden_dim=hidden_dim,
            max_horizon=max_horizon,
            projection_depth=action_projection_depth,
        )

        # Target Granularity Embeddings
        self.granularity_emb = nn.Embedding(4, hidden_dim) # 0=feature, 1=host, 2=subnet, 3=network
        self.entity_emb = nn.Embedding(16, hidden_dim)      # 0..15 entity index
        self.horizon_emb = nn.Embedding(max_horizon + 1, hidden_dim)

        # Transformer Predictor Encoder
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

        # Decoder for token-preserving cross-attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        z_t: torch.Tensor | ContextTokens,   # [B, hidden_dim], [B, L, D], or ContextTokens
        actions: torch.Tensor,               # [B, K] discrete action indices
        target_spec: TargetSpec | None = None,
        memory_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict z_{t+k} given context and action sequence."""
        memory_tokens: torch.Tensor | None = None
        if isinstance(z_t, ContextTokens):
            memory_tokens = z_t.tokens
            memory_padding_mask = z_t.padding_mask
            if z_t.global_token is not None and z_t.tokens.dim() == 3:
                z_t_ctx = z_t.global_token
            else:
                z_t_ctx = z_t.tokens
        else:
            z_t_ctx = z_t
            memory_tokens = z_t if z_t.dim() == 3 else None

        B = z_t_ctx.shape[0]
        device = z_t_ctx.device
        act_tokens = self.action_encoder(actions)
        K = act_tokens.shape[1]

        # Case A: Token-preserving cross-attention prediction (z_t_ctx is 3D memory [B, L, D])
        if z_t_ctx.dim() == 3:
            k = target_spec.horizon if target_spec is not None else K
            h_e = self.horizon_emb(torch.tensor(min(k, self.max_horizon), device=device))

            gran_map = {"feature": 0, "host": 1, "subnet": 2, "network": 3}
            g_id = gran_map.get(target_spec.granularity.lower(), 3) if target_spec is not None else 3
            g_e = self.granularity_emb(torch.tensor(g_id, device=device))

            t_id = min(target_spec.target_id, 15) if target_spec is not None else 0
            e_e = self.entity_emb(torch.tensor(t_id, device=device))

            target_query = (g_e + e_e + h_e).unsqueeze(0).unsqueeze(1).expand(B, 1, -1) # [B, 1, D]

            tgt_seq = torch.cat([target_query, act_tokens], dim=1)

            decoded = self.decoder(
                tgt=tgt_seq,
                memory=z_t_ctx,
                memory_key_padding_mask=memory_padding_mask,
            )
            decoded = self.norm(decoded)
            pred_latent = self.head(decoded[:, 0, :]) # [B, D]
            return cast(torch.Tensor, pred_latent)

        # Case B: Standard vector prediction (z_t_ctx is 2D [B, D])
        ctx_token = z_t_ctx.unsqueeze(1) # [B, 1, D]
        seq = torch.cat([ctx_token, act_tokens], dim=1) # [B, 1+K, D]

        causal_mask = torch.triu(torch.ones(1 + K, 1 + K, device=device) * float('-inf'), diagonal=1)
        out = self.transformer(seq, mask=causal_mask)
        out = self.norm(out)

        pred_latents = self.head(out[:, 1:, :])

        if target_spec is not None:
            k = min(target_spec.horizon, K)
            k_idx = k - 1
            pred_k = pred_latents[:, k_idx, :] # [B, D]

            gran_map = {"feature": 0, "host": 1, "subnet": 2, "network": 3}
            g_id = gran_map.get(target_spec.granularity.lower(), 3)
            g_emb = self.granularity_emb(torch.tensor(g_id, device=device))
            e_emb = self.entity_emb(torch.tensor(min(target_spec.target_id, 15), device=device))

            return cast(torch.Tensor, pred_k + g_emb + e_emb)
        else:
            return cast(torch.Tensor, pred_latents)
