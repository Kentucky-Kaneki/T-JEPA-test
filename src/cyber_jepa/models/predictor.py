"""
Action-Conditioned Transformer Predictor & Target Query Granularity Interface for Cyber-JEPA.

Takes encoded context representation z_t and K sequence of action tokens,
using cross-attention or causal self-attention over horizon sequence [1..K].
Exposes TargetSpec query interface for predicting feature, host, subnet, or network targets.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


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
    target_id: int                           # Entity index (0..12 for host, 0..2 for subnet, 0 for network)
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

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Build 66-element Scenario1b action mapping lookup buffers
        type_ids, host_ids, subnet_ids = self._build_scenario1b_action_tables()
        self.register_buffer("type_map", torch.tensor(type_ids, dtype=torch.long))
        self.register_buffer("host_map", torch.tensor(host_ids, dtype=torch.long))
        self.register_buffer("subnet_map", torch.tensor(subnet_ids, dtype=torch.long))

    def _build_scenario1b_action_tables(self) -> tuple[list[int], list[int], list[int]]:
        """Build deterministic lookup tables for discrete action indices 0..65."""
        # Standard Scenario1b 66-action mapping definition
        types = [0]*66
        hosts = [0]*66
        subnets = [0]*66

        # Fill via semantic rules matching ActionMapper
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
            # Discrete indices [B, K] -> deterministic lookup table
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
        return self.proj(token) # [B, K, hidden_dim]


class LatentPredictor(nn.Module):
    """Transformer Predictor mapping z_t and action sequence to target queries z_{t+k}."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        ffn_dim: int = 256,
        max_horizon: int = 16,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_horizon = max_horizon

        self.action_encoder = ActionEncoder(hidden_dim=hidden_dim, max_horizon=max_horizon)

        # Target Granularity Embeddings (Phase 5)
        self.granularity_emb = nn.Embedding(4, hidden_dim) # 0=feature, 1=host, 2=subnet, 3=network
        self.entity_emb = nn.Embedding(16, hidden_dim)      # 0..15 entity index

        # Transformer Predictor
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
        self.head = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        z_t: torch.Tensor,                   # [B, hidden_dim] OR [B, T_hist, hidden_dim]
        actions: torch.Tensor,               # [B, K] discrete action indices
        target_spec: TargetSpec | None = None,
    ) -> torch.Tensor:
        """
        Predict z_{t+k} given context z_t and action sequence.
        Returns predicted latent [B, K, hidden_dim] or single target latent [B, hidden_dim].
        """
        B = z_t.shape[0]
        device = z_t.device

        if z_t.dim() == 3:
            # Aggregate history tokens to context vector
            z_t_ctx = z_t.mean(dim=1) # [B, hidden_dim]
        else:
            z_t_ctx = z_t

        # Encode actions: [B, K, hidden_dim]
        act_tokens = self.action_encoder(actions)
        K = act_tokens.shape[1]

        # Context token: [B, 1, hidden_dim]
        ctx_token = z_t_ctx.unsqueeze(1)

        # Build sequence: [ctx_token, act_1, act_2, ..., act_K]
        seq = torch.cat([ctx_token, act_tokens], dim=1) # [B, 1+K, hidden_dim]

        # Causal mask for autoregressive propagation
        causal_mask = torch.triu(torch.ones(1 + K, 1 + K, device=device) * float('-inf'), diagonal=1)

        out = self.transformer(seq, mask=causal_mask)
        out = self.norm(out)

        # Latent predictions for horizons 1..K: [B, K, hidden_dim]
        pred_latents = self.head(out[:, 1:, :])

        if target_spec is not None:
            # Query target granularity & horizon k
            k = min(target_spec.horizon, K)
            k_idx = k - 1 # 0-indexed in pred_latents

            pred_k = pred_latents[:, k_idx, :] # [B, hidden_dim]

            # Query conditioning
            gran_map = {"feature": 0, "host": 1, "subnet": 2, "network": 3}
            g_id = gran_map.get(target_spec.granularity.lower(), 3)

            g_emb = self.granularity_emb(torch.tensor(g_id, device=device))
            e_emb = self.entity_emb(torch.tensor(min(target_spec.target_id, 15), device=device))

            query_conditioned = pred_k + g_emb + e_emb
            return query_conditioned
        else:
            return pred_latents
