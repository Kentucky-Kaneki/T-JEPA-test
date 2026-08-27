"""
Shared Entity Embedding Tables for Phase 4 Fusion Representations.

Test 2 (no separate ActionEncoder) is only a meaningful test of *early fusion with a
shared semantic space* if the action's target-host/target-subnet embeddings are the
SAME nn.Embedding parameters used by the state tokenizer for that host/subnet - not
merely same-shaped tables trained independently. This module owns those shared tables
so both a state tokenizer (e.g. `FeatureFusedRepresentation`) and the action-token
builder inside the same encoder hold a reference to the identical embedding instance.

Host id convention (matches `cyber_jepa.models.predictor.ActionEncoder`):
    0     = NONE / not host-specific (Sleep, Monitor, Router-targeted actions)
    1..13 = Scenario1b host slots

Subnet id convention: 0 = NONE, 1..3 = subnets (reserved; not populated by the current
Scenario1b action table - see `action_semantics.py`).
"""

import torch
import torch.nn as nn


class SharedEntityEmbeddings(nn.Module):
    """Owns the host/subnet embedding tables shared between state and action tokens
    *within one fused encoder instance* (flat-fused and feature-fused each own their
    own `SharedEntityEmbeddings` - sharing is within-representation, not across the
    two representations)."""

    def __init__(self, hidden_dim: int = 64, num_hosts: int = 14, num_subnets: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.host_emb = nn.Embedding(num_hosts, hidden_dim)
        self.subnet_emb = nn.Embedding(num_subnets, hidden_dim)

    def embed_host(self, host_ids: torch.Tensor) -> torch.Tensor:
        return self.host_emb(host_ids)

    def embed_subnet(self, subnet_ids: torch.Tensor) -> torch.Tensor:
        return self.subnet_emb(subnet_ids)


class UnifiedTimeEmbedding(nn.Module):
    """
    Single continuous relative-time axis spanning history (past) and prediction
    horizon (future), so state and action tokens sit on one shared clock instead of
    two independently-learned time embeddings.

    Index convention:
        0 .. history_len-1             -> history steps O_{t-history_len+1} .. O_t
        history_len .. history_len+K-1 -> horizon steps a_t .. a_{t+K-1}
    """

    def __init__(self, hidden_dim: int, history_len: int, max_horizon: int):
        super().__init__()
        self.history_len = history_len
        self.max_horizon = max_horizon
        self.emb = nn.Embedding(history_len + max_horizon, hidden_dim)

    def history_ids(self, batch_size: int, device: torch.device, length: int | None = None) -> torch.Tensor:
        """Ids for a history/state pass of `length` timesteps (defaults to the
        configured `history_len`). `length` must be passed explicitly whenever the
        actual sequence length can differ from `history_len` - e.g. the EMA target
        encoder's single-frame (T=1) pass over `O_{t+1}`, which reuses this same
        encoder class with `actions=None` (see `jepa_fused.py`)."""
        n = length if length is not None else self.history_len
        return torch.arange(n, device=device).unsqueeze(0).expand(batch_size, -1)

    def horizon_ids(self, batch_size: int, k: int, device: torch.device) -> torch.Tensor:
        return self.history_len + torch.arange(k, device=device).unsqueeze(0).expand(batch_size, -1)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.emb(ids)
