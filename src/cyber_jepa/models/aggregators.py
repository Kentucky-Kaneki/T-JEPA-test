"""
Explicit Context Aggregators for Cyber-JEPA Phase 3.

Isolates representation aggregation into explicit, configurable components:
1. LegacyLastStepMean: Final timestep mask-aware mean pooling (controlled baseline).
2. LearnedQueryPool: Cross-attention over all valid temporal/entity tokens with a learned query.
3. TokenPreservingAggregator: Preserves token memory [B, L, D] directly for cross-attention prediction.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
import torch
import torch.nn as nn

from cyber_jepa.models.context import ContextTokens


@dataclass
class AggregatedContext:
    """Output structure returned by ContextAggregator implementations."""
    latent: torch.Tensor                          # [B, D] or [B, L, D]
    memory_tokens: torch.Tensor                   # [B, L, D]
    attention_weights: torch.Tensor | None = None # [B, 1, L] or None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ContextAggregator(Protocol):
    """Protocol interface for Cyber-JEPA context aggregation interventions."""

    def __call__(self, context: ContextTokens) -> AggregatedContext:
        ...

    def forward(self, context: ContextTokens) -> AggregatedContext:
        ...


class LegacyLastStepMean(nn.Module):
    """Controlled compression baseline: mask-aware mean over final timestep entities."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, context: ContextTokens) -> AggregatedContext:
        context.validate()
        B, L, D = context.tokens.shape

        if context.global_token is not None:
            latent = context.global_token
        else:
            valid_mask = (~context.padding_mask).float().unsqueeze(-1) # [B, L, 1]
            sum_tokens = (context.tokens * valid_mask).sum(dim=1)      # [B, D]
            count_tokens = valid_mask.sum(dim=1).clamp(min=1.0)       # [B, 1]
            latent = sum_tokens / count_tokens

        return AggregatedContext(
            latent=latent,
            memory_tokens=context.tokens,
            attention_weights=None,
            metadata={"aggregator": "legacy_last_step_mean"},
        )


class LearnedQueryPool(nn.Module):
    """Learned query cross-attention pooling across all valid temporal and entity tokens."""

    def __init__(self, hidden_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, context: ContextTokens) -> AggregatedContext:
        context.validate()
        B, L, D = context.tokens.shape

        q = self.query.expand(B, -1, -1) # [B, 1, D]
        key_padding_mask = context.padding_mask # [B, L] (True = masked)

        attn_out, attn_weights = self.cross_attn(
            query=q,
            key=context.tokens,
            value=context.tokens,
            key_padding_mask=key_padding_mask,
        )

        latent = self.norm(q + attn_out).squeeze(1) # [B, D]

        return AggregatedContext(
            latent=latent,
            memory_tokens=context.tokens,
            attention_weights=attn_weights,
            metadata={"aggregator": "learned_query_pool"},
        )


class TokenPreservingAggregator(nn.Module):
    """Preserves full token memory [B, L, D] without pooling for token-preserving prediction."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, context: ContextTokens) -> AggregatedContext:
        context.validate()
        return AggregatedContext(
            latent=context.tokens,
            memory_tokens=context.tokens,
            attention_weights=None,
            metadata={"aggregator": "token_preserving_predictor"},
        )
