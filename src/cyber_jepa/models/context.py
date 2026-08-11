"""
Typed Context Tokens Interface for Cyber-JEPA Phase 3.

Replaces implicit tensor rank branching with an explicit representation contract
carrying tokens, padding masks, visibility masks, temporal/entity/type IDs,
and optional global tokens.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import torch


class TokenType(IntEnum):
    """Enumeration of token semantic types."""
    TEMPORAL_FLAT = 0
    FEATURE = 1
    HOST = 2
    SUBNET = 3
    GLOBAL = 4
    ACTION = 5
    TARGET_QUERY = 6


@dataclass
class ContextTokens:
    """
    Deliberate typed context representation passed from encoders to aggregators/predictors.

    Shapes:
        tokens:          [B, L, D]
        padding_mask:    [B, L] (bool: True indicates padded/invalid token)
        visibility_mask: [B, L] or None (bool: True indicates observed entity)
        time_ids:        [B, L] (int64: relative timestep indices)
        entity_ids:      [B, L] (int64: feature or host/subnet slot indices)
        token_type_ids:  [B, L] (int64: TokenType enum values)
        global_token:    [B, D] or None (optional summary token)
        metadata:        dict[str, Any]
    """
    tokens: torch.Tensor
    padding_mask: torch.Tensor
    visibility_mask: torch.Tensor | None
    time_ids: torch.Tensor
    entity_ids: torch.Tensor
    token_type_ids: torch.Tensor
    global_token: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device) -> "ContextTokens":
        """Move all underlying PyTorch tensors to target device."""
        return ContextTokens(
            tokens=self.tokens.to(device),
            padding_mask=self.padding_mask.to(device),
            visibility_mask=self.visibility_mask.to(device) if self.visibility_mask is not None else None,
            time_ids=self.time_ids.to(device),
            entity_ids=self.entity_ids.to(device),
            token_type_ids=self.token_type_ids.to(device),
            global_token=self.global_token.to(device) if self.global_token is not None else None,
            metadata=self.metadata,
        )

    def validate(self) -> None:
        """Validate shape consistency across all fields."""
        if self.tokens.dim() != 3:
            raise ValueError(f"tokens must be 3D tensor [B, L, D], got shape {self.tokens.shape}")

        B, L, D = self.tokens.shape

        if self.padding_mask.shape != (B, L):
            raise ValueError(f"padding_mask shape must be ({B}, {L}), got {self.padding_mask.shape}")
        if self.visibility_mask is not None and self.visibility_mask.shape != (B, L):
            raise ValueError(f"visibility_mask shape must be ({B}, {L}), got {self.visibility_mask.shape}")
        if self.time_ids.shape != (B, L):
            raise ValueError(f"time_ids shape must be ({B}, {L}), got {self.time_ids.shape}")
        if self.entity_ids.shape != (B, L):
            raise ValueError(f"entity_ids shape must be ({B}, {L}), got {self.entity_ids.shape}")
        if self.token_type_ids.shape != (B, L):
            raise ValueError(f"token_type_ids shape must be ({B}, {L}), got {self.token_type_ids.shape}")
        if self.global_token is not None and self.global_token.shape != (B, D):
            raise ValueError(f"global_token shape must be ({B}, {D}), got {self.global_token.shape}")
