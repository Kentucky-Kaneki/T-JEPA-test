"""
Target Sampling & Context Masking Module for Cyber-JEPA.

Implements semantic target block sampling (50% changed targets, 50% uniform)
and spatial/temporal context masking using learned mask tokens.
"""

from typing import Any
import torch
import torch.nn as nn


class TargetMasker(nn.Module):
    """Context masking module substituting target entities at timestep t with learned mask tokens."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mask_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

    def sample_targets_and_mask(
        self,
        context_tokens: torch.Tensor,       # [B, T_hist, N_entities, D]
        target_tokens: torch.Tensor,        # [B, N_entities, D]
        changed_mask: torch.Tensor | None = None, # [B, N_entities] boolean changed indicator
        uniform_ratio: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample target entities per batch item and apply learned mask tokens to timestep t.

        Returns:
            masked_context: [B, T_hist, N_entities, D]
            selected_targets: [B, D] target latents
            target_indices: [B] discrete entity index selected as target
            validity_mask: [B, T_hist, N_entities] 1 for valid context, 0 for target-masked
        """
        B, T_hist, N_entities, D = context_tokens.shape
        device = context_tokens.device

        selected_indices = []
        for b in range(B):
            if changed_mask is not None and (not self.training or torch.rand(1).item() > uniform_ratio):
                changed_units = torch.nonzero(changed_mask[b]).flatten()
                if len(changed_units) > 0:
                    idx = changed_units[torch.randint(0, len(changed_units), (1,)).item()].item()
                else:
                    idx = torch.randint(0, N_entities, (1,)).item()
            else:
                idx = torch.randint(0, N_entities, (1,)).item()
            selected_indices.append(idx)

        target_idx_tensor = torch.tensor(selected_indices, device=device, dtype=torch.long)

        # Clone context
        masked_context = context_tokens.clone()
        validity_mask = torch.ones((B, T_hist, N_entities), device=device, dtype=torch.float32)

        # Substitute target entity at latest context timestep (t = T_hist - 1) with learned mask token
        for b in range(B):
            t_target_idx = selected_indices[b]
            masked_context[b, T_hist - 1, t_target_idx] = self.mask_token.squeeze(0)
            validity_mask[b, T_hist - 1, t_target_idx] = 0.0

        # Select target latents
        target_latents = target_tokens[torch.arange(B, device=device), target_idx_tensor] # [B, D]

        return masked_context, target_latents, target_idx_tensor, validity_mask
