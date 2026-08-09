"""
Canonical Host Feature Schema & Extractor for CybORG Scenario1b.

Derives canonical host features from flat observations and BlueTableWrapper output
without argmax heuristics over arbitrary vector slices.
"""

from dataclasses import dataclass

import torch


@dataclass
class CanonicalHostState:
    """Canonical 13-host categorical state batch."""
    host_features: torch.Tensor             # [B, T, 13, 4] raw numerical feature vector per host
    activity_ids: torch.Tensor              # [B, T, 13] long: 0=None, 1=Scan, 2=Exploit, 3=UNKNOWN
    compromise_ids: torch.Tensor            # [B, T, 13] long: 0=No, 1=Unknown, 2=User, 3=Privileged, 4=UNKNOWN
    subnet_ids: torch.Tensor                # [13] long: 0=Enterprise, 1=Operational, 2=User
    known_mask: torch.Tensor                # [B, T, 13] bool: host visibility mask


class CanonicalHostExtractor:
    """Extracts canonical categorical and numerical host states from 52-dim vectors."""

    # Map each host (0..12) to subnet ID (0=Enterprise, 1=Operational, 2=User)
    SUBNET_MAP = torch.tensor([
        0, 0, 0, 0,  # Defender, Enterprise0, Enterprise1, Enterprise2
        1, 1, 1, 1,  # Op_Host0, Op_Host1, Op_Host2, Op_Server0
        2, 2, 2, 2, 2 # User0, User1, User2, User3, User4
    ], dtype=torch.long)

    @classmethod
    def extract_from_vector(
        cls,
        flat_obs: torch.Tensor,              # [B, T, 52] or [B, 52]
        known_mask: torch.Tensor | None = None, # [B, T, 13] or [B, 13]
    ) -> CanonicalHostState:
        """Parse 52-dim flat vector into canonical host features and categorical IDs."""
        has_time = (flat_obs.dim() == 3)
        if not has_time:
            flat_obs = flat_obs.unsqueeze(1) # [B, 1, 52]
            if known_mask is not None and known_mask.dim() == 2:
                known_mask = known_mask.unsqueeze(1) # [B, 1, 13]

        B, T, D = flat_obs.shape
        if D != 52:
            raise ValueError(f"Expected flat observation vector of size 52, got {D}")

        device = flat_obs.device

        # Reshape [B, T, 52] -> [B, T, 13, 4]
        host_features = flat_obs.view(B, T, 13, 4)

        # 1. Activity ID (0=None, 1=Scan, 2=Exploit, 3=UNKNOWN)
        act_f0 = host_features[:, :, :, 0] # Scan bit
        act_f1 = host_features[:, :, :, 1] # Exploit bit

        activity_ids = torch.zeros((B, T, 13), device=device, dtype=torch.long)
        activity_ids = torch.where((act_f0 > 0.5) & (act_f1 <= 0.5), torch.tensor(1, device=device), activity_ids)
        activity_ids = torch.where((act_f1 > 0.5), torch.tensor(2, device=device), activity_ids)

        # 2. Compromise ID (0=No, 1=Unknown, 2=User, 3=Privileged, 4=UNKNOWN)
        comp_f0 = host_features[:, :, :, 2]
        comp_f1 = host_features[:, :, :, 3]

        compromise_ids = torch.zeros((B, T, 13), device=device, dtype=torch.long)
        compromise_ids = torch.where((comp_f0 > 0.5) & (comp_f1 <= 0.5), torch.tensor(1, device=device), compromise_ids)
        compromise_ids = torch.where((comp_f0 <= 0.5) & (comp_f1 > 0.5), torch.tensor(2, device=device), compromise_ids)
        compromise_ids = torch.where((comp_f0 > 0.5) & (comp_f1 > 0.5), torch.tensor(3, device=device), compromise_ids)

        if known_mask is None:
            known_mask = torch.ones((B, T, 13), device=device, dtype=torch.bool)

        subnet_ids = cls.SUBNET_MAP.to(device)

        if not has_time:
            host_features = host_features.squeeze(1)
            activity_ids = activity_ids.squeeze(1)
            compromise_ids = compromise_ids.squeeze(1)
            known_mask = known_mask.squeeze(1)

        return CanonicalHostState(
            host_features=host_features,
            activity_ids=activity_ids,
            compromise_ids=compromise_ids,
            subnet_ids=subnet_ids,
            known_mask=known_mask,
        )
