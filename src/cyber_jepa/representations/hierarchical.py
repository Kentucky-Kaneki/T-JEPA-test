"""
Hierarchical Host/Subnet Representation for Cyber-JEPA.

Combines host tokens with explicit subnet tokens and global network tokens using a two-stage
attention hierarchy:
Stage 1: Host tokens attend within their subnets to update subnet tokens.
Stage 2: Subnet tokens and global network token attend globally.
Exposes a typed HierarchicalOutput dataclass.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from cyber_jepa.representations.host import HostTokenRepresentation


@dataclass
class HierarchicalOutput:
    """Typed output structure for HierarchicalHostSubnetRepresentation."""
    host_tokens: torch.Tensor                # [B, T_hist, 13, hidden_dim]
    subnet_tokens: torch.Tensor              # [B, T_hist, 3, hidden_dim]
    global_token: torch.Tensor               # [B, hidden_dim]


class HierarchicalHostSubnetRepresentation(nn.Module):
    """Hierarchical two-stage host/subnet/network Transformer encoder."""

    def __init__(
        self,
        num_hosts: int = 13,
        num_subnets: int = 3,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        ffn_dim: int = 256,
        history_len: int = 4,
    ):
        super().__init__()
        self.num_hosts = num_hosts
        self.num_subnets = num_subnets
        self.hidden_dim = hidden_dim
        self.history_len = history_len

        # Base host token encoder
        self.host_encoder = HostTokenRepresentation(
            num_hosts=num_hosts,
            num_subnets=num_subnets,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            history_len=history_len,
        )

        # Learned Subnet tokens & Global network token
        self.subnet_tokens = nn.Parameter(torch.randn(1, num_subnets, hidden_dim) * 0.02)
        self.global_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Subnet-level Cross Attention (Stage 1)
        self.stage1_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.stage1_norm = nn.LayerNorm(hidden_dim)

        # Global-level Self Attention (Stage 2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.stage2_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.stage2_norm = nn.LayerNorm(hidden_dim)

        # Subnet membership indices (0=Enterprise [4 hosts], 1=Operational [4 hosts], 2=User [5 hosts])
        subnet_masks = []
        host_subnets = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        for s in range(num_subnets):
            mask = [1.0 if host_subnets[h] == s else 0.0 for h in range(num_hosts)]
            subnet_masks.append(mask)
        self.register_buffer("subnet_masks", torch.tensor(subnet_masks, dtype=torch.float32)) # [3, 13]

    def forward(
        self,
        flat_obs: torch.Tensor,
        host_known_mask: torch.Tensor | None = None,
    ) -> HierarchicalOutput:
        """
        Input: flat_obs [B, T_hist, 52]
        Returns: HierarchicalOutput typed structure
        """
        B, T_hist, D = flat_obs.shape[0], flat_obs.shape[1], flat_obs.shape[2]
        device = flat_obs.device

        # Get base host tokens: [B, T_hist, 13, hidden_dim]
        host_out = self.host_encoder(flat_obs, host_known_mask)

        # Process per timestep
        sub_tokens_list = []
        global_tokens_list = []

        sub_queries = self.subnet_tokens.expand(B, -1, -1) # [B, 3, D]

        for t in range(T_hist):
            h_t = host_out[:, t, :, :] # [B, 13, D]

            # Stage 1: Subnet queries attend to host tokens
            stage1_out, _ = self.stage1_attn(query=sub_queries, key=h_t, value=h_t)
            stage1_out = self.stage1_norm(sub_queries + stage1_out) # [B, 3, D]
            sub_tokens_list.append(stage1_out.unsqueeze(1))

            # Stage 2: Subnet tokens + Global token attend globally
            g_t = self.global_token.expand(B, -1, -1) # [B, 1, D]
            stage2_in = torch.cat([stage1_out, g_t], dim=1) # [B, 4, D]
            stage2_out = self.stage2_transformer(stage2_in)
            stage2_out = self.stage2_norm(stage2_out)

            global_tokens_list.append(stage2_out[:, -1, :]) # [B, D]

        subnet_out = torch.cat(sub_tokens_list, dim=1) # [B, T_hist, 3, D]
        global_out = global_tokens_list[-1]            # Latest timestep global token [B, D]

        return HierarchicalOutput(
            host_tokens=host_out,
            subnet_tokens=subnet_out,
            global_token=global_out,
        )
