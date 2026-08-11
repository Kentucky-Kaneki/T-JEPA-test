"""
Hierarchical Host/Subnet Representation for Cyber-JEPA.

Combines host tokens with explicit subnet tokens and global network tokens using a two-stage
attention hierarchy:
Stage 1: Host tokens attend within their subnets to update subnet tokens using subnet masks.
Stage 2: Subnet tokens and global network token attend globally.
Exposes ContextTokens and HierarchicalOutput structures.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from cyber_jepa.models.context import ContextTokens, TokenType
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
            mask = [0.0 if host_subnets[h] == s else float("-inf") for h in range(num_hosts)]
            subnet_masks.append(mask)
        self.register_buffer("subnet_attn_mask", torch.tensor(subnet_masks, dtype=torch.float32)) # [3, 13]

    def encode_context(
        self,
        flat_obs: torch.Tensor,
        host_known_mask: torch.Tensor | None = None,
        return_context_tokens: bool = True,
    ) -> ContextTokens | HierarchicalOutput:
        """
        Input: flat_obs [B, T_hist, 52]
        Returns: ContextTokens or HierarchicalOutput typed structure
        """
        B, T_hist, D = flat_obs.shape[0], flat_obs.shape[1], flat_obs.shape[2]
        device = flat_obs.device

        # Get base host tokens: [B, T_hist * 13, hidden_dim] or ContextTokens
        host_ctx = self.host_encoder.encode_context(flat_obs, host_known_mask, return_context_tokens=True)
        assert isinstance(host_ctx, ContextTokens)
        host_out = host_ctx.tokens.view(B, T_hist, self.num_hosts, self.hidden_dim)

        sub_tokens_list = []
        global_tokens_list = []

        sub_queries = self.subnet_tokens.expand(B, -1, -1) # [B, 3, D]

        for t in range(T_hist):
            h_t = host_out[:, t, :, :] # [B, 13, D]

            # Stage 1: Subnet queries attend to host tokens with subnet topology mask
            stage1_out, _ = self.stage1_attn(
                query=sub_queries,
                key=h_t,
                value=h_t,
                attn_mask=self.subnet_attn_mask,
            )
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

        if not return_context_tokens:
            return HierarchicalOutput(
                host_tokens=host_out,
                subnet_tokens=subnet_out,
                global_token=global_out,
            )

        # Concatenate 13 host tokens and 3 subnet tokens per timestep -> 16 tokens per timestep
        combined_tokens = []
        type_ids_list = []
        entity_ids_list = []
        time_ids_list = []

        for t in range(T_hist):
            h_t = host_out[:, t, :, :] # [B, 13, D]
            s_t = subnet_out[:, t, :, :] # [B, 3, D]
            comb_t = torch.cat([h_t, s_t], dim=1) # [B, 16, D]
            combined_tokens.append(comb_t)

            t_ids = torch.full((B, 16), t, device=device, dtype=torch.long)
            e_ids = torch.cat([
                torch.arange(self.num_hosts, device=device),
                torch.arange(self.num_subnets, device=device)
            ]).unsqueeze(0).expand(B, -1)
            tp_ids = torch.cat([
                torch.full((self.num_hosts,), TokenType.HOST, device=device, dtype=torch.long),
                torch.full((self.num_subnets,), TokenType.SUBNET, device=device, dtype=torch.long)
            ]).unsqueeze(0).expand(B, -1)

            time_ids_list.append(t_ids)
            entity_ids_list.append(e_ids)
            type_ids_list.append(tp_ids)

        tokens = torch.cat(combined_tokens, dim=1) # [B, T_hist * 16, D]
        time_ids = torch.cat(time_ids_list, dim=1)
        entity_ids = torch.cat(entity_ids_list, dim=1)
        token_type_ids = torch.cat(type_ids_list, dim=1)

        total_L = T_hist * 16
        padding_mask = torch.zeros((B, total_L), device=device, dtype=torch.bool)

        ctx = ContextTokens(
            tokens=tokens,
            padding_mask=padding_mask,
            visibility_mask=None,
            time_ids=time_ids,
            entity_ids=entity_ids,
            token_type_ids=token_type_ids,
            global_token=global_out,
            metadata={"history_len": T_hist, "num_hosts": self.num_hosts, "num_subnets": self.num_subnets},
        )
        ctx.validate()
        return ctx

    def forward(
        self,
        flat_obs: torch.Tensor,
        host_known_mask: torch.Tensor | None = None,
    ) -> ContextTokens | HierarchicalOutput:
        return self.encode_context(flat_obs, host_known_mask=host_known_mask, return_context_tokens=True)
