"""
Host-Token Representation for Cyber-JEPA.

Encodes each of the 13 network hosts as an explicit semantic entity combining host identity,
subnet membership, activity, compromise status, visibility, and relative time embeddings.
"""

import torch
import torch.nn as nn


class HostTokenRepresentation(nn.Module):
    """Host-level tokenization and Transformer encoder."""

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

        # Categorical vocabulary embeddings
        self.host_id_emb = nn.Embedding(num_hosts, hidden_dim)
        self.subnet_id_emb = nn.Embedding(num_subnets, hidden_dim)
        self.activity_emb = nn.Embedding(5, hidden_dim)    # None, Scan, Exploit, UNKNOWN, Other
        self.compromise_emb = nn.Embedding(5, hidden_dim)  # No, Unknown, User, Privileged, UNKNOWN
        self.visibility_emb = nn.Embedding(2, hidden_dim)  # 0=unobserved, 1=known
        self.time_emb = nn.Embedding(history_len, hidden_dim)

        # Host-local MLP fusion
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 5, ffn_dim),
            nn.LayerNorm(ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
        )

        # Functional regularization token
        self.reg_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Fixed Scenario1b subnet mapping per host (0=Enterprise, 1=Operational, 2=User)
        host_subnets = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2] # 13 hosts
        self.register_buffer("host_subnet_ids", torch.tensor(host_subnets, dtype=torch.long))

        # Transformer Encoder
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

    def forward(
        self,
        flat_obs: torch.Tensor,
        host_known_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Input: flat_obs [B, T_hist, 52] or [B, T_hist, 13, 4]
        Output: host tokens [B, T_hist, 13, hidden_dim]
        """
        if flat_obs.dim() == 3:
            B, T_hist, D = flat_obs.shape
            x_hosts = flat_obs.view(B, T_hist, self.num_hosts, 4)
        else:
            B, T_hist, H, F = flat_obs.shape
            x_hosts = flat_obs
            D = H * F

        device = flat_obs.device

        # Extract activity & compromise indices from 4-dim per host
        # activity: 2 dims, compromise: 2 dims
        act_logits = x_hosts[:, :, :, :2].argmax(dim=-1) # [B, T_hist, 13]
        comp_logits = x_hosts[:, :, :, 2:].argmax(dim=-1) # [B, T_hist, 13]

        host_ids = torch.arange(self.num_hosts, device=device).unsqueeze(0).unsqueeze(0) # [1, 1, 13]
        subnet_ids = self.host_subnet_ids.unsqueeze(0).unsqueeze(0)                     # [1, 1, 13]

        e_host = self.host_id_emb(host_ids).expand(B, T_hist, -1, -1)
        e_sub = self.subnet_id_emb(subnet_ids).expand(B, T_hist, -1, -1)
        e_act = self.activity_emb(act_logits)
        e_comp = self.compromise_emb(comp_logits)

        if host_known_mask is not None:
            mask_long = host_known_mask.long()
            if mask_long.dim() == 2:
                mask_long = mask_long.unsqueeze(1).expand(-1, T_hist, -1)
            e_vis = self.visibility_emb(mask_long)
        else:
            e_vis = self.visibility_emb(torch.ones((B, T_hist, self.num_hosts), device=device, dtype=torch.long))

        # Concatenate embeddings and fuse with MLP
        cat_embs = torch.cat([e_host, e_sub, e_act, e_comp, e_vis], dim=-1) # [B, T_hist, 13, 5 * D]
        fused = self.fusion_mlp(cat_embs)                                  # [B, T_hist, 13, D]

        # Add relative time embeddings
        t_ids = torch.arange(T_hist, device=device).unsqueeze(0).unsqueeze(2) # [1, T_hist, 1]
        e_time = self.time_emb(t_ids)
        tokens = fused + e_time                                            # [B, T_hist, 13, D]

        # Reshape to sequence: [B, T_hist * 13, D]
        seq_tokens = tokens.view(B, T_hist * self.num_hosts, self.hidden_dim)

        # Append regularization token
        reg = self.reg_token.expand(B, -1, -1)                              # [B, 1, D]
        seq_tokens = torch.cat([seq_tokens, reg], dim=1)                   # [B, T_hist * 13 + 1, D]

        out = self.transformer(seq_tokens)
        out = self.norm(out)

        # Return host tokens: [B, T_hist, 13, hidden_dim]
        host_out = out[:, :-1, :].view(B, T_hist, self.num_hosts, self.hidden_dim)
        return host_out
