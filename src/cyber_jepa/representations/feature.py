"""
Feature-Token Representation for Cyber-JEPA.

Encodes 52 observation features as distinct semantic tokens combining value projection,
feature index, host identity, feature type, visibility, and relative time embeddings.
"""

import torch
import torch.nn as nn


class FeatureTokenRepresentation(nn.Module):
    """Feature-level tokenization and Transformer encoder."""

    def __init__(
        self,
        num_features: int = 52,
        num_hosts: int = 13,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        ffn_dim: int = 256,
        history_len: int = 4,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_hosts = num_hosts
        self.hidden_dim = hidden_dim
        self.history_len = history_len

        # Feature value projection
        self.val_proj = nn.Linear(1, hidden_dim)

        # Learned index and host embeddings
        self.feature_index_emb = nn.Embedding(num_features, hidden_dim)
        self.host_index_emb = nn.Embedding(num_hosts, hidden_dim)
        self.type_emb = nn.Embedding(2, hidden_dim) # 0=numerical, 1=categorical
        self.time_emb = nn.Embedding(history_len, hidden_dim)

        # Functional regularization token
        self.reg_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Feature-to-host mapping (4 features per host in Scenario1b vector)
        host_ids = []
        for i in range(num_features):
            host_ids.append(min(i // 4, num_hosts - 1))
        self.register_buffer("feature_host_ids", torch.tensor(host_ids, dtype=torch.long))

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: x of shape [B, T_hist, 52]
        Output: feature tokens of shape [B, T_hist, 52, hidden_dim]
        """
        B, T_hist, N_feat = x.shape
        device = x.device

        x_expanded = x.unsqueeze(-1) # [B, T_hist, 52, 1]
        val_tokens = self.val_proj(x_expanded) # [B, T_hist, 52, hidden_dim]

        # Add feature index and host index embeddings
        feat_ids = torch.arange(N_feat, device=device)
        host_ids = self.feature_host_ids[:N_feat]

        f_emb = self.feature_index_emb(feat_ids).unsqueeze(0).unsqueeze(0) # [1, 1, 52, D]
        h_emb = self.host_index_emb(host_ids).unsqueeze(0).unsqueeze(0)    # [1, 1, 52, D]

        tokens = val_tokens + f_emb + h_emb

        # Add relative time embeddings
        t_ids = torch.arange(T_hist, device=device)
        t_emb = self.time_emb(t_ids).unsqueeze(0).unsqueeze(2)             # [1, T_hist, 1, D]
        tokens = tokens + t_emb                                            # [B, T_hist, 52, D]

        # Reshape to sequence for Transformer: [B, T_hist * 52, D]
        flat_tokens = tokens.view(B, T_hist * N_feat, self.hidden_dim)

        # Append regularization token
        reg = self.reg_token.expand(B, -1, -1)                              # [B, 1, D]
        seq_tokens = torch.cat([flat_tokens, reg], dim=1)                  # [B, T_hist * 52 + 1, D]

        out = self.transformer(seq_tokens)
        out = self.norm(out)

        # Unflatten feature tokens: [B, T_hist, 52, hidden_dim]
        feat_out = out[:, :-1, :].view(B, T_hist, N_feat, self.hidden_dim)
        return feat_out
