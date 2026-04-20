"""
Downstream classifiers trained on top of T-JEPA representations.
Includes a simple MLP and a projection layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanPoolProjection(nn.Module):
    """
    Collapses (B, d, h)  →  (B, h) via mean pooling,
    then applies a linear projection.
    """
    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=1)          # (B, h)
        return self.proj(pooled)        # (B, out_dim)


class LinearFlattenProjection(nn.Module):
    """
    Flattens (B, d, h)  →  (B, d*h) then projects to out_dim.
    """
    def __init__(self, d: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(d * hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return self.proj(x.reshape(B, -1))


class DownstreamMLP(nn.Module):
    """
    MLP classifier on top of T-JEPA representations.
    Input: (B, d, h)  →  num_classes logits.
    """
    def __init__(
        self,
        d: int,
        hidden_dim: int,
        num_classes: int,
        mlp_hidden: int = 256,
        num_layers: int = 4,
        dropout: float = 0.3,
        projection: str = "mean",   # "mean" or "flatten"
    ):
        super().__init__()

        if projection == "flatten":
            proj_out = min(mlp_hidden, d * hidden_dim)
            self.proj = LinearFlattenProjection(d, hidden_dim, proj_out)
            in_dim = proj_out
        else:
            self.proj = MeanPoolProjection(hidden_dim, mlp_hidden)
            in_dim = mlp_hidden

        layers = []
        cur = in_dim
        for _ in range(num_layers - 1):
            layers += [
                nn.Linear(cur, mlp_hidden),
                nn.BatchNorm1d(mlp_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            cur = mlp_hidden
        layers.append(nn.Linear(cur, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = self.proj(h)
        return self.mlp(x)
