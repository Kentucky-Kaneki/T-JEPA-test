"""
T-JEPA: Tabular Joint-Embedding Predictive Architecture
Based on: https://arxiv.org/abs/2410.05016 (ICLR 2025)
"""

import math
import torch
import torch.nn as nn
import copy


# ─────────────────────────────────────────────
#  Feature Embedding Layer
# ─────────────────────────────────────────────

class FeatureEmbedding(nn.Module):
    """
    Per-feature linear projection + index embedding + type embedding.
    Numerical features: dim 1  →  h
    Categorical features: dim = cardinality  →  h  (after one-hot)
    """
    def __init__(self, feature_dims: list[int], hidden_dim: int):
        """
        feature_dims: list of length d.
            1   for numerical features
            >1  for categorical (cardinality)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feature_dims = feature_dims
        d = len(feature_dims)

        # One linear layer per feature (weights are NOT shared)
        self.projections = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in feature_dims
        ])
        # Learned index embedding (position of the feature in the table)
        self.index_emb = nn.Embedding(d, hidden_dim)
        # Learned type embedding: 0 = numerical, 1 = categorical
        self.type_emb  = nn.Embedding(2, hidden_dim)

        # REG token embedding
        self.reg_token = nn.Parameter(torch.randn(1, hidden_dim))

        self._init_weights()

    def _init_weights(self):
        for proj in self.projections:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    def embed_features(
        self,
        x_encoded: list[torch.Tensor],   # length d; each (B, e_j)
        feature_indices: list[int],       # which original features these are
        add_reg: bool = False,
    ) -> torch.Tensor:
        """
        Returns (B, L, h) where L = len(feature_indices) [+ 1 if add_reg].
        """
        B = x_encoded[0].shape[0]
        device = x_encoded[0].device
        tokens = []
        for k, (orig_idx, x_j) in enumerate(zip(feature_indices, x_encoded)):
            proj = self.projections[orig_idx](x_j)          # (B, h)
            proj = proj + self.index_emb(
                torch.tensor(orig_idx, device=device)
            )
            ftype = 0 if self.feature_dims[orig_idx] == 1 else 1
            proj = proj + self.type_emb(
                torch.tensor(ftype, device=device)
            )
            tokens.append(proj.unsqueeze(1))                # (B, 1, h)
        out = torch.cat(tokens, dim=1)                      # (B, L, h)
        if add_reg:
            reg = self.reg_token.unsqueeze(0).expand(B, -1, -1)  # (B, 1, h)
            out = torch.cat([out, reg], dim=1)              # (B, L+1, h)
        return out


# ─────────────────────────────────────────────
#  Transformer Encoder Block
# ─────────────────────────────────────────────

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,        # Pre-LN is more stable
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ─────────────────────────────────────────────
#  Predictor (lighter Transformer)
# ─────────────────────────────────────────────

class Predictor(nn.Module):
    """
    Receives context representation + mask tokens for target positions,
    outputs a predicted representation per target position.
    """
    def __init__(
        self,
        context_dim: int,
        pred_dim: int,
        num_heads: int,
        num_layers: int,
        num_features: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.down = nn.Linear(context_dim, pred_dim)
        self.up   = nn.Linear(pred_dim, context_dim)

        # Learnable mask token (one per feature index)
        self.mask_tokens = nn.Embedding(num_features, pred_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=pred_dim,
            nhead=num_heads,
            dim_feedforward=pred_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        h_context: torch.Tensor,   # (B, L_ctx, context_dim)
        target_indices: list[int],  # which features to predict
    ) -> torch.Tensor:
        """Returns (B, |target_indices|, context_dim)"""
        device = h_context.device
        B = h_context.shape[0]

        ctx = self.down(h_context)  # (B, L_ctx, pred_dim)

        # Build mask tokens for each target feature
        tgt_idx_t = torch.tensor(target_indices, device=device)
        mask_toks  = self.mask_tokens(tgt_idx_t)               # (T, pred_dim)
        mask_toks  = mask_toks.unsqueeze(0).expand(B, -1, -1)  # (B, T, pred_dim)

        # Concatenate context + mask tokens
        inp = torch.cat([ctx, mask_toks], dim=1)               # (B, L_ctx+T, pred_dim)
        out = self.encoder(inp)

        # Take only the mask-token positions
        pred = out[:, ctx.shape[1]:, :]                        # (B, T, pred_dim)
        return self.up(pred)                                   # (B, T, context_dim)


# ─────────────────────────────────────────────
#  T-JEPA Core
# ─────────────────────────────────────────────

class TJEPA(nn.Module):
    """
    T-JEPA: Tabular Joint-Embedding Predictive Architecture.

    Architecture:
      - FeatureEmbedding  (shared weights between context & target encoder input)
      - context_encoder   (fθ  — trained by gradient descent)
      - target_encoder    (fθ̄  — updated by EMA of context encoder)
      - predictor         (gϕ  — trained by gradient descent)

    Usage:
      loss = model.forward(x_encoded, feature_dims_info)
    """
    def __init__(
        self,
        feature_dims: list[int],
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.0,
        pred_dim: int = 32,
        pred_heads: int = 2,
        pred_layers: int = 2,
        pred_dropout: float = 0.0,
        ema_decay: float = 0.998,
        mask_min_ctx: float = 0.10,
        mask_max_ctx: float = 0.75,
        mask_min_tgt: float = 0.10,
        mask_max_tgt: float = 0.50,
        num_ctx_masks: int = 1,
        num_tgt_masks: int = 4,
        num_reg_tokens: int = 1,
    ):
        super().__init__()
        self.d = len(feature_dims)
        self.ema_decay = ema_decay
        self.mask_min_ctx  = mask_min_ctx
        self.mask_max_ctx  = mask_max_ctx
        self.mask_min_tgt  = mask_min_tgt
        self.mask_max_tgt  = mask_max_tgt
        self.num_ctx_masks = num_ctx_masks
        self.num_tgt_masks = num_tgt_masks
        self.num_reg_tokens = num_reg_tokens

        # Shared embedding layer
        self.embed = FeatureEmbedding(feature_dims, hidden_dim)

        # Context encoder (trained by grad)
        self.context_encoder = TransformerEncoder(
            hidden_dim, num_heads, num_layers, ffn_dim, dropout
        )

        # Target encoder (EMA copy — no grad)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        # Predictor
        self.predictor = Predictor(
            context_dim=hidden_dim,
            pred_dim=pred_dim,
            num_heads=pred_heads,
            num_layers=pred_layers,
            num_features=self.d,
            dropout=pred_dropout,
        )

    # ── EMA update ───────────────────────────
    @torch.no_grad()
    def update_target_encoder(self):
        for p_ctx, p_tgt in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            p_tgt.data.mul_(self.ema_decay).add_(
                p_ctx.data, alpha=1.0 - self.ema_decay
            )

    # ── Masking helpers ──────────────────────
    def _sample_mask(self, min_share: float, max_share: float) -> list[int]:
        """Returns list of UNMASKED feature indices."""
        share = min_share + torch.rand(1).item() * (max_share - min_share)
        n_keep = max(1, int(self.d * (1.0 - share)))
        return sorted(torch.randperm(self.d)[:n_keep].tolist())

    def _sample_mask_pair(self):
        """
        Sample one context mask and several non-overlapping target masks.
        Returns:
          ctx_indices  : list[int]   — unmasked features for context
          tgt_indices_list : list[list[int]]  — unmasked features per target mask
        """
        for _ in range(100):                      # retry until non-overlapping found
            ctx = self._sample_mask(self.mask_min_ctx, self.mask_max_ctx)
            ctx_set = set(ctx)
            tgts = []
            pool = [i for i in range(self.d) if i not in ctx_set]
            if len(pool) == 0:
                continue
            ok = True
            used = set()
            for _ in range(self.num_tgt_masks):
                share = (self.mask_min_tgt +
                         torch.rand(1).item() * (self.mask_max_tgt - self.mask_min_tgt))
                n_keep = max(1, int(len(pool) * (1.0 - share)))
                n_keep = min(n_keep, len(pool) - len(used))
                if n_keep <= 0:
                    ok = False; break
                available = [p for p in pool if p not in used]
                if len(available) < n_keep:
                    ok = False; break
                chosen = sorted(
                    torch.tensor(available)[
                        torch.randperm(len(available))[:n_keep]
                    ].tolist()
                )
                used.update(chosen)
                tgts.append(chosen)
            if ok and tgts:
                return ctx, tgts
        # fallback: use first half / second half
        half = self.d // 2
        return list(range(half)), [list(range(half, self.d))]

    # ── Forward ──────────────────────────────
    def forward(self, x_batch: list[torch.Tensor]) -> torch.Tensor:
        """
        x_batch: list of length d, each tensor (B, e_j)
        Returns scalar loss.
        """
        ctx_indices, tgt_indices_list = self._sample_mask_pair()

        # ── Target encoder (no grad, full unmasked input) ──
        with torch.no_grad():
            z_full = self.embed.embed_features(
                x_batch, list(range(self.d)), add_reg=True
            )
            h_target_full = self.target_encoder(z_full)   # (B, d+1, h)
            # Drop REG token (last position) — keep feature tokens only
            h_target_full = h_target_full[:, :-1, :]      # (B, d, h)

        # ── Context encoder ──────────────────────────────
        ctx_feats  = [x_batch[i] for i in ctx_indices]
        z_ctx = self.embed.embed_features(ctx_feats, ctx_indices, add_reg=True)
        h_ctx_full = self.context_encoder(z_ctx)           # (B, L_ctx+1, h)
        h_ctx = h_ctx_full[:, :-1, :]                      # drop REG  (B, L_ctx, h)

        # ── Predictor + loss ─────────────────────────────
        total_loss = torch.tensor(0.0, device=z_ctx.device)
        count = 0
        for tgt_idx in tgt_indices_list:
            h_pred = self.predictor(h_ctx, tgt_idx)        # (B, T, h)
            h_tgt  = h_target_full[:, tgt_idx, :]          # (B, T, h)
            loss = (h_pred - h_tgt).pow(2).mean()
            total_loss = total_loss + loss
            count += 1

        return total_loss / max(count, 1)

    # ── Encode (inference) ───────────────────
    @torch.no_grad()
    def encode(self, x_batch: list[torch.Tensor]) -> torch.Tensor:
        """
        Returns (B, d, h) feature representations from the context encoder
        using the full (unmasked) input — used for downstream tasks.
        """
        z = self.embed.embed_features(
            x_batch, list(range(self.d)), add_reg=False
        )
        return self.context_encoder(z)   # (B, d, h)
