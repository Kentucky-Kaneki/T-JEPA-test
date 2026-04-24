"""
T-JEPA: Tabular Joint-Embedding Predictive Architecture
Based on: https://arxiv.org/abs/2410.05016 (ICLR 2025)

Task 1 fixes:
  1. target_embed is a SEPARATE EMA-tracked embedding (previously shared, breaking EMA).
  2. L2 normalization of h_pred/h_tgt before MSE (anti-collapse).
  3. Variance regularization (VICReg-style) on context representations.
  4. Latent variable z_uncertainty in Predictor (information bottleneck, KL loss).

Task 2 additions:
  5. TemporalPredictor — action-conditioned next-state prediction in latent space.
     Action injected as a prepended token; maintains per-host token structure (no collapse).
  6. TJEPA.forward() accepts optional (a_batch, x_next) for the temporal JEPA path.
     Combined loss: L = L_masked + temporal_weight * L_future + kl_weight * KL_total
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


# ─────────────────────────────────────────────
#  Feature Embedding Layer
# ─────────────────────────────────────────────

class FeatureEmbedding(nn.Module):
    """
    Per-feature linear projection + index embedding + type embedding.

    For NASim host-level tokenization:
        feature_dims = [host_features] * num_hosts
        Each host is one "feature" with dim = host_features.
        The per-feature Linear projects (B, host_features) → (B, hidden_dim).

    For tabular data:
        Numerical: dim = 1
        Categorical: dim = cardinality (one-hot input)
    """
    def __init__(self, feature_dims: list[int], hidden_dim: int):
        super().__init__()
        self.hidden_dim   = hidden_dim
        self.feature_dims = feature_dims
        d = len(feature_dims)

        self.projections = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in feature_dims
        ])
        self.index_emb = nn.Embedding(d, hidden_dim)
        self.type_emb  = nn.Embedding(2, hidden_dim)  # 0=numerical, 1=categorical
        self.reg_token = nn.Parameter(torch.randn(1, hidden_dim))

        self._init_weights()

    def _init_weights(self):
        for proj in self.projections:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    def embed_features(
        self,
        x_encoded: list[torch.Tensor],  # length d; each (B, e_j)
        feature_indices: list[int],
        add_reg: bool = False,
    ) -> torch.Tensor:
        """Returns (B, L, h) where L = len(feature_indices) [+ 1 if add_reg]."""
        B      = x_encoded[0].shape[0]
        device = x_encoded[0].device
        tokens = []
        for orig_idx, x_j in zip(feature_indices, x_encoded):
            proj = self.projections[orig_idx](x_j)
            proj = proj + self.index_emb(torch.tensor(orig_idx, device=device))
            ftype = 0 if self.feature_dims[orig_idx] == 1 else 1
            proj = proj + self.type_emb(torch.tensor(ftype, device=device))
            tokens.append(proj.unsqueeze(1))
        out = torch.cat(tokens, dim=1)
        if add_reg:
            reg = self.reg_token.unsqueeze(0).expand(B, -1, -1)
            out = torch.cat([out, reg], dim=1)
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
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ─────────────────────────────────────────────
#  Spatial Predictor
# ─────────────────────────────────────────────

class Predictor(nn.Module):
    """
    Spatial JEPA predictor (Task 1).

    Given masked context h_ctx, predicts latent representations of specific
    masked positions using learnable mask tokens.

    Returns (pred, mu, logvar) each (B, |target_indices|, context_dim).
    Latent variable z_u ~ N(mu, exp(logvar)) added to deterministic prediction.
    KL(N(mu,σ²)||N(0,1)) is minimised to limit uncertainty information content.
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

        self.mu_head     = nn.Linear(pred_dim, context_dim)
        self.logvar_head = nn.Linear(pred_dim, context_dim)

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
        h_context: torch.Tensor,
        target_indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = h_context.device
        B = h_context.shape[0]

        ctx = self.down(h_context)

        tgt_idx_t = torch.tensor(target_indices, device=device)
        mask_toks = self.mask_tokens(tgt_idx_t).unsqueeze(0).expand(B, -1, -1)

        inp  = torch.cat([ctx, mask_toks], dim=1)
        out  = self.encoder(inp)
        pred_raw = out[:, ctx.shape[1]:, :]

        det_pred = self.up(pred_raw)

        mu     = self.mu_head(pred_raw)
        logvar = self.logvar_head(pred_raw)
        logvar = torch.clamp(logvar, -10.0, 2.0)

        if self.training:
            z_u = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        else:
            z_u = mu

        return det_pred + z_u, mu, logvar


# ─────────────────────────────────────────────
#  Temporal Predictor  (Task 2)
# ─────────────────────────────────────────────

class TemporalPredictor(nn.Module):
    """
    Action-conditioned temporal JEPA predictor (Task 2).

    Predicts z_{t+1} from masked context z_t and a discrete action a_t.
    Action is injected as an extra prepended token — the Transformer attends
    to it and learns to route action information to relevant host tokens.

    Sequence fed to Transformer:
        [ action_token | h_ctx_down | next_state_queries ]
          (B,1,P)         (B,L_ctx,P)   (B,d,P)

    Output: last d positions → (B, d, hidden_dim) — one prediction per host.
    Token-wise structure is preserved throughout (no collapse / pooling).

    Also outputs latent uncertainty (mu, logvar) using the same information-
    bottleneck design as the spatial Predictor (Task 1 §7).

    max 2 attention heads to stay hardware-feasible.
    """
    def __init__(
        self,
        context_dim: int,
        pred_dim: int,
        num_actions: int,
        num_hosts: int,       # d — one query token per host position
        num_heads: int = 2,   # keep low — hardware constraint
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_hosts = num_hosts

        # Action embedding directly to pred_dim (no separate projection needed)
        self.action_embed = nn.Embedding(num_actions, pred_dim)

        # Down-project context representations
        self.down = nn.Linear(context_dim, pred_dim)

        # Up-project predictions back to context_dim
        self.up = nn.Linear(pred_dim, context_dim)

        # Latent uncertainty heads (same pattern as spatial Predictor)
        self.mu_head     = nn.Linear(pred_dim, context_dim)
        self.logvar_head = nn.Linear(pred_dim, context_dim)

        # Learnable next-state query tokens — one per host position
        self.next_state_queries = nn.Embedding(num_hosts, pred_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=pred_dim,
            nhead=num_heads,         # max 2
            dim_feedforward=pred_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.action_embed.weight,       std=0.02)
        nn.init.normal_(self.next_state_queries.weight, std=0.02)

    def forward(
        self,
        h_ctx: torch.Tensor,    # (B, L_ctx, context_dim)
        actions: torch.Tensor,  # (B,) int64
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (pred, mu, logvar), each (B, num_hosts, context_dim).
        pred = deterministic prediction + sampled z_u.
        """
        B      = h_ctx.shape[0]
        device = h_ctx.device

        # 1. Action → extra token
        action_tok = self.action_embed(actions).unsqueeze(1)   # (B, 1, pred_dim)

        # 2. Down-project context representations
        ctx_down = self.down(h_ctx)                            # (B, L_ctx, pred_dim)

        # 3. Next-state query tokens (learnable per host)
        host_idx = torch.arange(self.num_hosts, device=device)
        queries  = self.next_state_queries(host_idx)           # (d, pred_dim)
        queries  = queries.unsqueeze(0).expand(B, -1, -1)      # (B, d, pred_dim)

        # 4. Concatenate: [action | context | queries]
        seq = torch.cat([action_tok, ctx_down, queries], dim=1) # (B, 1+L_ctx+d, pred_dim)

        # 5. Transformer encoder
        out = self.encoder(seq)                                 # (B, 1+L_ctx+d, pred_dim)

        # 6. Slice the last d (query) positions
        pred_raw = out[:, -self.num_hosts:, :]                 # (B, d, pred_dim)

        # 7. Deterministic prediction path
        det_pred = self.up(pred_raw)                           # (B, d, context_dim)

        # 8. Latent uncertainty variable
        mu     = self.mu_head(pred_raw)                        # (B, d, context_dim)
        logvar = self.logvar_head(pred_raw)                    # (B, d, context_dim)
        logvar = torch.clamp(logvar, -10.0, 2.0)

        if self.training:
            z_u = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        else:
            z_u = mu

        return det_pred + z_u, mu, logvar


# ─────────────────────────────────────────────
#  T-JEPA Core
# ─────────────────────────────────────────────

class TJEPA(nn.Module):
    """
    T-JEPA: Tabular / Temporal Joint-Embedding Predictive Architecture.

    Supports two training modes depending on what is passed to forward():

    (A) Spatial-only  [Task 1]:
        forward(x_batch)
        → L_masked only (masked feature prediction in latent space)

    (B) Spatial + Temporal  [Task 2]:
        forward(x_batch, a_batch, x_next)
        → L = L_masked + temporal_weight * L_future + kl_weight * KL_total
        where L_future = prediction of next-state latents from masked s_t + action

    Key design:
      - h_ctx from the context encoder is SHARED between both paths.
        One encoder forward → two loss terms. No redundant computation.
      - Target branch (target_embed + target_encoder) remains fully EMA-tracked.
      - All collapse guarantees from Task 1 apply to both loss paths.
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
        normalize: bool = True,
        var_reg_weight: float = 0.04,
        kl_weight: float = 0.01,
        # ── Task 2 params ────────────────────────────────────────
        num_actions: int = 0,             # 0 = spatial-only mode
        temporal_pred_layers: int = 2,
        temporal_weight: float = 0.5,     # λ in L = L_masked + λ * L_future
    ):
        super().__init__()
        self.d              = len(feature_dims)
        self.ema_decay      = ema_decay
        self.mask_min_ctx   = mask_min_ctx
        self.mask_max_ctx   = mask_max_ctx
        self.mask_min_tgt   = mask_min_tgt
        self.mask_max_tgt   = mask_max_tgt
        self.num_ctx_masks  = num_ctx_masks
        self.num_tgt_masks  = num_tgt_masks
        self.num_reg_tokens = num_reg_tokens
        self.normalize      = normalize
        self.var_reg_weight = var_reg_weight
        self.kl_weight      = kl_weight
        self.temporal_weight = temporal_weight

        # ── Context branch (gradient-trained) ────────────────────
        self.embed = FeatureEmbedding(feature_dims, hidden_dim)
        self.context_encoder = TransformerEncoder(
            hidden_dim, num_heads, num_layers, ffn_dim, dropout
        )

        # ── Target branch (EMA — no grad per step) ───────────────
        self.target_embed = copy.deepcopy(self.embed)
        for p in self.target_embed.parameters():
            p.requires_grad_(False)

        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        # ── Spatial Predictor (Task 1) ────────────────────────────
        self.predictor = Predictor(
            context_dim=hidden_dim,
            pred_dim=pred_dim,
            num_heads=pred_heads,
            num_layers=pred_layers,
            num_features=self.d,
            dropout=pred_dropout,
        )

        # ── Temporal Predictor (Task 2 — optional) ───────────────
        if num_actions > 0:
            self.temporal_predictor = TemporalPredictor(
                context_dim=hidden_dim,
                pred_dim=pred_dim,
                num_actions=num_actions,
                num_hosts=self.d,
                num_heads=2,               # hardware-safe max
                num_layers=temporal_pred_layers,
                dropout=pred_dropout,
            )
        else:
            self.temporal_predictor = None

    # ── EMA update ────────────────────────────────────────────────
    @torch.no_grad()
    def update_target_encoder(self):
        """
        EMA-update the full target branch: both embedding and TransformerEncoder.
        ξ ← m·ξ + (1-m)·θ  for every parameter in {target_embed, target_encoder}
        """
        m = self.ema_decay
        for p_ctx, p_tgt in zip(self.embed.parameters(), self.target_embed.parameters()):
            p_tgt.data.mul_(m).add_(p_ctx.data, alpha=1.0 - m)
        for p_ctx, p_tgt in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            p_tgt.data.mul_(m).add_(p_ctx.data, alpha=1.0 - m)

    # ── Masking helpers ───────────────────────────────────────────
    def _sample_mask(self, min_share: float, max_share: float) -> list[int]:
        share  = min_share + torch.rand(1).item() * (max_share - min_share)
        n_keep = max(1, int(self.d * (1.0 - share)))
        return sorted(torch.randperm(self.d)[:n_keep].tolist())

    def _sample_mask_pair(self):
        for _ in range(100):
            ctx = self._sample_mask(self.mask_min_ctx, self.mask_max_ctx)
            ctx_set = set(ctx)
            pool = [i for i in range(self.d) if i not in ctx_set]
            if len(pool) == 0:
                continue
            ok, used, tgts = True, set(), []
            for _ in range(self.num_tgt_masks):
                share  = self.mask_min_tgt + torch.rand(1).item() * (self.mask_max_tgt - self.mask_min_tgt)
                n_keep = max(1, int(len(pool) * (1.0 - share)))
                n_keep = min(n_keep, len(pool) - len(used))
                if n_keep <= 0:
                    ok = False; break
                available = [p for p in pool if p not in used]
                if len(available) < n_keep:
                    ok = False; break
                chosen = sorted(
                    torch.tensor(available)[torch.randperm(len(available))[:n_keep]].tolist()
                )
                used.update(chosen)
                tgts.append(chosen)
            if ok and tgts:
                return ctx, tgts
        half = self.d // 2
        return list(range(half)), [list(range(half, self.d))]

    # ── Forward ──────────────────────────────────────────────────
    def forward(
        self,
        x_batch: list[torch.Tensor],                      # current state (masked)
        a_batch: torch.Tensor | None = None,              # (B,) int64 actions
        x_next:  list[torch.Tensor] | None = None,        # full next state
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            x_batch : list of d tensors (B, feature_dim_j) — current state
            a_batch : (B,) int64 — discrete actions  [Task 2, optional]
            x_next  : list of d tensors (B, feature_dim_j) — next state  [Task 2, optional]

        Returns:
            total_loss : scalar tensor
            stats      : dict — pred_loss, var_loss, kl_loss, temporal_loss,
                                kl_temporal, repr_std
        """
        ctx_indices, tgt_indices_list = self._sample_mask_pair()

        # ── Target branch: encode full current state (no grad) ────
        with torch.no_grad():
            z_full = self.target_embed.embed_features(
                x_batch, list(range(self.d)), add_reg=True
            )
            h_target_full = self.target_encoder(z_full)[:, :-1, :]   # drop REG → (B, d, h)

        # ── Context branch: encode masked current state ───────────
        ctx_feats  = [x_batch[i] for i in ctx_indices]
        z_ctx = self.embed.embed_features(ctx_feats, ctx_indices, add_reg=True)
        h_ctx_full = self.context_encoder(z_ctx)
        h_ctx = h_ctx_full[:, :-1, :]                                 # drop REG → (B, L_ctx, h)

        # ── Variance regularization ───────────────────────────────
        h_flat      = h_ctx.reshape(-1, h_ctx.shape[-1])
        std_per_dim = h_flat.std(dim=0)
        repr_std    = std_per_dim.detach().mean().item()
        var_loss    = F.relu(1.0 - std_per_dim).mean()

        # ── Spatial JEPA loss (L_masked) ─────────────────────────
        pred_loss = torch.tensor(0.0, device=h_ctx.device)
        kl_loss   = torch.tensor(0.0, device=h_ctx.device)
        count = 0

        for tgt_idx in tgt_indices_list:
            h_pred, mu, logvar = self.predictor(h_ctx, tgt_idx)
            h_tgt = h_target_full[:, tgt_idx, :]

            if self.normalize:
                h_pred = F.normalize(h_pred, dim=-1)
                h_tgt  = F.normalize(h_tgt,  dim=-1)

            pred_loss = pred_loss + (h_pred - h_tgt).pow(2).mean()
            kl_loss   = kl_loss   + (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())).mean()
            count += 1

        n = max(count, 1)
        pred_loss = pred_loss / n
        kl_loss   = kl_loss   / n

        # ── Temporal JEPA loss (L_future) — Task 2 ───────────────
        temporal_loss = torch.tensor(0.0, device=h_ctx.device)
        kl_temporal   = torch.tensor(0.0, device=h_ctx.device)

        if (self.temporal_predictor is not None
                and a_batch is not None
                and x_next is not None):

            # Encode full next state with EMA target branch (no grad)
            with torch.no_grad():
                z_next = self.target_embed.embed_features(
                    x_next, list(range(self.d)), add_reg=True
                )
                h_t1 = self.target_encoder(z_next)[:, :-1, :]        # (B, d, h)

            # Temporal prediction from masked context + action
            z_pred, mu_t, logvar_t = self.temporal_predictor(h_ctx, a_batch)  # (B, d, h)

            if self.normalize:
                z_pred = F.normalize(z_pred, dim=-1)
                h_t1   = F.normalize(h_t1,   dim=-1)

            temporal_loss = (z_pred - h_t1).pow(2).mean()
            kl_temporal   = (-0.5 * (1.0 + logvar_t - mu_t.pow(2) - logvar_t.exp())).mean()

        # ── Combined loss ─────────────────────────────────────────
        total_loss = (
            pred_loss
            + self.var_reg_weight * var_loss
            + self.kl_weight * (kl_loss + kl_temporal)
            + self.temporal_weight * temporal_loss
        )

        stats = {
            "pred_loss":     pred_loss.item(),
            "var_loss":      var_loss.item(),
            "kl_loss":       kl_loss.item(),
            "temporal_loss": temporal_loss.item(),
            "kl_temporal":   kl_temporal.item(),
            "repr_std":      repr_std,
        }

        return total_loss, stats

    # ── Encode (inference) ────────────────────────────────────────
    @torch.no_grad()
    def encode(self, x_batch: list[torch.Tensor]) -> torch.Tensor:
        """
        Returns (B, d, h) representations from the context encoder.
        Full (unmasked) input. Used for downstream tasks.
        """
        z = self.embed.embed_features(x_batch, list(range(self.d)), add_reg=False)
        return self.context_encoder(z)
