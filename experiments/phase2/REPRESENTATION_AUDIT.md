# Representation Fairness Audit

## Overview & Scope
This audit inspects the exact tensor transformations, parameter counts, context aggregation mechanisms, and temporal encodings across all candidate representations (`flat`, `feature`, `host`, `hierarchical`) in the Cyber-JEPA codebase.

---

## 1. Information Equivalence Audit

All four representations consume the same underlying CybORG Scenario1b observation vector of length **52**.

| Property | Flat (`flat`) | Feature (`feature`) | Host (`host`) | Hierarchical (`hierarchical`) |
|---|---|---|---|---|
| **Raw Input Dim** | 52 | 52 | 52 (or 13×4) | 52 (or 13×4) |
| **Tokens per step** | 1 (per step) | 52 | 13 | 13 host + 3 subnet + 1 global |
| **Token Dim ($D$)** | 64 | 64 | 64 | 64 |
| **Total Context Tokens** | $4 \times 1 + 1 = 5$ | $4 \times 52 + 1 = 209$ | $4 \times 13 + 1 = 53$ | $4 \times 13 + 1 = 53$ (stage 1) |
| **Temporal History Length** | $T=4$ | $T=4$ | $T=4$ | $T=4$ |
| **Information Discarded in Preprocessing** | None | None | None | None |
| **Underlying Semantic Value Equality** | Equivalent | Equivalent | Equivalent | Equivalent |

---

## 2. Parameter Budget & Disparity Analysis

Parameter counts were audited at `hidden_dim = 64`, `ffn_dim = 256`, `num_layers = 3`, `num_heads = 4`:

| Representation | Encoder Params | Full JEPA Model Params | Ratio vs. Flat Encoder |
|---|---:|---:|---:|
| **`flat`** | **180,928** | **540,672** | **1.00×** |
| **`feature`** | **154,816** | **488,448** | **0.85×** |
| **`host`** | **251,328** | **681,472** | **1.39×** |
| **`hierarchical`** | **318,464** | **815,744** | **1.76×** |

### Fairness Finding 1: Parameter Asymmetry
- `hierarchical` possesses **1.76× more parameters** than `flat` (318,464 vs 180,928), yet performed worst in Phase 1 (F1=0.4279 vs 0.6722).
- `feature` has **15% fewer parameters** than `flat` (154,816 vs 180,928).
- *Implication*: `flat`'s Phase 1 victory occurred despite having fewer parameters than `host` and `hierarchical`. Capacity rescue (Experiment 2D) will test if `feature`, `host`, and `hierarchical` recover when given matched or expanded capacity (2x and 4x hidden dimension).

---

## 3. Context Aggregation & Interface Disparity

Each representation encodes context $O_{t-h+1:t}$ into latent representations. However, the JEPA predictor interface `g_phi` expects a single vector $z_t \in \mathbb{R}^{B \times D}$:

```text
raw CybORG observation [B, T=4, 52]
        ↓
Encoder forward pass
        ↓
flat         → out[:, -1, :]              [B, 64]  (Reg token)
feature      → out[:, -1, :, :].mean(1)   [B, 64]  (Mean-pooled over 52 tokens at t)
host         → out[:, -1, :, :].mean(1)   [B, 64]  (Mean-pooled over 13 tokens at t)
hierarchical → global_token               [B, 64]  (Stage-2 global token)
        ↓
Predictor input z_t                       [B, 64]
```

### Fairness Finding 2: Context Token Mean-Pooling
- `feature` and `host` token encoders produce fine-grained token tensors (`[B, T, 52, 64]` and `[B, T, 13, 64]`), but `CyberJEPA.forward()` **mean-pools across tokens at the last timestep** to form $z_t \in \mathbb{R}^{B \times 64}$.
- *Implication*: The structured token information is collapsed via unweighted mean pooling before reaching the predictor. `flat` uses a learned reg token that attends across all timesteps via Transformer self-attention.

---

## 4. Target Encoder Input Construction

In `jepa.py` (lines 93–96):
```python
if target_obs.dim() == 2:
    hist_len = getattr(self.online_encoder, "history_len", 4)
    target_in = target_obs.unsqueeze(1).expand(-1, hist_len, -1)
```

### Fairness Finding 3: Target Frame Expansion
- The target encoder $f_{\bar{\theta}}$ expects a sequence of length $T=4$. To encode the single target observation $O_{t+k}^{Blue} \in \mathbb{R}^{B \times 52}$, the tensor is expanded to `[B, 4, 52]` by repeating $O_{t+k}$ four times.
- *Implication*: The target encoder sees artificial static history $[O_{t+k}, O_{t+k}, O_{t+k}, O_{t+k}]$. This treatment is identical across all candidate representations.

---

## 5. Temporal & Feature Order Representations

- **Temporal Order Encoding**: `time_emb(t)` (learned embedding for timesteps $0, 1, 2, 3$) is added to all representations.
- **Feature Identity Encoding**: `flat` relies on linear projection row index; `feature` uses `feature_index_emb(0..51)`; `host` uses host/subnet/activity/compromise categorical embeddings; `hierarchical` uses host + two-stage cross-attention.

---

## Conclusion & Experimental Action Plan
1. Preserve representations as defined (do not alter architectural code per Phase 2 rules).
2. Execute Controlled Ablations 2A (temporal & feature shuffling), 2B (history lengths), 2C (current-only control), and 2D (capacity rescue at 2x and 4x hidden dim).
3. Evaluate whether `flat`'s superiority is due to temporal context, feature ordering, or parameter efficiency.
