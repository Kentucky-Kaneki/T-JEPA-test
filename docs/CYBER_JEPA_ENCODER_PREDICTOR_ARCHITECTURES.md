# Cyber-JEPA Encoder & Predictor Architectures

Detailed reference for every neural component in the project: the two Context
Encoders (`flat`, `feature`), the Target Encoder (EMA copy of whichever Context
Encoder is in use), the separate `ActionEncoder` + `LatentPredictor` (Test 1 / Phase 3,
frozen baseline), and the fused encoders + `FusedLatentReadout` (Test 2 / Phase 4).
All shapes below use the project's default hyperparameters unless noted:

| Symbol | Default | Meaning |
|---|---|---|
| `B` | — | Batch size |
| `obs_dim` | 52 | Raw ChallengeWrapper feature vector width ($13\text{ hosts} \times 4\text{ features}$) |
| `T_hist` | 4 | History length (timesteps fed to the context encoder) |
| `K` | $\le 16$ (Phase 4 uses 8) | Action horizon length (`max_horizon = 16` default spec; `8` in Phase 4 runs) |
| `D` (`hidden_dim`) | 64 | Shared latent width every encoder/predictor projects into |
| `H` (`num_heads`) | 4 | Transformer attention heads |
| `ffn_dim` | 256 | Transformer feed-forward width |
| `num_layers` | 3 (Test 1) / 10 (Test 2) | Transformer depth |

Every Transformer layer in the project uses the same recipe: **Pre-LN**
(`norm_first=True`), GELU activation, dropout 0.1, `batch_first=True`. Pre-LN is used
throughout for training stability at this depth (normalizes before attention/FFN
rather than after).

---

## 1. Context Encoder — Flat (`FlatVectorRepresentation`)

**Purpose:** summarize $O_{0:t}$ (raw 52-dim ChallengeWrapper vectors over $T_{\text{hist}}$
steps) into a single global latent + per-timestep tokens, with no attack/defense
labeling — pure observation-history dynamics.

**Input layer**
```
x: [B, T_hist, 52]
  -> Linear(52, 256) -> LayerNorm(256) -> GELU -> Linear(256, 64)
  = input_proj(x): [B, T_hist, 64]
```
A 2-layer MLP with a `LayerNorm` in between projects each timestep's raw 52-dim
vector independently into the shared $D=64$ latent space (weights shared across
timesteps — this is a per-timestep, not per-sequence, projection).

**Positional structure**
```
time_emb = Embedding(T_hist=4, 64)                # one learned vector per history position
tokens = input_proj(x) + time_emb(0..T_hist-1)    # [B, T_hist, 64]
```

**Regularization/global token**
```
reg_token: Parameter [1, 1, 64], initialized N(0, 0.02^2)
seq = concat([tokens, reg_token.expand(B,-1,-1)], dim=1)   # [B, T_hist+1, 64]
```
A single learned token appended to the sequence — this is what the Transformer
uses to accumulate a global summary (functions like a CLS token, called "reg
token" in the code, for representation-collapse regularization purposes).

**Transformer stack**
```
TransformerEncoder(
    TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=256,
                             dropout=0.1, activation="gelu",
                             batch_first=True, norm_first=True),
    num_layers=3
)
out = transformer(seq)          # [B, T_hist+1, 64]
out = LayerNorm(64)(out)
```
Full (unmasked) bidirectional self-attention over all $T_{\text{hist}}+1$ tokens — every
history step and the reg token can attend to every other.

**Output layer**
```
global_latent = out[:, -1, :]     # [B, 64]  <- reg token's position after attention
flat_tokens   = out[:, :-1, :]    # [B, T_hist, 64]
```
Wrapped in a `ContextTokens` dataclass: `tokens=flat_tokens`, `global_token=global_latent`,
`token_type_ids` all set to `TokenType.TEMPORAL_FLAT`, `time_ids=0..T_hist-1`,
`padding_mask` all-`False` (no padding used), `entity_ids` all-zero (flat tokens
aren't tied to a specific host/entity).

**Parameters** (measured, default hyperparameters): **180,928** trainable —
`input_proj` (52→256→64 MLP + LayerNorm: 30,144) + `time_emb` (4×64: 256) + `reg_token` (64) +
3-layer Transformer encoder (150,336) + final `LayerNorm` (128).

---

## 2. Context Encoder — Feature Token (`FeatureTokenRepresentation`)

> **Reconstruction note:** this module is rebuilt from the project's spec docs
> (`evaluation_procedure.md`). The 52 ChallengeWrapper features are mapped as
> 13 hosts $\times$ 4 features/host (`feature_idx // 4` $\to$ host, `feature_idx % 4` $\to$
> intra-host slot). Host indices follow the 1..13 (0=NONE) convention matching
> `ActionEncoder`, enabling shared embeddings in fused configurations.

**Purpose:** the structured alternative to flat — instead of one token per
timestep, every one of the 52 raw features gets its **own** token, preserving
per-host, per-feature structure instead of collapsing it into an MLP projection.

**Input layer**
```
x: [B, T_hist, 52]
values = x.reshape(B, T_hist*52, 1)
val_tok = Linear(1, 64)(values)          # [B, T_hist*52, 64]   scalar -> D projection
```
Each individual feature *value* (a scalar) is projected independently into $D=64$
— contrast with flat's projection, which mixes all 52 features together per timestep.

**Per-token semantic embeddings** (added, not concatenated, to `val_tok`)
```
feature_index_emb : Embedding(52, 64)         # which of the 52 raw features (3,328)
host_index_emb    : Embedding(14, 64)         # 0=NONE, 1..13=host slot (896)
feature_type_emb  : Embedding(4, 64)          # intra-host feature slot 0..3 (256)
time_emb          : Embedding(T_hist=4, 64)   # which history timestep (256)

token = val_tok + feature_index_emb + host_index_emb + feature_type_emb + time_emb
      : [B, T_hist, 52, 64] -> reshape -> [B, T_hist*52, 64]
```
An **exactness assertion** enforces `tokens.shape[1] == T_hist * 52` (52 features
must map to exactly 52 tokens per timestep, every timestep — guards against a
tokenization bug silently dropping/duplicating features).

**Regularization/global token:** identical mechanism to the flat encoder — one
learned `[1,1,64]` token appended: `seq = [T_hist*52 feature tokens] + [1 reg token]`.

**Transformer stack:** identical recipe to the flat encoder (`d_model=64, nhead=4,
ffn=256, layers=3, Pre-LN, GELU`), but now attending over a much longer sequence:
$T_{\text{hist}} \times 52 + 1 = 209$ tokens instead of $5$. This is the structural cost of the
feature-token representation — same depth/width, $\approx 40\times$ longer sequence.

**Output layer:** same split as flat — `global_token = out[:, -1, :]`,
`tokens = out[:, :-1, :]` (shape `[B, 208, 64]`), `token_type_ids` all
`TokenType.FEATURE`, `entity_ids` = each token's host id (1..13), `time_ids` =
each token's timestep.

**Parameters** (measured, default hyperparameters): **155,008** trainable — fewer
than flat despite the longer sequence, because there's no 52→256→64 MLP
(`input_proj`); per-feature embeddings are cheaper than a shared wide MLP.

---

## 3. Target Encoder (EMA copy — both representations)

**Not a distinct architecture.** The Target Encoder is a `copy.deepcopy()` of
whichever Context Encoder is in use (Flat or Feature), with `requires_grad=False`
on every parameter and `.eval()` mode fixed. It is never touched by
backpropagation from the JEPA loss.

**Update rule** (called once per training step, outside the loss computation):
```
theta_target <- m * theta_target + (1 - m) * theta_online
m schedule:   m_init = 0.996  ->  m_final = 1.000, linear over training steps
buffers (e.g. LayerNorm running stats, if any): hard-copied, not EMA'd
```

**Forward pass — single-frame ($T=1$), not the $T_{\text{hist}}$ history:**
```
target_obs: [B, 52]  ->  unsqueeze(1)  ->  [B, 1, 52]
target_ctx = target_encoder.encode_context(target_single, return_context_tokens=True)
target_latent = target_ctx.global_token     # [B, 64]
```
The target branch **only ever sees $O_{t+1}$ alone** — never the history, never
the action. This is what the architecture doc calls the "neutral prediction
target": it's not told what led to this observation, just encodes it as-is. Both
encoder classes handle $T_{\text{hist}}=1$ correctly. In the fused encoders (Test 2),
calling `encode_context(x, actions=None)` skips action-token construction entirely,
executing the neutral forward pass through the deep stack.

**Parameters:** identical count to the online Context Encoder it copies
(180,928 for flat, 155,008 for feature) — but **non-trainable**, so excluded from
`total_trainable` and reported separately as `total_non_trainable_ema`.

---

## 4. Action Encoder (`ActionEncoder`, Test 1 — separate module)

**Purpose:** turn a sequence of $K$ discrete Blue action indices into $K$
semantic action tokens, independent of any context. This is Test 1's dedicated
module — the component Test 2 fuses into the encoder.

**Input layer**
```
actions: [B, K]   discrete indices in [0, 65]  (66 possible Scenario1b actions)
```

**Semantic field resolution** — looked up in three static buffers (built once at
construction, `register_buffer`, not learned):
```
type_map   [66] -> action type id   (0=Sleep/Misinform, 1=Monitor, 2=Analyse, 3=Remove, 4=Restore)
host_map   [66] -> target host id   (0=NONE, 1..13=host slot)
subnet_map [66] -> target subnet id (currently always 0 — reserved, not populated for Scenario1b)
```
> **Known quirk preserved:** Misinform/Decoy actions currently share type-id `0`
> with Sleep in this table — preserved across Test 1 and Test 2 for consistency.

**Embedding layers**
```
type_emb   : Embedding(16, 64)              # 16-way action-type vocabulary (5 used: 1,024)
host_emb   : Embedding(14, 64)              # 0=NONE, 1..13=host (896)
subnet_emb : Embedding(4, 64)               # 0=NONE, 1..3=subnet (unused, reserved: 256)
pos_emb    : Embedding(max_horizon, 64)     # position within horizon (16*64=1,024 at K=16; 8*64=512 at K=8)
valid_emb  : Embedding(2, 64)               # action validity flag (128)
```

**Combination + projection**
```
token = type_emb[type_ids] + host_emb[host_ids] + subnet_emb[subnet_ids]
      + pos_emb[0..K-1] + valid_emb[valid_ids]           # [B, K, 64], summed (not concatenated)

proj = Linear(64,64) -> GELU -> Linear(64,64)           # 8,320 params
action_tokens = proj(token)                                # [B, K, 64]
```

**Output layer:** `[B, K, 64]` — one token per horizon step, ready to be
concatenated/cross-attended against context inside `LatentPredictor`.

**Parameters** (measured):
- **At default `max_horizon=16`**: **11,648** trainable (`pos_emb` has 16 rows: 1,024).
- **At Phase 4 `max_horizon=8`**: **11,136** trainable (`pos_emb` has 8 rows: 512).

---

## 5. Predictor (`LatentPredictor`, Test 1)

**Purpose:** combine the context representation ($C_t$, from the Context Encoder,
via whichever aggregator is configured) with the action tokens (from
`ActionEncoder`, owned internally by `LatentPredictor`) to produce a predicted
future latent $\hat{Z}_{t+k}$ — the only place context and action actually meet in Test 1.

**Owns internally:** one `ActionEncoder` instance (see §4), plus:
```
granularity_emb : Embedding(4, 64)               # 0=feature, 1=host, 2=subnet, 3=network (256)
entity_emb      : Embedding(16, 64)              # which entity within that granularity (1,024)
horizon_emb     : Embedding(max_horizon + 1, 64) # horizon step k (17*64=1,088 at K=16; 9*64=576 at K=8)
```
These implement the `TargetSpec(horizon, granularity, target_id)` query interface
— i.e. "give me the prediction for host 3's compromise status at horizon $k=4$."

**Two Transformer sub-modules, both always instantiated** (only one is used per
forward call, depending on the shape of the incoming context):

**(a) `TransformerEncoder` — "standard vector prediction" path**, used when the
context arrives as a single pooled vector `[B, D]` (i.e. the `flat` representation
via `LegacyLastStepMean` aggregation):
```
ctx_token = z_t.unsqueeze(1)                       # [B, 1, 64]
seq = concat([ctx_token, action_tokens], dim=1)     # [B, 1+K, 64]
causal_mask: upper-triangular, -inf above diagonal  # position i cannot see position >i
out = TransformerEncoder(seq, mask=causal_mask)     # 3 layers, d=64, h=4, ffn=256 (150,336)
pred_latents = Linear(64,64)(LayerNorm(out[:, 1:, :]))   # [B, K, 64] — one prediction per horizon step
```
The causal mask ensures the prediction for horizon step $k$ can only attend to the
context token and action tokens $1..k$ — it cannot "see" later planned actions
when predicting an earlier step.

**(b) `TransformerDecoder` — "token-preserving cross-attention" path**, used when
the context arrives as full token memory `[B, L, D]` (i.e. the `feature_token_predictor`
representation via `TokenPreservingAggregator`, $L = T_{\text{hist}} \times 52 = 208$):
```
target_query = granularity_emb + entity_emb + horizon_emb   # [1, 1, 64], the TargetSpec query
tgt_seq = concat([target_query, action_tokens], dim=1)       # [B, 1+K, 64]
decoded = TransformerDecoder(tgt=tgt_seq, memory=z_t_ctx,
                              memory_key_padding_mask=padding_mask)   # 3 layers (200,192)
pred_latent = Linear(64,64)(LayerNorm(decoded[:, 0, :]))      # [B, 64] — single prediction
```
Here the query+action sequence cross-attends into the **full** $L=208$ token
memory (every feature token from every history step) rather than a single pooled
vector — the mechanism by which the feature-token representation avoids an early
pooling bottleneck.

**Output layer:** either `[B, K, 64]` (vector-path, all horizons at once) or
`[B, 64]` (token-preserving path, a specific `TargetSpec`-queried horizon), plus a
`+ granularity_emb + entity_emb` addition when a `TargetSpec` is supplied.

**Parameters** (`predictor_body` = total `LatentPredictor` params minus internal `ActionEncoder`):
- **At default `max_horizon=16`**: **356,864** trainable (total `LatentPredictor` = **368,512**).
- **At Phase 4 `max_horizon=8`**: **356,352** trainable (total `LatentPredictor` = **367,488**).
Both the `TransformerEncoder` and `TransformerDecoder` are always built regardless
of which path is fired at runtime.

---

## 6. Fused Encoders (Test 2 — `FlatFusedRepresentation` / `FeatureFusedRepresentation`)

**Purpose:** everything §1/§2 (context tokenization) and §4/§5 (action embedding +
combination) do, but as **one** encoder and **one** Transformer stack — action
tokens are built and appended to the *same* sequence the state tokens live in,
instead of being produced by a separate module and combined downstream.

**State-side input layer:** identical to §1 (flat) or §2 (feature) — same
`input_proj` / per-feature tokenization, same reg token.

**Shared semantic fields** (`SharedEntityEmbeddings`, owned once per fused
encoder instance):
```
host_emb   : Embedding(14, 64)   # used by BOTH state tokenizer (feature-fused) and action builder
subnet_emb : Embedding(4, 64)    # shared parameter matrix
```
In `feature_fused.py`, a feature token's `host_index_emb` lookup and an action
token's target-host lookup both call `shared_embeddings.embed_host(...)` on the
*same* `Parameter` object. In `flat_fused.py`, the same table is used for the
action side, but flat tokens aggregate across all hosts.

**Unified time axis** (`UnifiedTimeEmbedding`) — one embedding table spanning
both history and horizon:
```
Embedding(history_len + max_horizon, 64)     # e.g. 4 + 16 = 20 rows (or 4 + 8 = 12 rows)
ids 0 .. history_len-1        -> history steps (state tokens)
ids history_len .. +K-1       -> horizon steps (action tokens)
```

**Action-only fields** (private tables):
```
action_type_emb  : Embedding(16, 64)
action_pos_emb   : Embedding(max_horizon, 64)
action_valid_emb : Embedding(2, 64)
action_proj      : Linear(64,64) -> GELU -> Linear(64,64)
```

**Sequence assembly and masking:**
```
seq = concat([state_tokens (+ reg token), action_tokens], dim=1)
      flat:    [B, T_hist+1+K, 64]           (e.g. 4+1+8 = 13 tokens at K=8)
      feature: [B, T_hist*52+1+K, 64]        (e.g. 208+1+8 = 217 tokens at K=8)

mask = build_fusion_attention_mask(state_len, action_len, mode)
  "strict"     -> state/reg rows blocked (-inf) from attending to action columns;
                  action rows can attend to everything
  "permissive" -> mask = None (full bidirectional attention)
```

**Transformer stack:** same per-layer recipe as Test 1 (`d_model=64, nhead=4,
ffn_dim=256, Pre-LN, GELU`), with depth matched at **`num_layers=10`** for both
`flat_fused` and `feature_fused` (each layer contains 50,112 params).

**Output layer:**
```
out = LayerNorm(64)(Transformer(seq, mask=mask))
global_token = out[at reg token position]                           # [B, 64]
tokens = concat([state_tokens_out, action_tokens_out], dim=1)        # reg token excluded
```
Returned as `ContextTokens` with `token_type_ids` marking state vs `ACTION`, and
`metadata["action_start_idx"]` recording where action tokens begin.

**Parameters (trainable)**:
- **At Phase 4 `max_horizon=8`**:
  - `flat_fused`: **542,464** (online encoder) + **5,568** (readout head) = **548,032** total.
  - `feature_fused`: **515,648** (online encoder) + **5,568** (readout head) = **521,216** total.
- **At default `max_horizon=16`**:
  - `flat_fused`: **543,488** (online encoder) + **5,568** (readout head) = **549,056** total (0.07% gap to 549,440).
  - `feature_fused`: **516,672** (online encoder) + **5,568** (readout head) = **522,240** total (0.24% gap to 523,520).

---

## 7. Fused Readout Head (`FusedLatentReadout`, Test 2 — replaces §5)

**Purpose:** with context+action fusion now happening inside the encoder (§6),
this is all that's left downstream — a thin per-horizon readout head owned by
`CyberJEPAFused`.

**Input layer**
```
context: ContextTokens (from §6, must have been built with actions != None)
action_tokens = context.tokens[:, action_start_idx:, :]     # [B, K, 64]
```

**Architecture** (no Transformer — `LayerNorm` + `Linear` head + `TargetSpec` embeddings):
```
norm : LayerNorm(64)
head : Linear(64, 64)
granularity_emb : Embedding(4, 64)     # 256
entity_emb      : Embedding(16, 64)    # 1,024

pred_latents = head(norm(action_tokens))          # [B, K, 64]
# if a TargetSpec is supplied:
pred_k = pred_latents[:, target_spec.horizon-1, :] + granularity_emb[g] + entity_emb[e]   # [B, 64]
```

**Output layer:** `[B, K, 64]` (all horizons) or `[B, 64]` (a specific
`TargetSpec`-queried horizon).

**Parameters** (constant regardless of horizon or depth): **5,568** trainable —
`granularity_emb` 256 + `entity_emb` 1,024 + `head` (64×64 Linear) 4,160 + `norm` 128.

---

## 8. Side-by-side Parameter Accounting

### 8.1 Evaluated Phase 4 Configuration ($K=8$, `HORIZON=8`):

| Component | Test 1 — Flat | Test 1 — Feature | Test 2 — Flat Fused | Test 2 — Feature Fused |
|---|---:|---:|---:|---:|
| Context/state encoder | 180,928 | 155,008 | — (folded in) | — (folded in) |
| Action encoder | 11,136 | 11,136 | — (folded in) | — (folded in) |
| Predictor body | 356,352 | 356,352 | — (folded in) | — (folded in) |
| **Fused encoder (10 layers)** | — | — | 542,464 | 515,648 |
| Readout head | — (n/a) | — (n/a) | 5,568 | 5,568 |
| **Total Trainable** | **548,416** | **522,496** | **548,032** | **521,216** |
| Target encoder (frozen, EMA) | 180,928 | 155,008 | 180,928\* | 155,008\* |
| Capacity Match Gap vs. Test 1 | Baseline | Baseline | **-0.07%** | **-0.24%** |

### 8.2 Default Specification ($K=16$, `max_horizon=16`):

| Component | Test 1 — Flat | Test 1 — Feature | Test 2 — Flat Fused | Test 2 — Feature Fused |
|---|---:|---:|---:|---:|
| Context/state encoder | 180,928 | 155,008 | — (folded in) | — (folded in) |
| Action encoder | 11,648 | 11,648 | — (folded in) | — (folded in) |
| Predictor body | 356,864 | 356,864 | — (folded in) | — (folded in) |
| **Fused encoder (10 layers)** | — | — | 543,488 | 516,672 |
| Readout head | — (n/a) | — (n/a) | 5,568 | 5,568 |
| **Total Trainable** | **549,440** | **523,520** | **549,056** | **522,240** |
| Target encoder (frozen, EMA) | 180,928 | 155,008 | 180,928\* | 155,008\* |
| Capacity Match Gap vs. Test 1 | Baseline | Baseline | **-0.07%** | **-0.24%** |

\*Target encoder for Test 2 is an EMA copy of the *fused* encoder class, but is
always called with `actions=None`, so its effective active parameter count during
the single-frame target forward pass matches the corresponding Test 1 context-encoder-only
figure — the extra depth added for action-fusion sits unused on that call.
```
