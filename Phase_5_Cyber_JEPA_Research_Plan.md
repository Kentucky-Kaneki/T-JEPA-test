# Phase 5 — Cyber-JEPA Research Plan

## 1. Phase 5 Objective

### Primary research question

Can a separately conditioned, flat-representation JEPA learn a high-rank latent representation that captures action-dependent cyber dynamics rather than relying primarily on environmental persistence?

### Phase 5 design principles

- Apply the **separate state/action encoder + predictor architecture** from the beginning.
- Apply **VICReg-style variance and covariance regularization** from the beginning.
- Use a **flat input representation**.
- Address persistence through **transition-aware dataset balancing** before testing action-conditioned contrast.
- Test the **action-conditioned contrastive objective** after the transition-balancing design is fixed.
- Perform architectural and hyperparameter ablations only after the core Phase 5 objective is established.
- Keep validation/test distributions natural unless an experiment explicitly specifies otherwise.
- Preserve clean baselines and fixed evaluation protocols across all ablations.
- Treat representation quality and dynamics quality as first-class outcomes; do not select models by Macro-F1 alone.

---

# 2. Core Phase 5 Architecture

## 2.1 State pathway

- Observation history:
  - Use the established temporal observation window.
  - Encode the full observation at each timestep as a **single flat token**.
- State encoder:
  - Separate from the action encoder.
  - Transformer-based.
  - Start from the lower-capacity architecture used by the successful Phase 3 pathway rather than the 10-layer fused Phase 4 encoder.
- Produce:
  \[
  Z_t = E_\theta(O_{t-H:t})
  \]

## 2.2 Action pathway

- Encode the future/planned action sequence independently.
- Preserve action temporal ordering.
- Do not fuse state and action tokens into a single encoder stack.
- Produce:
  \[
  A^Z = E_A(A_{t:t+K})
  \]

## 2.3 Predictor

- Predict future target representations conditioned on both current state representation and action representation:
  \[
  \hat Z_{t+k}=P_\phi(Z_t,A^Z)
  \]
- Use a separate predictor Transformer/MLP pathway.
- Initial depth should remain close to the established Phase 3 design.
- Predictor bottleneck is an explicit later ablation, not an initial uncontrolled change.

## 2.4 Target encoder

- Maintain an EMA target encoder.
- Target encoder receives future observations only.
- Produce:
  \[
  Z_{t+k}=E_{\bar\theta}(O_{t+k})
  \]
- Do not allow gradients through the target encoder.

---

# 3. Representation

## 3.1 Flat representation

- Use flat timestep-level tokens.
- Do not use the feature-token representation in the primary Phase 5 model.
- Preserve the same basic feature dimensionality and tokenization semantics used for the Phase 3 flat baseline wherever possible.
- Keep representation dimensionality fixed at the established latent dimension unless an explicit dimensionality ablation is being performed.

## 3.2 Representation objective

Primary objective:

\[
L_{\mathrm{JEPA}}
\]

augmented from the start with:

\[
L_{\mathrm{VICReg}}
=
\lambda_v L_{\mathrm{variance}}
+
\lambda_c L_{\mathrm{covariance}}
\]

Initial combined objective:

\[
L_{\mathrm{base}}
=
L_{\mathrm{JEPA}}
+
\lambda_v L_{\mathrm{variance}}
+
\lambda_c L_{\mathrm{covariance}}
\]

Do not introduce Barlow Twins simultaneously with VICReg in the primary Phase 5 model.

---

# 4. Anti-Collapse Regularization

## 4.1 Variance regularization

- Apply variance regularization to discourage latent dimensions from becoming inactive.
- Monitor per-dimension batch variance.
- Define and freeze a minimum acceptable variance threshold before final experiments.
- Report:
  - minimum latent standard deviation,
  - median latent standard deviation,
  - distribution of latent standard deviations,
  - number/fraction of dimensions below the variance threshold.

## 4.2 Covariance regularization

- Penalize off-diagonal covariance between latent dimensions.
- Monitor the covariance matrix and eigenvalue spectrum.
- Track whether variance is distributed across multiple latent directions rather than concentrated in a single principal component.

## 4.3 Anti-collapse safeguards

- Do not interpret non-zero overall latent variance as sufficient evidence of a healthy representation.
- Require both:
  - adequate per-dimension variance,
  - adequate effective rank / non-degenerate covariance spectrum.
- Keep the anti-collapse weights fixed while testing transition-balancing and action-contrast hypotheses unless those weights are explicitly under ablation.

---

# 5. Phase 5 Experiment Order

## Stage 1 — Transition Balancing

### Objective

Reduce the incentive to exploit the persistence shortcut by controlling the training distribution of static versus dynamic transitions.

### 5.1 Define transition magnitude

Define a deterministic transition-distance function:

\[
\Delta_{t,k}=d(O_t,O_{t+k})
\]

or, where more appropriate, a normalized state-change measure.

The distance function must be:

- defined before final model evaluation,
- independent of model predictions,
- reproducible,
- calculated consistently across training/validation/test diagnostics.

### 5.2 Static/dynamic threshold

Define a threshold:

\[
\tau
\]

such that:

\[
\Delta \leq \tau \Rightarrow \text{static}
\]

\[
\Delta > \tau \Rightarrow \text{dynamic}
\]

Before selecting the final threshold:

- inspect the empirical transition-distance distribution on the training data;
- report candidate thresholds;
- avoid choosing a threshold solely because it produces the best downstream F1;
- document the final threshold and rationale;
- retain the threshold unchanged for all Phase 5 core comparisons.

### 5.3 1:1 sampling

Primary training distribution:

\[
P(\text{static})=0.5
\]

\[
P(\text{dynamic})=0.5
\]

Use transition-aware sampling rather than deleting all naturally occurring static transitions.

Do not force the validation/test sets to 1:1.

### 5.4 Natural-distribution evaluation

- Preserve the natural transition distribution for:
  - validation,
  - standard test,
  - OOD Meander test.
- Report both:
  - balanced-training performance,
  - natural-distribution performance.
- Report the static/dynamic composition of every evaluation split.

### 5.5 Transition magnitude diagnostics

For every split report:

- median \(\Delta\),
- mean \(\Delta\),
- percentile distribution,
- static fraction,
- dynamic fraction,
- action-associated transition fraction where measurable.

### 5.6 Balancing ablations

Do not assume 1:1 is optimal.

After the primary 1:1 experiment, consider:

- natural distribution,
- 75:25 static:dynamic,
- 50:50 static:dynamic,
- 25:75 static:dynamic.

Use the same evaluation distribution for fair comparison.

Primary hypothesis:

\[
\text{balanced dynamic exposure}
\rightarrow
\text{higher persistence-relative improvement}
\]

---

# 6. Stage 2 — Action-Conditioned Contrastive Objective

## 6.1 Objective

Encourage the predictor to distinguish futures that differ because of action-conditioned dynamics.

Base prediction:

\[
\hat Z_A=P(Z_t,A)
\]

For an alternative action sequence:

\[
\hat Z_B=P(Z_t,B)
\]

When the corresponding observed futures are meaningfully different:

\[
d(O^A_{t+k},O^B_{t+k})>\tau_{\mathrm{future}}
\]

encourage:

\[
\hat Z_A \not\approx \hat Z_B
\]

while retaining:

\[
\hat Z_A\approx Z_A
\]

and:

\[
\hat Z_B\approx Z_B
\]

## 6.2 Positive/negative pair construction

- Do not contrast arbitrary action pairs.
- Do not force different actions apart when their observable futures are effectively identical.
- Prefer pairs satisfying:
  - comparable initial state/context,
  - different action sequences,
  - sufficiently different observed future outcomes.
- Define the future-difference criterion before final evaluation.
- Keep pair-selection rules identical across experiments.

## 6.3 Contrastive loss

Initial objective:

\[
L =
L_{\mathrm{JEPA}}
+
\lambda_v L_{\mathrm{variance}}
+
\lambda_c L_{\mathrm{covariance}}
+
\lambda_a L_{\mathrm{action}}
\]

Treat:

\[
\lambda_a
\]

as an explicit hyperparameter.

Test a small predefined set rather than tuning continuously against the test set.

## 6.4 Contrastive safeguards

- Prevent the contrastive term from dominating JEPA prediction.
- Monitor:
  - JEPA prediction error,
  - action sensitivity,
  - action-conditioned separation,
  - latent rank,
  - Macro-F1/OOD Macro-F1.
- Verify that improved action separation does not simply create artificial latent dispersion without better future prediction.

---

# 7. Stage 3 — Architectural and Hyperparameter Ablations

Only begin after the transition-balancing and action-contrast experiments have produced a stable core configuration.

## 7.1 Predictor depth

Test a predefined shallow-to-moderate range around the Phase 3-style predictor depth.

Example:

- 1 layer,
- 2 layers,
- 3 layers.

Do not return to the Phase 4 10-layer fused stack as the default architecture.

Evaluate:

- effective rank,
- JEPA error,
- persistence gain,
- action sensitivity,
- F1,
- OOD F1.

## 7.2 State encoder depth

Test:

- 1 layer,
- 2 layers,
- 3 layers,
- optionally 4 layers if justified by capacity/performance.

Do not increase depth merely to improve training loss.

## 7.3 Action encoder depth

Test a small range appropriate to action-sequence complexity.

Prioritize:

- parameter efficiency,
- action representation quality,
- predictor performance.

## 7.4 Predictor bottleneck

Test:

- no bottleneck:
  \[
  64\rightarrow64
  \]
- moderate:
  \[
  64\rightarrow32\rightarrow64
  \]
- stronger:
  \[
  64\rightarrow16\rightarrow64
  \]

Use bottleneck width as an explicit architectural variable.

Do not assume a stronger bottleneck is better.

## 7.5 Latent dimensionality

Primary latent dimension remains fixed.

Optional ablation:

- 32,
- 64,
- 128.

Use this to distinguish:

- genuine representation-capacity requirements,
- artificial collapse caused by an unnecessarily large latent space.

Do not compare models only by raw F1; report effective rank relative to latent dimension.

## 7.6 Context length

Test temporal history length around the established value.

Example:

- short,
- baseline,
- longer history.

Monitor whether longer context improves genuine dynamics learning or merely improves persistence estimation.

## 7.7 Prediction horizon

Test:

- shorter horizon,
- baseline horizon,
- longer horizon.

Report horizon-specific prediction error.

Do not aggregate all horizons into a single number only.

Report:

\[
E_{t+1},E_{t+2},...,E_{t+K}
\]

to determine whether the model learns dynamics at short horizons but collapses into persistence at longer horizons.

---

# 8. EMA Hyperparameters

## 8.1 EMA momentum

Primary configuration:

- retain the established EMA schedule initially.

Then ablate:

- lower momentum,
- baseline momentum,
- higher momentum.

Where feasible test both:

- constant momentum,
- scheduled momentum.

Example schedule family:

\[
m_t:m_{\mathrm{start}}\rightarrow m_{\mathrm{end}}
\]

## 8.2 EMA evaluation

Monitor:

- target/prediction distribution mismatch,
- latent variance,
- effective rank,
- JEPA error,
- training stability.

Do not select EMA solely by downstream F1.

---

# 9. Target Representation Controls

## 9.1 Target normalization

Evaluate:

- no target normalization,
- batch-level normalization,
- feature/dimension normalization where compatible with the objective.

Keep the normalization definition fixed during each experiment.

## 9.2 Target centering

Evaluate:

\[
Z' = Z-\mu_Z
\]

versus uncentered targets.

Measure whether centering:

- reduces dominant mean components,
- improves covariance spectrum,
- improves effective rank,
- changes JEPA prediction stability.

## 9.3 Ordering

Test target normalization and centering separately before combining them.

Do not introduce both simultaneously without an ablation.

---

# 10. Additional Architectural Ablations

## 10.1 State/action fusion location

Primary model:

- separate state and action encoders,
- fusion only at predictor input.

Optional ablation:

- shallow fusion before predictor,
- cross-attention between state and action representations,
- late cross-attention.

Avoid returning to unrestricted encoder-level fusion as the primary design.

## 10.2 Action representation

Test whether actions are represented as:

- learned action embeddings,
- action embeddings + temporal positional encoding,
- action embeddings + action-type embedding,
- action embeddings + timestep/horizon embedding.

Keep the action semantics identical across comparisons.

## 10.3 Temporal positional encoding

Compare, where supported:

- learned positional embeddings,
- sinusoidal encoding,
- relative positional encoding.

Do not change temporal encoding simultaneously with depth unless explicitly intended.

## 10.4 Predictor conditioning mechanism

Compare:

- concatenation,
- additive conditioning,
- cross-attention,
- gated conditioning.

Primary baseline should remain the simplest successful Phase 3-style conditioning mechanism.

## 10.5 Residual/gating controls

Optional:

- gated action injection,
- FiLM-style action conditioning,
- residual action pathway.

Evaluate whether the action pathway can influence prediction without dominating the state representation.

---

# 11. Additional Loss Ablations

Do not add all alternatives simultaneously.

## 11.1 VICReg weight

Ablate:

\[
\lambda_v,\lambda_c
\]

jointly only after establishing that VICReg is stabilizing the latent.

## 11.2 Variance-only

\[
L=L_{\mathrm{JEPA}}+\lambda_vL_{\mathrm{variance}}
\]

Purpose:

- determine whether collapse is primarily caused by inactive dimensions.

## 11.3 Covariance-only

\[
L=L_{\mathrm{JEPA}}+\lambda_cL_{\mathrm{covariance}}
\]

Purpose:

- determine whether collapse is primarily caused by redundancy.

## 11.4 Full VICReg

\[
L=L_{\mathrm{JEPA}}
+\lambda_vL_{\mathrm{variance}}
+\lambda_cL_{\mathrm{covariance}}
\]

Primary anti-collapse configuration.

## 11.5 Alternative anti-collapse method

Optional later comparison:

- Barlow Twins-style correlation penalty,
- whitening,
- other decorrelation/isotropy objective.

Do not include these in the primary model unless justified by an earlier ablation.

---

# 12. Persistence Diagnostics

Every primary Phase 5 experiment must include a persistence baseline.

## 12.1 Latent persistence

Use:

\[
\hat Z_{t+k}=Z_t
\]

as the simple persistence predictor.

## 12.2 Persistence-relative improvement

Define:

\[
G_{\mathrm{persist}}
=
\frac{
E_{\mathrm{persist}}-E_{\mathrm{JEPA}}
}{
E_{\mathrm{persist}}
}
\]

Report:

- absolute persistence error,
- JEPA prediction error,
- absolute improvement,
- normalized improvement.

## 12.3 Horizon-specific persistence

Report persistence-relative improvement for every prediction horizon:

\[
G_{t+1},G_{t+2},...,G_{t+K}
\]

This is required to determine whether JEPA learns genuine short-term dynamics or merely copies state.

---

# 13. Action Sensitivity Diagnostics

For each trained model evaluate:

### Normal

\[
P(Z_t,A)
\]

### Zeroed action

\[
P(Z_t,0)
\]

### Shuffled action

\[
P(Z_t,A_{\mathrm{shuffle}})
\]

Report:

\[
\Delta_{\mathrm{zero}}
\]

and:

\[
\Delta_{\mathrm{shuffle}}
\]

relative to normal prediction.

Action sensitivity must not be interpreted alone as proof of learned causal dynamics.

---

# 14. Required Metrics

## 14.1 Representation geometry

- Effective Rank Fraction
- Effective Rank
- PC1 variance explained
- PC5 cumulative variance explained
- median latent standard deviation
- per-dimension latent standard deviation
- covariance spectrum
- covariance off-diagonal magnitude
- latent mean magnitude

## 14.2 Dynamics

- JEPA prediction error
- horizon-specific JEPA prediction error
- persistence error
- persistence-relative improvement
- action sensitivity
- action-conditioned prediction separation

## 14.3 Downstream utility

- standard Macro-F1
- OOD Macro-F1
- per-class F1 where useful
- OOD degradation relative to in-distribution performance

---

# 15. Primary Representation-Quality Gates

Use the Phase 4 representation-quality philosophy.

Do not declare a model successful solely because F1 is high.

Track at minimum:

\[
\text{Effective Rank Fraction}
\]

\[
\text{PC1 variance}
\]

\[
\text{PC5 variance}
\]

\[
\text{median latent std}
\]

\[
\text{covariance spectrum}
\]

A model showing high F1 but severe dimensional collapse should be classified as a representation-quality failure rather than a clean success.

---

# 16. Statistical Protocol

## 16.1 Seeds

- Use multiple independent random seeds.
- Keep the same seed set across core model comparisons.
- Do not drop failed seeds without documenting the reason.

## 16.2 Reporting

For every primary model report:

- mean,
- standard deviation,
- median where useful,
- per-seed values,
- confidence intervals where appropriate.

## 16.3 Comparisons

Primary comparisons should be paired by seed where possible.

Report both:

- absolute difference,
- relative difference.

Do not select the best seed for presentation.

---

# 17. Experimental Controls

Keep constant unless explicitly being tested:

- dataset split,
- OOD split,
- observation preprocessing,
- action semantics,
- latent dimensionality,
- optimizer,
- base learning rate,
- batch size,
- training budget,
- random seed set,
- evaluation code,
- persistence baseline,
- downstream probe protocol.

When an item changes, record it as an explicit experimental factor.

---

# 18. Recommended Phase 5 Execution Sequence

## P5.0 — Core model initialization

Apply from the beginning:

- separate state encoder,
- separate action encoder,
- separate predictor,
- flat representation,
- VICReg variance,
- VICReg covariance,
- Phase 3-style lower-capacity depth,
- EMA target encoder.

## P5.1 — Transition threshold study

Determine and freeze:

\[
\tau
\]

for static/dynamic classification.

## P5.2 — Transition-ratio study

Evaluate:

- natural,
- 75:25,
- 50:50,
- 25:75.

Select the ratio based on:
- persistence-relative improvement,
- dynamics error,
- representation quality,
- downstream performance.

Do not select solely by F1.

## P5.3 — Action-contrast study

With the selected transition distribution:

- no action contrast,
- low \(\lambda_a\),
- medium \(\lambda_a\),
- high \(\lambda_a\).

Select based on action-conditioned dynamics and representation quality.

## P5.4 — Architecture ablations

Test:

- predictor depth,
- state encoder depth,
- action encoder depth,
- predictor bottleneck,
- latent dimension,
- context length,
- prediction horizon.

## P5.5 — EMA/target ablations

Test:

- EMA momentum,
- EMA schedule,
- target normalization,
- target centering.

## P5.6 — Optional alternative anti-collapse methods

Only after the VICReg configuration is understood:

- variance-only,
- covariance-only,
- Barlow Twins-style penalty,
- whitening/isotropy approaches.

---

# 19. Phase 5 Success Criteria

A strong Phase 5 result should demonstrate several outcomes simultaneously:

### Representation

\[
\text{Effective Rank}\uparrow
\]

\[
\text{PC1 dominance}\downarrow
\]

\[
\text{PC5 coverage}\uparrow
\]

\[
\text{latent variance remains healthy}
\]

### Dynamics

\[
\text{JEPA error}\downarrow
\]

\[
\text{persistence-relative improvement}\uparrow
\]

\[
\text{action sensitivity}\uparrow
\]

### Downstream

\[
\text{Macro-F1}\uparrow
\]

\[
\text{OOD Macro-F1}\uparrow
\]

The strongest evidence is not simply:

\[
F1_{\mathrm{Phase5}}>F1_{\mathrm{Phase4}}
\]

but:

\[
\boxed{
\text{high-rank representation}
+
\text{action-sensitive prediction}
+
\text{improvement over persistence}
+
\text{strong OOD performance}
}
\]

---

# 20. Required Final Phase 5 Reporting

Final report must include:

1. Architecture diagram.
2. Exact parameter count.
3. Exact trainable/frozen components.
4. Full hyperparameter table.
5. Transition-distance definition.
6. Static/dynamic threshold.
7. Sampling ratios.
8. Natural versus balanced split distributions.
9. Full loss equation.
10. VICReg coefficients.
11. Action-contrast definition and pair-selection criteria.
12. EMA schedule.
13. Predictor bottleneck configuration.
14. Target normalization/centering configuration.
15. Per-seed results.
16. Representation-quality metrics.
17. Persistence comparison.
18. Action-ablation results.
19. Standard Macro-F1.
20. OOD Macro-F1.
21. Ablation table.
22. Failure cases.
23. Evidence for or against persistence shortcut.
24. Evidence for or against dimensional collapse.
25. Final selected configuration and rationale.
