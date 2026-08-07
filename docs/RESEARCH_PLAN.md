# Research Implementation & Experimentation Roadmap

This document outlines the phased research roadmap for evaluating **Defender-Oriented JEPA** on **CybORG**.

---

## Phase 1: Modular Infrastructure & Trajectory Collection (Current Step)

### Objectives
1. Build a clean, isolated `cyborg/` module structure without touching existing `T-JEPA` tabular code.
2. Implement an automated trajectory collection script ([cyborg/collect_trajectories.py](file:///c:/Users/rohil/Documents/GenAI%20Micro%20Project/T-JEPA-test/cyborg/collect_trajectories.py)) storing structured JSON/HDF5 transitions:
   $$\{t, O_t^{Blue}, a_t^{Blue}, r_t, O_{t+1}^{Blue}, \text{true\_state}_t\}$$
3. Collect baseline trajectory datasets using diverse Red/Blue policy pairs:
   * **Dataset 1**: Blue `Sleep` vs. Red `B_lineAgent` (deterministic aggressive attack path).
   * **Dataset 2**: Blue `Sleep` vs. Red `RedMeanderAgent` (stochastic exploratory attack path).
   * **Dataset 3**: Blue `Random` vs. Red `B_lineAgent` / `RedMeanderAgent` (action diversity).

---

## Phase 2: Trajectory Dataset Characterization & Sparsity Analysis

### Objectives
1. Implement a dataset diagnostic script ([cyborg/analyze_dataset.py](file:///c:/Users/rohil/Documents/GenAI%20Micro%20Project/T-JEPA-test/cyborg/analyze_dataset.py)).
2. Empirical measurement of:
   * **State Sparsity**: Percentage of host features changing per timestep ($\Delta(O_t, O_{t+1})$).
   * **Action Sensitivity**: Feature change rate under defensive actions (`Restore`, `Remove`, `Analyse`) vs. `Sleep`.
   * **Prediction Horizon Curves**: State change divergence across horizons $k \in \{1, 2, 4, 8, 16\}$.
3. Establish loss weighting / host masking rules to prevent JEPA representation collapse.

---

## Phase 3: Candidate JEPA Tokenizer & Encoder Implementation

### Objectives
1. Implement custom CybORG feature/host tokenizers in `cyborg/tokenizer.py`.
2. Adapt T-JEPA model core ([TJEPA](file:///c:/Users/rohil/Documents/GenAI%20Micro%20Project/T-JEPA-test/model.py#L173-L343)) for action-conditioned temporal prediction:
   * Input: Observation history sequence $O_{t-h:t}^{Blue}$ + Action token $a_t^{Blue}$.
   * Target: EMA-encoded future state representation $z_{t+k}$.
3. Train baseline JEPA models offline on collected trajectory datasets.

---

## Phase 4: Representation Evaluation & Downstream Probing

### Objectives
1. **Linear Probing**: Evaluate representation quality by training frozen linear classifiers $z_t \to y$:
   * Probe 1: Host compromise detection ($y \in \{\text{Clean}, \text{User}, \text{System}\}$).
   * Probe 2: Attacker presence / activity detection.
   * Probe 3: Future compromise risk prediction ($z_t \to \text{compromised at } t+4$).
2. **Ablation Comparisons**:
   * Model A: Baseline Flat Vector MLP.
   * Model B: Standard Tabular T-JEPA (within-row feature masking).
   * Model C: Temporal Action-Conditioned Cyber-JEPA (future latent prediction).
