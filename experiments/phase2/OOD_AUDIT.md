# Out-of-Distribution (OOD) Data Split Audit

## Executive Summary
This document provides an audit of the train, validation, and Out-of-Distribution (OOD) test splits used in the Cyber-JEPA experimental framework.

---

## 1. Dataset Composition
The dataset consists of **18 shards**, each containing 100 episodes of 50 timesteps (total **300 episodes** / **91,800 transitions**):

- **Red Policies**: `bline` (deterministic shortest-path kill chain), `meander` (stochastic scanning & lateral movement)
- **Blue Policies**: `random` (random defender actions), `coverage` (systematic host monitoring), `sleep` (no-op defender)
- **Random Seeds**: 1001, 2003, 3005

---

## 2. Phase 1 & 2 Official Split Implementation

In `run_full_experiment_sweep.py` (and `dataset.py` `generate_episode_splits`):

```python
unique_episodes = sorted(list(set(all_episodes))) # 300 episodes
rng = np.random.RandomState(42)
rng.shuffle(unique_episodes)

train_end = int(0.70 * N_ep)  # 210 episodes
val_end   = int(0.85 * N_ep)  # 45 episodes
test_end  = N_ep              # 45 episodes
```

### Audit Finding: Episode-Level Random Holdout
- **Holdout Mechanism**: The 300 total episodes are randomly shuffled under seed 42 and split into **70% Train (210 episodes)**, **15% Validation (45 episodes)**, and **15% OOD Test (45 episodes)**.
- **Factor Overlap**: Because episodes from all Red policy × Blue policy × Seed combinations are randomly shuffled into the pool prior to splitting, **the OOD test set contains episodes generated from the same Red/Blue policies and scenarios seen during training**.
- **Scope of Holdout**: The holdout evaluates **unseen trajectory episodes**, NOT unseen Red policies, unseen network topologies, or unseen scenario configurations.

---

## 3. Alternative Policy Transfer Split (OOD Transfer Sidecar)

In `dataset.py`, `generate_ood_splits()` provides a true zero-shot policy transfer split:

- `bline_to_meander`: Train on all `bline` episodes (150 ep), test on all `meander` episodes (150 ep).
- `meander_to_bline`: Train on all `meander` episodes (150 ep), test on all `bline` episodes (150 ep).

*Note*: In Phase 1 and Phase 2 primary sweeps, the **70/15/15 episode random holdout** is maintained as the official benchmark to preserve comparability with preregistered Section 13 rules.

---

## 4. Terminology Classification Rule

Per Section 18 & 26 instructions:
- Do **NOT** refer to the primary 15% test split as "compositional OOD" or "zero-shot policy transfer."
- Refer to it precisely as **"unseen episode-level holdout"**.
- Reserve "policy-transfer OOD" strictly for the `bline_to_meander` / `meander_to_bline` sidecar splits.
