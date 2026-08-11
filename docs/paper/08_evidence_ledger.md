# Phase 3 Evidence Ledger

This document logs all empirical claims, decision-tree verdicts, and statistical evidence established in **Phase 3** of the Cyber-JEPA project.

---

## Preregistered Decision Tree Outcomes

| Claim / Hypothesis | Status | Paired Delta vs `flat_h4_control` | 95% Confidence Interval | p-value | Key Evidence & Observations |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Feature Tokens Match Flat Performance** (Decision Rule 14.4) | **SUPPORTED (Competitive)** | -0.0023 | [-0.0065, +0.0028] | 0.3617 | `feature_token_predictor` (F1 0.8753) matches `flat_h4_control` (F1 0.8776) within 0.0023 F1 ($p=0.3617$). |
| **Feature Legacy Mean Competitiveness** | **SUPPORTED (Competitive)** | -0.0010 | [-0.0064, +0.0051] | 0.7845 | `feature_legacy_mean` (F1 0.8766) matches flat within 0.0010 F1 ($p=0.7845$). |
| **Mean-Pooling Bottleneck Hypothesis** (Decision Rule 14.3) | **REJECTED** | -0.0192 (query pool) | [-0.0296, -0.0059] | 0.0041 | Learned query pooling degraded performance ($0.8584$) and increased rank collapse (4/5 runs collapsed). |
| **Host Token Competitiveness** | **PARTIALLY SUPPORTED** | -0.0119 | [-0.0283, +0.0006] | 0.1222 | `host_token_predictor` (F1 0.8657) remains competitive, though slightly below feature-level representations. |
| **Hierarchical Representation** | **REJECTED (Rank Collapsed)** | -0.2173 | [-0.2833, -0.1236] | 0.0001 | Subnet-level two-stage pooling suffers severe rank collapse (EffRank 1.4, 5/5 runs collapsed). |

---

## Primary Metric Artifact Hashes

* **Cohort File**: `experiments/phase3/phase3_cohort.parquet`
* **Cohort SHA-256**: Verified in `experiments/phase3/phase3_cohort.sha256`
* **Full Matrix Results**: `experiments/phase3/phase3_sweep_results.json` (45 runs completed)
* **Summary Report**: `experiments/phase3/PHASE3_REPORT.md`
