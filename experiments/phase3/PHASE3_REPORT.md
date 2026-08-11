# Cyber-JEPA Phase 3: Representation Fairness Audit & Aggregator Interventions

## Executive Summary

Phase 3 audited structured representations (`feature`, `host`, `hierarchical`) under **parameter-matched capacity**, **explicit context aggregator interventions**, and **single-frame target encoding (T=1)**.

## Core Sweep Results Summary (5 Seeds / Config)

| Configuration | Standard F1 | Policy-Transfer F1 | AUROC | Effective Rank | Collapsed Runs | Trainable Params |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `feature_legacy_mean` | 0.8766 ± 0.0063 | 0.8315 ± 0.0086 | 0.9483 | 7.2 | 1/5 | 522,432 |
| `feature_query_pool` | 0.8584 ± 0.0169 | 0.7952 ± 0.0266 | 0.9380 | 4.5 | 4/5 | 539,264 |
| `feature_token_predictor` | 0.8753 ± 0.0045 | 0.8309 ± 0.0060 | 0.9481 | 7.3 | 1/5 | 522,432 |
| `flat_h4_control` | 0.8776 ± 0.0043 | 0.8314 ± 0.0055 | 0.9514 | 8.7 | 0/5 | 548,416 |
| `hierarchical_current` | 0.6603 ± 0.1062 | 0.5663 ± 0.1341 | 0.7439 | 1.4 | 5/5 | 685,952 |
| `hierarchical_masked_predictor` | 0.6603 ± 0.1062 | 0.5663 ± 0.1341 | 0.7439 | 1.4 | 5/5 | 685,952 |
| `host_legacy_mean` | 0.8657 ± 0.0192 | 0.8153 ± 0.0224 | 0.9425 | 4.8 | 4/5 | 618,816 |
| `host_query_pool` | 0.8655 ± 0.0035 | 0.8198 ± 0.0048 | 0.9407 | 4.8 | 4/5 | 635,648 |
| `host_token_predictor` | 0.8657 ± 0.0192 | 0.8153 ± 0.0224 | 0.9425 | 4.8 | 4/5 | 618,816 |

## Paired Bootstrap Analysis vs `flat_h4_control`

### `feature_legacy_mean` vs `flat_h4_control`
- **Standard F1 Delta**: -0.0010 (95% CI: [-0.0064, +0.0051], p=0.7845)
- **Policy-Transfer F1 Delta**: +0.0001 (95% CI: [-0.0070, +0.0071], p=0.9591)

### `feature_query_pool` vs `flat_h4_control`
- **Standard F1 Delta**: -0.0192 (95% CI: [-0.0296, -0.0059], p=0.0041)
- **Policy-Transfer F1 Delta**: -0.0361 (95% CI: [-0.0545, -0.0136], p=0.0018)

### `feature_token_predictor` vs `flat_h4_control`
- **Standard F1 Delta**: -0.0023 (95% CI: [-0.0065, +0.0028], p=0.3617)
- **Policy-Transfer F1 Delta**: -0.0005 (95% CI: [-0.0062, +0.0052], p=0.8405)

### `host_legacy_mean` vs `flat_h4_control`
- **Standard F1 Delta**: -0.0119 (95% CI: [-0.0283, +0.0006], p=0.1222)
- **Policy-Transfer F1 Delta**: -0.0161 (95% CI: [-0.0344, -0.0037], p=0.0493)

### `host_query_pool` vs `flat_h4_control`
- **Standard F1 Delta**: -0.0121 (95% CI: [-0.0157, -0.0083], p=0.0001)
- **Policy-Transfer F1 Delta**: -0.0116 (95% CI: [-0.0157, -0.0071], p=0.0001)

### `host_token_predictor` vs `flat_h4_control`
- **Standard F1 Delta**: -0.0119 (95% CI: [-0.0283, +0.0006], p=0.1222)
- **Policy-Transfer F1 Delta**: -0.0161 (95% CI: [-0.0344, -0.0037], p=0.0493)

### `hierarchical_current` vs `flat_h4_control`
- **Standard F1 Delta**: -0.2173 (95% CI: [-0.2833, -0.1236], p=0.0001)
- **Policy-Transfer F1 Delta**: -0.2651 (95% CI: [-0.3578, -0.1393], p=0.0001)

### `hierarchical_masked_predictor` vs `flat_h4_control`
- **Standard F1 Delta**: -0.2173 (95% CI: [-0.2833, -0.1236], p=0.0001)
- **Policy-Transfer F1 Delta**: -0.2651 (95% CI: [-0.3578, -0.1393], p=0.0001)

## Preregistered Section 13 Decision Tree & Scientific Findings

1. **Aggregator Hypothesis**: Mean pooling over structured entities was tested directly against learned query pooling and token-preserving prediction.
2. **Representation Collapse Audit**: Effective rank diagnostics confirm whether structured representations maintain rank stability or experience collapse under JEPA losses.
3. **Policy-Transfer Robustness**: Generalization across adversarial red-agent strategies evaluates structural inductive bias.

---
*Generated automatically by `scripts/analyze_phase3_results.py`*