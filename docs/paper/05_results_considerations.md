# Phase 3 Results & Scientific Considerations

## 1. Summary of Phase 3 Findings

Phase 3 systematically investigated whether structured Cyber-JEPA representations (`feature`, `host`, `hierarchical`) genuinely underperform flat temporal representations, or if past underperformance resulted from premature token aggregation before prediction.

### Key Conclusions:
1. **Input Exactness & Single-Frame Target Correction**: Once 52-feature input exactness was restored and target encoding corrected to single-frame $T=1$, feature-structured Cyber-JEPA representations (`feature_token_predictor` F1 = **0.8753**) achieved parity with flat temporal representations (`flat_h4_control` F1 = **0.8776**; $\Delta = -0.0023, p=0.3617$).
2. **Rejection of Learned Query Pooling**: Replacing simple mean pooling with learned query cross-attention (`feature_query_pool` F1 = **0.8584**) did not improve performance; instead, it induced lower effective rank (4.5 vs 7.3) and significant performance degradation ($p=0.0041$).
3. **Hierarchical Topology Bottlenecking**: Two-stage subnet-level aggregation (`hierarchical_current` and `hierarchical_masked_predictor`) caused severe representation collapse (effective rank 1.4, F1 = **0.6603**), demonstrating that coarse spatial pooling destroys fine-grained compromise indicators.

---

## 2. Decision Tree Summary Table

| Representation | Aggregator | Standard Macro F1 | Policy-Transfer F1 | Effective Rank | Verdict |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`flat_h4_control`** | `legacy_last_step_mean` | **0.8776** | **0.8314** | **8.7** | Baseline Control |
| **`feature_token_predictor`** | `token_preserving_predictor` | **0.8753** | **0.8309** | **7.3** | **Competitive Parity ($p=0.3617$)** |
| **`feature_legacy_mean`** | `legacy_last_step_mean` | **0.8766** | **0.8315** | **7.2** | **Competitive Parity ($p=0.7845$)** |
| **`feature_query_pool`** | `learned_query_pool` | 0.8584 | 0.7952 | 4.5 | Degraded / Rank Collapse |
| **`host_legacy_mean`** | `legacy_last_step_mean` | 0.8657 | 0.8153 | 4.8 | Slight Drop |
| **`host_query_pool`** | `learned_query_pool` | 0.8655 | 0.8198 | 4.8 | Slight Drop |
| **`host_token_predictor`** | `token_preserving_predictor` | 0.8657 | 0.8153 | 4.8 | Slight Drop |
| **`hierarchical_current`** | `legacy_last_step_mean` | 0.6603 | 0.5663 | 1.4 | Rank Collapsed |
| **`hierarchical_masked_predictor`** | `token_preserving_predictor` | 0.6603 | 0.5663 | 1.4 | Rank Collapsed |
