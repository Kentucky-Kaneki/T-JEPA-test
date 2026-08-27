# Phase 4 comparison readiness

The source layout and Phase 4 implementation are ready for the four fused-action
conditions. The action-semantic lookup is an exact port of the frozen Test 1
`ActionEncoder` lookup logic, and each condition is constructed through the same
capacity-matching search.

The required comparison baselines are the completed Phase 3 configurations
`flat_h4_control` and `feature_token_predictor`. Their reported policy-transfer
macro-F1 values are 0.8314 and 0.8309 respectively; their parameter budgets are
548,416 and 522,432. These records remain in
`experiments/phase3/phase3_sweep_results.json` and the summary is in
`experiments/phase3/PHASE3_REPORT.md`.

The fused candidates are correctly separated into flat/feature and
strict/permissive masking pairs. They use a strict mask that prevents state and
global queries from attending to action keys, while action tokens can attend to
context in both modes. `evaluation/leakage_probe.py` is included because ordinary
prediction loss, action degradation, and action-blind probing cannot distinguish
the two masking modes. It probes the global token with actions present.

Before a valid final comparison can be produced, restore or regenerate the
referenced Scenario1b shard directories. The original Phase 3 checkpoints are not
required: Phase 3's archived per-seed results are the frozen Test 1 evidence, and
the untrained Test 1 architecture is instantiated only to capacity-match Test 2.
Do not retrain Test 1 baselines as part of Phase 4. Phase 4 outcomes
also need to be evaluated on the same policy-transfer direction, split map, and
seed schedule as the Phase 3 evidence before their F1 figures can be compared
directly. `run_phase4.py` enforces this contract: horizon 8, history length 4,
the five Phase 3 seeds, deterministic group splits, train-fitted normalisers,
and the `bline_to_meander` policy-transfer direction. It never offers baseline
retraining or requires baseline checkpoints. Capacity matching is performed at
the same eight-step action horizon used by the Phase 3 models.

The feature representation remains a reconstructed 13 hosts by 4 features mapping.
No original Phase 3 feature encoder is present to verify it. Confirm that mapping
against the archived Phase 3 implementation before treating the feature comparison
as conclusive.
