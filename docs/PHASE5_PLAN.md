# Phase 5 — balanced, regularised separate-action JEPA

## Compatibility decision

Phase 5 uses the retained Phase 3-style `FlatVectorRepresentation`, separate
`ActionEncoder`, `LatentPredictor`, and frozen EMA target encoder. No Phase 4
fused encoder is used. The current checkout does not retain a feature-token
representation, therefore the fixed Phase 5 core is intentionally flat-only;
flat-versus-feature is not an executable Phase 5 factor without first restoring
and auditing the deleted feature implementation.

## Objective

The model optimises

`L = L_JEPA + lambda_v L_variance + lambda_c L_covariance + lambda_a L_action`.

`lambda_a` is zero in Stage 1. Variance and covariance terms act on predicted
latents from the start. The optional action term only separates pairs whose
last observed states are close, first actions differ, and observed future states
are sufficiently distinct. It does not claim counterfactual action causality.

## Stages

1. **Transition balancing:** freeze the median training-set RMS flat-observation
   delta as the static/dynamic threshold. Compare natural, 3:1, 1:1, and 1:3
   static:dynamic *training* sampling. Validation, standard test, and policy
   transfer data retain natural distributions. Select using validation only.
2. **Action-conditioned contrast:** run only the Stage 1 selected ratio and
   predefined `lambda_a` values. Verify action sensitivity and predictive
   separation alongside dispersion metrics.
3. **Architecture ablations:** automatically run the configured reduced
   one-factor-at-a-time sweep, never a factorial grid. Select the best completed
   Stage 2 candidate using the existing validation-only ranking over all five
   seeds, and freeze that choice for restarts. Reuse its three baseline records
   for seeds 1001/2003/3005. Train six alternatives on those seeds (18 new fits):
   latent dimension 32, history length 2/8, horizon 4/16, and `batch_center`.
   Preserve baseline normalizers, transition threshold, ratio, and action
   weight; rebuild windows for history/horizon changes. Skip completed matching
   ablations and atomically persist per-ablation and overall progress.

## Metrics and decision constraints

Every run records JEPA/persistence-relative prediction metrics, static/dynamic
errors, normal/zeroed/shuffled action degradation, effective-rank fraction,
PC1 and cumulative PC5 variance, per-dimension variation, off-diagonal
covariance, standard Macro-F1, and `bline_to_meander` Macro-F1. High F1 does
not qualify as success if encoder latents are collapsed. Test/OOD values are
reported but excluded from Stage 1 candidate selection.

## Commands

From the repository root:

```powershell
python scripts/run_phase5_pipeline.py --stage stage1 --dry-run
python scripts/run_phase5_pipeline.py --stage stage1
```

The pipeline verifies completed shards by checksum and collects only missing
Scenario1b shards from `configs/data/collection_full.yaml` before training.

After reviewing `data/phase5_runs/stage1_results.json`, run Stage 2 only with
the selected non-natural ratio, for example:

```powershell
python scripts/run_phase5_pipeline.py --stage stage2 --selected-ratio static50_dynamic50
```

Stage 3 selects its baseline and executes all reduced ablations automatically:

```powershell
python scripts/run_phase5_pipeline.py --stage stage3
```

It uses the existing shards and Stage 2 results in the default directories;
no selection or skip-collection flags are needed. Add `--dry-run` to preview
the selected baseline and pending/reused counts. Missing completed baseline
results stop the run rather than retraining earlier stages. The naming scheme
and restart behavior are documented in the README's Stage 3 section.
