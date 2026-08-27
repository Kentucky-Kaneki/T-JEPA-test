# Cyber-JEPA

Defender-oriented, action-conditioned JEPA world-model research for CybORG.

## Layout

All importable code lives under `src/cyber_jepa/`:

- `models/`: frozen separate-action Test 1 modules and fused Test 2 modules.
- `representations/`: flat and feature encoders, including fused variants.
- `evaluation/`: metrics, probes, Phase 4 variant construction, leakage probe,
  capacity gate, and runner.
- `data/`, `env/`, `training/`, `utils/`: data contracts, collection adapters,
  training, and reproducibility support.

`experiments/phase2/` and `experiments/phase3/` retain completed evidence. Phase 4
outputs are written to `data/phase4_runs/` and are intentionally ignored until a
real run is performed.

## Collecting shards

The collection matrix is defined in `configs/data/collection_full.yaml`. Inspect it
without collecting data with:

```powershell
python scripts/run_collection.py --dry-run
```

Run the full non-overwriting collection and its checksum/quality-gate audit with:

```powershell
python scripts/run_collection.py
```

## Phase 4: fused action pathway

Phase 4 compares the frozen Phase 3 separate-action baselines
`flat_h4_control` and `feature_token_predictor` against four capacity-matched fused
conditions:

- `flat_fused_strict`
- `flat_fused_permissive`
- `feature_fused_strict`
- `feature_fused_permissive`

Run Phase 4 after restoring or regenerating the Phase 3 dataset shards. It does
not retrain or require Test 1 checkpoints: the completed Phase 3 result file is
the frozen Test 1 comparison, while Test 1 architecture definitions are used only
to capacity-match Test 2:

```powershell
python run_phase4.py
```

It uses Phase 3's horizon-8, history-4, five-seed schedule, group split,
normalisation, and `bline_to_meander` policy-transfer direction. It writes the
direct F1 comparison to `data/phase4_runs/phase3_vs_phase4_comparison.json`.

To collect/resume the configured Scenario1b shards and then run the complete
Phase 4 flow in one command, use. The flow verifies the 18-shard Scenario1b
collection contract and that every archived Phase 3 cohort transition is present:

```powershell
python scripts/run_phase4_pipeline.py
```

One-time environment setup (including the pinned CybORG simulator) is:

```powershell
python -m pip install -r requirements/sim.in -e .
```

See `PHASE4_README.md` for architecture details and
`docs/PHASE4_COMPARISON_READINESS.md` for the current audit.
