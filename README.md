# Cyber-JEPA — Phase 5

This checkout is intentionally trimmed to the Phase 5 separate-action JEPA
study. It uses the fixed Scenario1b collection (18 shards), a flat state
encoder, a separate action encoder, an EMA target encoder, VICReg regularisers,
and stage-gated transition-balancing/action-contrast experiments.

See [docs/PHASE5_PLAN.md](docs/PHASE5_PLAN.md) for the protocol and outputs.

## JarvisLabs setup and Stage 1

From the repository root, install all project and simulator dependencies. The
editable Git dependency in `requirements/sim.in` automatically fetches the
pinned CybORG revision (`dd586a3`):

```bash
python -m pip install -r requirements/sim.in -e .
```

Then run this single reproducible execution command:

```bash
python scripts/run_phase5_pipeline.py --stage stage1
```

It checksum-verifies any existing Scenario1b shards, collects only missing ones,
then runs the 20 Stage 1 fits (four transition ratios × five fixed seeds). Shards
and results are stored under `data/shards/` and `data/phase5_runs/`.

For a safe wiring check without collecting/training:

```bash
python scripts/run_phase5_pipeline.py --stage stage1 --dry-run
```

Stage 2 is deliberately unlocked only after choosing its transition ratio using
the Stage 1 validation-only ranking:

```bash
python scripts/run_phase5_pipeline.py --stage stage2 --selected-ratio static50_dynamic50
```

Use `--skip-collection` only when the fixed 18 shards have already been audited.
