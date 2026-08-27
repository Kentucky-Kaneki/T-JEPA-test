"""Run the Phase 4 fused-action sweep on the frozen Phase 3 comparison setup.

This driver intentionally never retrains the separate-action baselines. It reuses
their archived Phase 3 results and architecture-derived parameter budgets, then
uses Phase 3's data split, normalisation, horizon, seed schedule, and
policy-transfer direction to train/evaluate the four fused conditions. Results are
written below ``data/phase4_runs`` by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader

from cyber_jepa.data.dataset import (
    CyberJEPADataset,
    generate_group_splits,
    generate_policy_transfer_splits,
    verify_dataset_integrity,
)
from cyber_jepa.data.storage import DatasetStorageManager
from cyber_jepa.evaluation.capacity_budget_gate import check_all_variants_capacity_budget
from cyber_jepa.evaluation.phase4_runner import (
    build_selection_reports,
    evaluate_variant,
    run_vram_preflight,
    train_variant,
)
from cyber_jepa.evaluation.phase4_variants import build_all_phase4_variants
from cyber_jepa.models.jepa import CyberJEPA
from cyber_jepa.representations.feature import FeatureTokenRepresentation
from cyber_jepa.representations.flat import FlatVectorRepresentation
from cyber_jepa.utils.reproducibility import seed_worker, set_deterministic_seed


PHASE3_SEEDS = [1001, 2003, 3005, 4007, 5009]
HORIZON = 8
HISTORY_LEN = 4
HIDDEN_DIM = 64
BATCH_SIZE = 64
MAX_EPOCHS = 20
MIN_EPOCHS = 10
PATIENCE = 5
POLICY_TRANSFER_DIRECTION = "bline_to_meander"


def _build_baseline_spec(representation: str) -> CyberJEPA:
    """Build a Phase 3 baseline *shape* for capacity matching only.

    No baseline weights are needed: Test 1's completed Phase 3 metrics are the
    frozen comparison result, while parameter matching depends solely on the
    architecture and not on checkpoint values.
    """
    if representation == "flat_h4_control":
        encoder = FlatVectorRepresentation(obs_dim=52, hidden_dim=HIDDEN_DIM, history_len=HISTORY_LEN)
        aggregator_mode = "legacy_last_step_mean"
    elif representation == "feature_token_predictor":
        encoder = FeatureTokenRepresentation(obs_dim=52, hidden_dim=HIDDEN_DIM, history_len=HISTORY_LEN)
        aggregator_mode = "token_preserving_predictor"
    else:
        raise ValueError(f"Unsupported Phase 3 baseline: {representation}")

    model = CyberJEPA(
        online_encoder=encoder,
        hidden_dim=HIDDEN_DIM,
        max_horizon=HORIZON,
        aggregator_mode=aggregator_mode,
    )
    return model


def _phase3_reference_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 3 results: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    wanted = {"flat_h4_control", "feature_token_predictor"}
    return [row for row in rows if row.get("config_id") in wanted]


def _write_comparison(phase4_runs: list[dict[str, Any]], phase3_rows: list[dict[str, Any]], output_dir: Path) -> None:
    """Write a direct Phase 3/Test 1 versus Phase 4/Test 2 F1 comparison."""
    phase3_by_name: dict[str, list[float]] = defaultdict(list)
    for row in phase3_rows:
        phase3_by_name[str(row["config_id"])].append(float(row["policy_transfer_macro_f1"]))
    phase4_by_name: dict[str, list[float]] = defaultdict(list)
    for row in phase4_runs:
        phase4_by_name[str(row["representation"])].append(float(row["ood_future_compromise_macro_f1"]))

    payload = {
        "comparison_contract": {
            "phase3_baselines": ["flat_h4_control", "feature_token_predictor"],
            "horizon": HORIZON,
            "history_len": HISTORY_LEN,
            "seeds": PHASE3_SEEDS,
            "policy_transfer_direction": POLICY_TRANSFER_DIRECTION,
            "phase3_baselines_retrained": False,
            "phase3_baseline_checkpoints_required": False,
        },
        "phase3_policy_transfer_macro_f1": {
            name: {"per_seed": values, "mean": sum(values) / len(values)}
            for name, values in phase3_by_name.items()
        },
        "phase4_policy_transfer_macro_f1": {
            name: {"per_seed": values, "mean": sum(values) / len(values)}
            for name, values in phase4_by_name.items()
        },
    }
    (output_dir / "phase3_vs_phase4_comparison.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-dir", type=Path, default=Path("data/shards"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/phase4_runs"))
    parser.add_argument("--phase3-results", type=Path, default=Path("experiments/phase3/phase3_sweep_results.json"))
    parser.add_argument("--skip-vram-preflight", action="store_true")
    args = parser.parse_args(argv)

    shard_dirs = sorted(path for path in args.shards_dir.iterdir() if (path / "transitions.parquet").exists())
    if not shard_dirs:
        raise FileNotFoundError(f"No Phase 3 shards found in {args.shards_dir}.")
    phase3_rows = _phase3_reference_results(args.phase3_results)
    if len(phase3_rows) != 10:
        raise ValueError("Expected five archived Phase 3 rows for each frozen baseline.")
    cohort = Path("experiments/phase3/phase3_cohort.parquet")
    cohort_hash = cohort.with_suffix(".sha256")
    if not cohort.exists() or not cohort_hash.exists():
        raise FileNotFoundError("Missing Phase 3 cohort or checksum.")
    if hashlib.sha256(cohort.read_bytes()).hexdigest() != cohort_hash.read_text(encoding="utf-8").strip():
        raise ValueError("Phase 3 cohort checksum mismatch.")
    verify_dataset_integrity(shard_dirs)
    for shard_dir in shard_dirs:
        DatasetStorageManager.verify_shard_checksums(shard_dir)

    transitions = pd.concat([pd.read_parquet(path / "transitions.parquet") for path in shard_dirs], ignore_index=True)
    splits = generate_group_splits(sorted(transitions["split_group_id"].unique().tolist()))
    ood_trajectories = generate_policy_transfer_splits(transitions)[POLICY_TRANSFER_DIRECTION]["test"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # These untrained instances are used only to derive Phase 3's architecture
    # parameter budgets. They are never trained or evaluated.
    variants = build_all_phase4_variants(
        _build_baseline_spec("flat_h4_control"),
        _build_baseline_spec("feature_token_predictor"),
        history_len=HISTORY_LEN,
        max_horizon=HORIZON,
    )
    check_all_variants_capacity_budget(variants)
    if not args.skip_vram_preflight:
        run_vram_preflight(variants, device, batch_size=BATCH_SIZE, history_len=HISTORY_LEN, horizon=HORIZON)

    all_runs: list[dict[str, Any]] = []
    for seed in PHASE3_SEEDS:
        generator = set_deterministic_seed(seed)
        train_ds = CyberJEPADataset(shard_dirs, splits["train"], horizon=HORIZON, history_len=HISTORY_LEN, fit_normalizers=True)
        normalizers = train_ds.normalizer_stats
        val_ds = CyberJEPADataset(shard_dirs, splits["val"], horizon=HORIZON, history_len=HISTORY_LEN, normalizer_stats=normalizers)
        test_ds = CyberJEPADataset(shard_dirs, splits["test"], horizon=HORIZON, history_len=HISTORY_LEN, normalizer_stats=normalizers)
        ood_ds = CyberJEPADataset(shard_dirs, trajectory_set=ood_trajectories, horizon=HORIZON, history_len=HISTORY_LEN, normalizer_stats=normalizers)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=generator, worker_init_fn=seed_worker)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        ood_loader = DataLoader(ood_ds, batch_size=BATCH_SIZE, shuffle=False)

        for variant in variants:
            model, _ = train_variant(variant, train_loader, val_loader, device, args.output_dir / f"seed_{seed}", max_epochs=MAX_EPOCHS, min_epochs=MIN_EPOCHS, patience=PATIENCE, lr=1e-3, weight_decay=1e-4)
            result = evaluate_variant(model, variant, train_loader, test_loader, ood_loader, device, seed)
            all_runs.append(result)
            (args.output_dir / f"{variant.name}_seed{seed}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    (args.output_dir / "phase4_runs.json").write_text(json.dumps(all_runs, indent=2), encoding="utf-8")
    _write_comparison(all_runs, phase3_rows, args.output_dir)
    build_selection_reports(all_runs, [], args.output_dir)


if __name__ == "__main__":
    main()
