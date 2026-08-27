"""Re-evaluate completed Phase 4 checkpoints with the exact Phase 3 probe contract.

This is evaluation only: it neither trains models nor changes their weights. It
loads each saved ``last.pt`` checkpoint, extracts full action-blind encoder
latents from the standard and policy-transfer test loaders, and applies the same
``LinearProbeEvaluator`` and ``compute_latent_geometry_diagnostics`` used in
``scripts/run_phase3_sweep.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from cyber_jepa.data.dataset import CyberJEPADataset, generate_group_splits, generate_policy_transfer_splits
from cyber_jepa.evaluation.diagnostics import compute_latent_geometry_diagnostics
from cyber_jepa.evaluation.phase4_variants import build_all_phase4_variants
from cyber_jepa.evaluation.probes import LinearProbeEvaluator
from cyber_jepa.models.jepa import CyberJEPA
from cyber_jepa.representations.feature import FeatureTokenRepresentation
from cyber_jepa.representations.flat import FlatVectorRepresentation
from cyber_jepa.utils.reproducibility import set_deterministic_seed


SEEDS = [1001, 2003, 3005, 4007, 5009]
HORIZON = 8
HISTORY_LEN = 4
HIDDEN_DIM = 64
BATCH_SIZE = 64
POLICY_TRANSFER_DIRECTION = "bline_to_meander"


def _baseline_shape(name: str) -> CyberJEPA:
    if name == "flat_h4_control":
        encoder = FlatVectorRepresentation(obs_dim=52, hidden_dim=HIDDEN_DIM, history_len=HISTORY_LEN)
        aggregator_mode = "legacy_last_step_mean"
    elif name == "feature_token_predictor":
        encoder = FeatureTokenRepresentation(obs_dim=52, hidden_dim=HIDDEN_DIM, history_len=HISTORY_LEN)
        aggregator_mode = "token_preserving_predictor"
    else:
        raise ValueError(f"Unknown Phase 3 baseline: {name}")
    return CyberJEPA(encoder, hidden_dim=HIDDEN_DIM, max_horizon=HORIZON, aggregator_mode=aggregator_mode)


def _load_checkpoint(model: torch.nn.Module, checkpoint: Path, device: torch.device) -> None:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    for module_name in ("online_encoder", "target_encoder", "readout"):
        if module_name not in saved:
            raise KeyError(f"{checkpoint} lacks '{module_name}'.")
        getattr(model, module_name).load_state_dict(saved[module_name])


def _aggregate(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["representation"])].append(float(row[key]))
    return {name: {"per_seed": values, "mean": sum(values) / len(values)} for name, values in grouped.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-dir", type=Path, default=Path("data/shards"))
    parser.add_argument("--phase4-runs-dir", type=Path, default=Path("data/phase4_runs"))
    parser.add_argument("--phase3-results", type=Path, default=Path("experiments/phase3/phase3_sweep_results.json"))
    parser.add_argument("--output", type=Path, default=Path("data/phase4_runs/phase4_action_blind_reevaluation.json"))
    args = parser.parse_args()

    archived_phase4 = json.loads((args.phase4_runs_dir / "phase4_runs.json").read_text(encoding="utf-8"))
    if len(archived_phase4) != 20:
        raise ValueError("Expected exactly 20 completed Phase 4 runs.")
    phase3_rows = json.loads(args.phase3_results.read_text(encoding="utf-8"))
    phase3_rows = [row for row in phase3_rows if row.get("config_id") in {"flat_h4_control", "feature_token_predictor"}]
    if len(phase3_rows) != 10:
        raise ValueError("Expected ten archived Phase 3 baseline rows.")
    cohort = Path("experiments/phase3/phase3_cohort.parquet")
    cohort_hash = cohort.with_suffix(".sha256")
    if not cohort.exists() or hashlib.sha256(cohort.read_bytes()).hexdigest() != cohort_hash.read_text(encoding="utf-8").strip():
        raise ValueError("Phase 3 cohort is missing or its checksum does not match.")

    shard_dirs = sorted(path for path in args.shards_dir.iterdir() if (path / "transitions.parquet").exists())
    if len(shard_dirs) != 18:
        raise ValueError(f"Expected 18 Phase 3 shards; found {len(shard_dirs)}.")
    transitions = pd.concat([pd.read_parquet(path / "transitions.parquet") for path in shard_dirs], ignore_index=True)
    splits = generate_group_splits(sorted(transitions["split_group_id"].unique().tolist()))
    ood_trajectories = generate_policy_transfer_splits(transitions)[POLICY_TRANSFER_DIRECTION]["test"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variants = {variant.name: variant for variant in build_all_phase4_variants(
        _baseline_shape("flat_h4_control"), _baseline_shape("feature_token_predictor"),
        history_len=HISTORY_LEN, max_horizon=HORIZON,
    )}

    reevaluated: list[dict[str, Any]] = []
    for seed in SEEDS:
        set_deterministic_seed(seed)
        train_ds = CyberJEPADataset(shard_dirs, splits["train"], horizon=HORIZON, history_len=HISTORY_LEN, fit_normalizers=True)
        normalizers = train_ds.normalizer_stats
        test_ds = CyberJEPADataset(shard_dirs, splits["test"], horizon=HORIZON, history_len=HISTORY_LEN, normalizer_stats=normalizers)
        ood_ds = CyberJEPADataset(shard_dirs, trajectory_set=ood_trajectories, horizon=HORIZON, history_len=HISTORY_LEN, normalizer_stats=normalizers)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        ood_loader = DataLoader(ood_ds, batch_size=BATCH_SIZE, shuffle=False)

        for name, variant in variants.items():
            model = variant.build().to(device)
            _load_checkpoint(model, args.phase4_runs_dir / f"seed_{seed}" / name / "last.pt", device)
            model.eval()
            probe = LinearProbeEvaluator()
            train_latents, train_labels, _ = probe.extract_latents_and_labels(model.online_encoder, train_loader, device)
            test_latents, test_labels, _ = probe.extract_latents_and_labels(model.online_encoder, test_loader, device)
            ood_latents, ood_labels, _ = probe.extract_latents_and_labels(model.online_encoder, ood_loader, device)
            standard = probe.train_and_evaluate_probe(train_latents, train_labels, test_latents, test_labels)
            policy_transfer = probe.train_and_evaluate_probe(train_latents, train_labels, ood_latents, ood_labels)
            geometry = compute_latent_geometry_diagnostics(torch.tensor(test_latents))
            reevaluated.append({
                "representation": name,
                "seed": seed,
                "checkpoint": "last.pt",
                "standard_macro_f1": standard["macro_f1"],
                "policy_transfer_macro_f1": policy_transfer["macro_f1"],
                "standard_probe": standard,
                "policy_transfer_probe": policy_transfer,
                "action_blind_encoder_geometry": geometry,
            })

    payload = {
        "evaluation_contract": {
            "checkpoint": "last.pt",
            "geometry_source": "full action-blind standard-test online-encoder latents",
            "probe_contract": "same LinearProbeEvaluator and loaders as Phase 3",
            "horizon": HORIZON,
            "history_len": HISTORY_LEN,
            "policy_transfer_direction": POLICY_TRANSFER_DIRECTION,
        },
        "phase3_archived": phase3_rows,
        "phase4_action_blind": reevaluated,
        "phase3_policy_transfer_summary": _aggregate(
            [{"representation": row["config_id"], "metric": row["policy_transfer_macro_f1"]} for row in phase3_rows], "metric"
        ),
        "phase4_policy_transfer_summary": _aggregate(reevaluated, "policy_transfer_macro_f1"),
        "phase4_standard_summary": _aggregate(reevaluated, "standard_macro_f1"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved Phase 3-matched Phase 4 re-evaluation to {args.output}")


if __name__ == "__main__":
    main()
