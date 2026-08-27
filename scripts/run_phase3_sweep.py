"""Run the reproducible Phase 3 flat/feature separate-action baseline sweep.

The archived nine-condition script cannot run because its host and hierarchical
representation modules are absent. This restored runner executes the four available
baseline conditions needed to compare Phase 4 fused action pathways.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from cyber_jepa.data.dataset import CyberJEPADataset, generate_group_splits, generate_policy_transfer_splits
from cyber_jepa.evaluation.diagnostics import compute_latent_geometry_diagnostics
from cyber_jepa.evaluation.probes import LinearProbeEvaluator
from cyber_jepa.models.jepa import CyberJEPA, count_subsystem_parameters
from cyber_jepa.representations.feature import FeatureTokenRepresentation
from cyber_jepa.representations.flat import FlatVectorRepresentation
from cyber_jepa.training.trainer import Trainer
from cyber_jepa.utils.reproducibility import seed_worker, set_deterministic_seed

SEEDS = [1001, 2003, 3005, 4007, 5009]
CONFIGS = [
    ("flat_h4_control", FlatVectorRepresentation, "legacy_last_step_mean"),
    ("feature_legacy_mean", FeatureTokenRepresentation, "legacy_last_step_mean"),
    ("feature_query_pool", FeatureTokenRepresentation, "learned_query_pool"),
    ("feature_token_predictor", FeatureTokenRepresentation, "token_preserving_predictor"),
]


def run_phase3_sweep(
    seeds: list[int] = SEEDS, max_epochs: int = 20, min_epochs: int = 10, patience: int = 5,
) -> None:
    shards_dir, cohort = Path("data/shards"), Path("experiments/phase3/phase3_cohort.parquet")
    cohort_hash = cohort.with_suffix(".sha256")
    if not cohort.exists() or not cohort_hash.exists():
        raise FileNotFoundError("Run scripts/build_phase3_cohort.py first.")
    if hashlib.sha256(cohort.read_bytes()).hexdigest() != cohort_hash.read_text(encoding="utf-8").strip():
        raise ValueError("Phase 3 cohort checksum mismatch.")
    shard_dirs = sorted(p for p in shards_dir.iterdir() if (p / "transitions.parquet").exists())
    if not shard_dirs:
        raise FileNotFoundError("No collected shards found. Run scripts/run_collection.py first.")
    transitions = pd.concat([pd.read_parquet(p / "transitions.parquet") for p in shard_dirs], ignore_index=True)
    splits = generate_group_splits(sorted(transitions["split_group_id"].unique().tolist()))
    ood_trajectories = generate_policy_transfer_splits(transitions)["bline_to_meander"]["test"]
    run_root = Path("runs/phase3/sweep"); run_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []

    for config_id, encoder_cls, aggregator_mode in CONFIGS:
        for seed in seeds:
            run_dir = run_root / f"{config_id}_seed{seed}"
            metrics_path = run_dir / "eval_metrics.json"
            if metrics_path.exists():
                results.append(json.loads(metrics_path.read_text(encoding="utf-8"))); continue
            generator = set_deterministic_seed(seed)
            train_ds = CyberJEPADataset(shard_dirs, splits["train"], horizon=8, history_len=4, fit_normalizers=True)
            normalizers = train_ds.normalizer_stats
            val_ds = CyberJEPADataset(shard_dirs, splits["val"], horizon=8, history_len=4, normalizer_stats=normalizers)
            test_ds = CyberJEPADataset(shard_dirs, splits["test"], horizon=8, history_len=4, normalizer_stats=normalizers)
            ood_ds = CyberJEPADataset(shard_dirs, trajectory_set=ood_trajectories, horizon=8, history_len=4, normalizer_stats=normalizers)
            train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, generator=generator, worker_init_fn=seed_worker)
            val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = CyberJEPA(encoder_cls(obs_dim=52, hidden_dim=64, history_len=4), hidden_dim=64, max_horizon=8, aggregator_mode=aggregator_mode).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            trainer = Trainer(model, train_loader, val_loader, optimizer, torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs), run_dir, device, max_epochs=max_epochs, min_epochs=min_epochs, patience=patience)
            history = trainer.fit()
            probe = LinearProbeEvaluator()
            train_z, train_y, _ = probe.extract_latents_and_labels(model.online_encoder, train_loader, device)
            test_z, test_y, _ = probe.extract_latents_and_labels(model.online_encoder, DataLoader(test_ds, batch_size=64), device)
            ood_z, ood_y, _ = probe.extract_latents_and_labels(model.online_encoder, DataLoader(ood_ds, batch_size=64), device)
            standard, ood = probe.train_and_evaluate_probe(train_z, train_y, test_z, test_y), probe.train_and_evaluate_probe(train_z, train_y, ood_z, ood_y)
            diagnostic, params = compute_latent_geometry_diagnostics(torch.tensor(test_z)), count_subsystem_parameters(model)
            result = {"config_id": config_id, "seed": seed, "rep_type": "flat" if encoder_cls is FlatVectorRepresentation else "feature", "aggregator_mode": aggregator_mode, "train_loss": history["train_loss"][-1], "val_loss": history["val_loss"][-1], "standard_macro_f1": standard["macro_f1"], "standard_auroc": standard["auroc"], "policy_transfer_macro_f1": ood["macro_f1"], "policy_transfer_auroc": ood["auroc"], "effective_rank": diagnostic["effective_rank"], "is_collapsed": diagnostic["is_collapsed"], "trainable_params": params["total_trainable"], "non_trainable_ema_params": params["total_non_trainable_ema"]}
            run_dir.mkdir(parents=True, exist_ok=True); metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8"); results.append(result)
    Path("experiments/phase3").mkdir(parents=True, exist_ok=True)
    Path("experiments/phase3/phase3_sweep_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run_phase3_sweep()
