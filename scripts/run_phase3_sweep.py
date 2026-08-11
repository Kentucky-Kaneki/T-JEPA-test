"""
Phase 3 Core Experiment Sweep Script.

Executes the 45-run sweep (9 configurations x 5 seeds) evaluating representation fairness,
context aggregation, single-frame target encoding, and policy-transfer robustness.
"""

import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cyber_jepa.utils.reproducibility import set_deterministic_seed, seed_worker, Phase3SeedConfig
from cyber_jepa.data.dataset import CyberJEPADataset, generate_group_splits, generate_policy_transfer_splits
from cyber_jepa.representations.flat import FlatVectorRepresentation
from cyber_jepa.representations.feature import FeatureTokenRepresentation
from cyber_jepa.representations.host import HostTokenRepresentation
from cyber_jepa.representations.hierarchical import HierarchicalHostSubnetRepresentation
from cyber_jepa.models.jepa import CyberJEPA, count_subsystem_parameters
from cyber_jepa.training.trainer import Trainer
from cyber_jepa.evaluation.probes import LinearProbeEvaluator
from cyber_jepa.evaluation.diagnostics import compute_latent_geometry_diagnostics


SWEEP_SEEDS = [1001, 2003, 3005, 4007, 5009]

CORE_CONFIGS = [
    {
        "config_id": "flat_h4_control",
        "rep_type": "flat",
        "aggregator_mode": "legacy_last_step_mean",
    },
    {
        "config_id": "feature_legacy_mean",
        "rep_type": "feature",
        "aggregator_mode": "legacy_last_step_mean",
    },
    {
        "config_id": "feature_query_pool",
        "rep_type": "feature",
        "aggregator_mode": "learned_query_pool",
    },
    {
        "config_id": "feature_token_predictor",
        "rep_type": "feature",
        "aggregator_mode": "token_preserving_predictor",
    },
    {
        "config_id": "host_legacy_mean",
        "rep_type": "host",
        "aggregator_mode": "legacy_last_step_mean",
    },
    {
        "config_id": "host_query_pool",
        "rep_type": "host",
        "aggregator_mode": "learned_query_pool",
    },
    {
        "config_id": "host_token_predictor",
        "rep_type": "host",
        "aggregator_mode": "token_preserving_predictor",
    },
    {
        "config_id": "hierarchical_current",
        "rep_type": "hierarchical",
        "aggregator_mode": "legacy_last_step_mean",
    },
    {
        "config_id": "hierarchical_masked_predictor",
        "rep_type": "hierarchical",
        "aggregator_mode": "token_preserving_predictor",
    },
]


def create_encoder(rep_type: str, hidden_dim: int = 64, history_len: int = 4) -> torch.nn.Module:
    """Factory creating representation encoder instance."""
    if rep_type == "flat":
        return FlatVectorRepresentation(obs_dim=52, hidden_dim=hidden_dim, history_len=history_len)
    elif rep_type == "feature":
        return FeatureTokenRepresentation(num_features=52, hidden_dim=hidden_dim, history_len=history_len)
    elif rep_type == "host":
        return HostTokenRepresentation(num_hosts=13, hidden_dim=hidden_dim, history_len=history_len)
    elif rep_type == "hierarchical":
        return HierarchicalHostSubnetRepresentation(num_hosts=13, num_subnets=3, hidden_dim=hidden_dim, history_len=history_len)
    else:
        raise ValueError(f"Unknown rep_type: {rep_type}")


def run_phase3_sweep():
    """Execute 45-run Phase 3 experiment sweep."""
    shards_dir = Path("data/shards")
    cohort_path = Path("experiments/phase3/phase3_cohort.parquet")
    cohort_hash_path = Path("experiments/phase3/phase3_cohort.sha256")
    runs_dir = Path("runs/phase3/sweep")
    runs_dir.mkdir(parents=True, exist_ok=True)

    if not cohort_path.exists() or not cohort_hash_path.exists():
        raise FileNotFoundError("Phase 3 cohort not found. Run scripts/build_phase3_cohort.py first.")

    # 1. Verify Cohort Hash
    expected_hash = cohort_hash_path.read_text().strip()
    actual_hash = hashlib.sha256(cohort_path.read_bytes()).hexdigest()
    if expected_hash != actual_hash:
        raise ValueError(f"Cohort hash mismatch! Expected {expected_hash}, got {actual_hash}")
    print(f"[+] Verified cohort hash: {actual_hash[:16]}...")

    # 2. Split Setup
    trans_list = [pd.read_parquet(d / "transitions.parquet") for d in shards_dir.glob("*") if (d / "transitions.parquet").exists()]
    all_trans = pd.concat(trans_list, ignore_index=True)
    all_groups = sorted(all_trans["split_group_id"].unique().tolist())
    standard_splits = generate_group_splits(all_groups, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    policy_splits = generate_policy_transfer_splits(all_trans)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Running Phase 3 Core Sweep on device: {device} ({len(CORE_CONFIGS) * len(SWEEP_SEEDS)} total runs)")

    all_sweep_results = []

    for cfg in CORE_CONFIGS:
        cid = cfg["config_id"]
        for seed in SWEEP_SEEDS:
            run_dir = runs_dir / f"{cid}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            if (run_dir / "eval_metrics.json").exists():
                print(f"[*] Skipping completed run: {cid}_seed{seed}")
                with open(run_dir / "eval_metrics.json") as f:
                    all_sweep_results.append(json.load(f))
                continue

            print(f"\n==================================================")
            print(f"   Executing Run: {cid} (seed={seed})")
            print(f"==================================================")

            gen = set_deterministic_seed(seed)

            train_ds = CyberJEPADataset(
                shard_dirs=sorted([d for d in shards_dir.glob("*") if d.is_dir() and (d / "transitions.parquet").exists()]),
                split_group_set=standard_splits["train"],
                horizon=8,
                history_len=4,
                fit_normalizers=True,
            )
            val_ds = CyberJEPADataset(
                shard_dirs=sorted([d for d in shards_dir.glob("*") if d.is_dir() and (d / "transitions.parquet").exists()]),
                split_group_set=standard_splits["val"],
                horizon=8,
                history_len=4,
                fit_normalizers=False,
                normalizer_stats=train_ds.normalizer_stats,
            )
            test_ds = CyberJEPADataset(
                shard_dirs=sorted([d for d in shards_dir.glob("*") if d.is_dir() and (d / "transitions.parquet").exists()]),
                split_group_set=standard_splits["test"],
                horizon=8,
                history_len=4,
                fit_normalizers=False,
                normalizer_stats=train_ds.normalizer_stats,
            )
            policy_ood_ds = CyberJEPADataset(
                shard_dirs=sorted([d for d in shards_dir.glob("*") if d.is_dir() and (d / "transitions.parquet").exists()]),
                trajectory_set=policy_splits["bline_to_meander"]["test"],
                horizon=8,
                history_len=4,
                fit_normalizers=False,
                normalizer_stats=train_ds.normalizer_stats,
            )

            train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, generator=gen, worker_init_fn=seed_worker)
            val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
            policy_ood_loader = DataLoader(policy_ood_ds, batch_size=64, shuffle=False)

            encoder = create_encoder(cfg["rep_type"], hidden_dim=64, history_len=4)
            jepa = CyberJEPA(
                online_encoder=encoder,
                hidden_dim=64,
                max_horizon=8,
                aggregator_mode=cfg["aggregator_mode"],
            )

            optimizer = torch.optim.AdamW(jepa.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

            trainer = Trainer(
                model=jepa,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                run_dir=run_dir,
                device=device,
                max_epochs=20,
                min_epochs=10,
                patience=5,
            )

            history = trainer.fit()

            # Evaluations
            evaluator = LinearProbeEvaluator()

            # 1. Standard Test Probe
            train_z, train_y, _ = evaluator.extract_latents_and_labels(jepa.online_encoder, train_loader, device=device)
            test_z, test_y, _ = evaluator.extract_latents_and_labels(jepa.online_encoder, test_loader, device=device)
            test_metrics = evaluator.train_and_evaluate_probe(train_z, train_y, test_z, test_y)

            # 2. Policy-Transfer OOD Probe
            ood_z, ood_y, _ = evaluator.extract_latents_and_labels(jepa.online_encoder, policy_ood_loader, device=device)
            ood_metrics = evaluator.train_and_evaluate_probe(train_z, train_y, ood_z, ood_y)

            diag = compute_latent_geometry_diagnostics(torch.tensor(test_z))
            params = count_subsystem_parameters(jepa)

            run_summary = {
                "config_id": cid,
                "seed": seed,
                "rep_type": cfg["rep_type"],
                "aggregator_mode": cfg["aggregator_mode"],
                "train_loss": float(history["train_loss"][-1]),
                "val_loss": float(history["val_loss"][-1]),
                "standard_macro_f1": float(test_metrics["macro_f1"]),
                "standard_auroc": float(test_metrics["auroc"]),
                "policy_transfer_macro_f1": float(ood_metrics["macro_f1"]),
                "policy_transfer_auroc": float(ood_metrics["auroc"]),
                "effective_rank": float(diag["effective_rank"]),
                "is_collapsed": bool(diag["is_collapsed"]),
                "trainable_params": params["total_trainable"],
                "non_trainable_ema_params": params["total_non_trainable_ema"],
            }

            with open(run_dir / "eval_metrics.json", "w") as f:
                json.dump(run_summary, f, indent=2)

            all_sweep_results.append(run_summary)

            print(
                f"  --> [{cid}_seed{seed} Complete]: Std F1={test_metrics['macro_f1']:.4f} | "
                f"OOD F1={ood_metrics['macro_f1']:.4f} | EffRank={diag['effective_rank']:.1f}"
            )

    # 3. Save Combined Sweep Results
    sweep_summary_path = Path("experiments/phase3/phase3_sweep_results.json")
    with open(sweep_summary_path, "w") as f:
        json.dump(all_sweep_results, f, indent=2)

    print(f"\n[+] Phase 3 Core Sweep complete! All 45 run metrics saved to {sweep_summary_path}")


if __name__ == "__main__":
    run_phase3_sweep()
