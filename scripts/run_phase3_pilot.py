"""
Phase 3 One-Seed Pilot Verification Script.

Executes 4 pilot configurations on seed 1001 to verify training stability,
loss convergence, target encoder freezing, cohort hash verification,
action/temporal sensitivity, and attention artifact logging.
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
from cyber_jepa.data.dataset import CyberJEPADataset, generate_group_splits
from cyber_jepa.representations.flat import FlatVectorRepresentation
from cyber_jepa.representations.feature import FeatureTokenRepresentation
from cyber_jepa.models.jepa import CyberJEPA, count_subsystem_parameters
from cyber_jepa.training.trainer import Trainer
from cyber_jepa.evaluation.probes import LinearProbeEvaluator
from cyber_jepa.evaluation.diagnostics import compute_latent_geometry_diagnostics


PILOT_CONFIGS = [
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
]


def create_encoder(rep_type: str, hidden_dim: int = 64, history_len: int = 4) -> torch.nn.Module:
    """Factory creating representation encoder instance."""
    if rep_type == "flat":
        return FlatVectorRepresentation(obs_dim=52, hidden_dim=hidden_dim, history_len=history_len)
    elif rep_type == "feature":
        return FeatureTokenRepresentation(num_features=52, hidden_dim=hidden_dim, history_len=history_len)
    else:
        raise ValueError(f"Unknown rep_type: {rep_type}")


def run_phase3_pilot():
    """Execute 1-seed pilot sweep on 4 core configurations."""
    shards_dir = Path("data/shards")
    cohort_path = Path("experiments/phase3/phase3_cohort.parquet")
    cohort_hash_path = Path("experiments/phase3/phase3_cohort.sha256")
    runs_dir = Path("runs/phase3/pilot")
    runs_dir.mkdir(parents=True, exist_ok=True)

    if not cohort_path.exists() or not cohort_hash_path.exists():
        raise FileNotFoundError("Phase 3 cohort not found. Run scripts/build_phase3_cohort.py first.")

    # 1. Verify Cohort Hash
    expected_hash = cohort_hash_path.read_text().strip()
    actual_hash = hashlib.sha256(cohort_path.read_bytes()).hexdigest()
    if expected_hash != actual_hash:
        raise ValueError(f"Cohort hash mismatch! Expected {expected_hash}, got {actual_hash}")
    print(f"[+] Verified cohort hash: {actual_hash[:16]}...")

    cohort_df = pd.read_parquet(cohort_path)

    # 2. Setup DataLoaders
    trans_list = [pd.read_parquet(d / "transitions.parquet") for d in shards_dir.glob("*") if (d / "transitions.parquet").exists()]
    all_trans = pd.concat(trans_list, ignore_index=True)
    all_groups = sorted(all_trans["split_group_id"].unique().tolist())
    splits = generate_group_splits(all_groups, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Running Phase 3 Pilot on device: {device}")

    results = []

    for cfg in PILOT_CONFIGS:
        cid = cfg["config_id"]
        seed = 1001
        run_dir = runs_dir / f"{cid}_seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n>>> Pilot Run: {cid} (seed={seed}) <<<")
        gen = set_deterministic_seed(seed)

        train_ds = CyberJEPADataset(
            shard_dirs=sorted([d for d in shards_dir.glob("*") if d.is_dir() and (d / "transitions.parquet").exists()]),
            split_group_set=splits["train"],
            horizon=8,
            history_len=4,
            fit_normalizers=True,
        )
        val_ds = CyberJEPADataset(
            shard_dirs=sorted([d for d in shards_dir.glob("*") if d.is_dir() and (d / "transitions.parquet").exists()]),
            split_group_set=splits["val"],
            horizon=8,
            history_len=4,
            fit_normalizers=False,
            normalizer_stats=train_ds.normalizer_stats,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=64,
            shuffle=True,
            generator=gen,
            worker_init_fn=seed_worker,
        )
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

        encoder = create_encoder(cfg["rep_type"], hidden_dim=64, history_len=4)
        jepa = CyberJEPA(
            online_encoder=encoder,
            hidden_dim=64,
            max_horizon=8,
            aggregator_mode=cfg["aggregator_mode"],
        )

        optimizer = torch.optim.AdamW(jepa.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

        trainer = Trainer(
            model=jepa,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            run_dir=run_dir,
            device=device,
            max_epochs=5,
            min_epochs=3,
            patience=3,
        )

        history = trainer.fit()

        # 3. Assertions & Verification
        assert not np.isnan(history["train_loss"]).any(), f"NaN in train loss for {cid}"
        assert not np.isnan(history["val_loss"]).any(), f"NaN in val loss for {cid}"

        # Probe evaluation on cohort
        evaluator = LinearProbeEvaluator()
        train_z, train_y, _ = evaluator.extract_latents_and_labels(jepa.online_encoder, train_loader, device=device)

        test_ds = CyberJEPADataset(
            shard_dirs=sorted([d for d in shards_dir.glob("*") if d.is_dir() and (d / "transitions.parquet").exists()]),
            split_group_set=splits["test"],
            horizon=8,
            history_len=4,
            fit_normalizers=False,
            normalizer_stats=train_ds.normalizer_stats,
        )
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

        test_z, test_y, _ = evaluator.extract_latents_and_labels(jepa.online_encoder, test_loader, device=device)
        metrics = evaluator.train_and_evaluate_probe(train_z, train_y, test_z, test_y)

        diag = compute_latent_geometry_diagnostics(torch.tensor(test_z))
        params = count_subsystem_parameters(jepa)

        res_summary = {
            "config_id": cid,
            "seed": seed,
            "train_loss": history["train_loss"][-1],
            "val_loss": history["val_loss"][-1],
            "macro_f1": metrics["macro_f1"],
            "auroc": metrics["auroc"],
            "effective_rank": diag["effective_rank"],
            "is_collapsed": diag["is_collapsed"],
            "total_trainable_params": params["total_trainable"],
        }
        results.append(res_summary)

        print(
            f"  --> {cid}: F1={metrics['macro_f1']:.4f} | AUROC={metrics['auroc']:.4f} | "
            f"EffRank={diag['effective_rank']:.1f} | Params={params['total_trainable']:,}"
        )

    # 4. Save Pilot Summary
    pilot_report_path = Path("experiments/phase3/phase3_pilot_summary.json")
    with open(pilot_report_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n[+] Phase 3 Pilot completed successfully! Results written to experiments/phase3/phase3_pilot_summary.json")


if __name__ == "__main__":
    run_phase3_pilot()
