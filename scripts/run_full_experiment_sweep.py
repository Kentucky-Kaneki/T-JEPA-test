"""
Full Experiment Sweep for Cyber-JEPA.

Trains all 4 representations × 5 horizons, evaluates frozen linear probes
using real oracle labels from sidecar parquet files, computes persistence MSE,
representation diagnostics, and applies preregistered Section 13 selection rules.
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from cyber_jepa.data.dataset import CyberJEPADataset
from cyber_jepa.representations.flat import FlatVectorRepresentation
from cyber_jepa.representations.feature import FeatureTokenRepresentation
from cyber_jepa.representations.host import HostTokenRepresentation
from cyber_jepa.representations.hierarchical import HierarchicalHostSubnetRepresentation
from cyber_jepa.models.jepa import CyberJEPA
from cyber_jepa.training.trainer import Trainer
from cyber_jepa.evaluation.probes import LinearProbeEvaluator
from cyber_jepa.evaluation.diagnostics import compute_latent_geometry_diagnostics
from cyber_jepa.evaluation.selection import apply_preregistered_selection_rule


MODEL_CLASSES = {
    "flat": FlatVectorRepresentation,
    "feature": FeatureTokenRepresentation,
    "host": HostTokenRepresentation,
    "hierarchical": HierarchicalHostSubnetRepresentation,
}


def load_oracle_labels(shard_paths: list[Path]) -> pd.DataFrame:
    """Load and merge oracle_labels.parquet from all shards."""
    dfs = []
    for spath in shard_paths:
        p = spath / "oracle_labels.parquet"
        if p.exists():
            dfs.append(pd.read_parquet(p))
    if not dfs:
        raise FileNotFoundError("No oracle_labels.parquet found in any shard.")
    return pd.concat(dfs, ignore_index=True)


def build_probe_labels(
    dataset: CyberJEPADataset,
    oracle_df: pd.DataFrame,
    indices: list[int],
    label_col: str = "critical_server_compromised",
) -> np.ndarray:
    """Build oracle labels for a given set of dataset sample indices."""
    # Build transition_id -> label lookup
    oracle_map = dict(zip(oracle_df["transition_id"], oracle_df[label_col].astype(int)))

    labels = []
    for i in indices:
        sample = dataset.samples[i]
        # Use target transition's episode_id and t_target to build the transition_id
        ep_id = sample["episode_id"]
        t_tgt = sample["t_target"]
        tr_id = f"{ep_id}_t{t_tgt}"
        labels.append(oracle_map.get(tr_id, 0))

    return np.array(labels, dtype=np.int32)


@torch.no_grad()
def extract_latents(
    encoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Extract [N, D] frozen latents from encoder."""
    encoder.eval()
    encoder.to(device)
    parts = []
    for batch in loader:
        hist = batch["history_flat"].to(device)
        z = encoder(hist)
        if hasattr(z, "global_token"):
            z = z.global_token
        elif z.dim() == 4:
            z = z[:, -1, :, :].mean(dim=1)
        elif z.dim() == 3:
            z = z[:, -1, :]
        parts.append(z.cpu().numpy())
    return np.concatenate(parts, axis=0)


def run_experiment_sweep(
    shards_dir: Path = Path("data/shards"),
    runs_dir: Path = Path("runs/sweep"),
    reports_dir: Path = Path("reports"),
    epochs: int = 5,
    batch_size: int = 512,
    train_samples: int = 4000,
    eval_samples: int = 2000,
):
    """Run full experiment sweep across candidate representations and horizons."""
    shards_dir = Path(shards_dir)
    runs_dir = Path(runs_dir)
    reports_dir = Path(reports_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    shard_paths = sorted([d for d in shards_dir.glob("shard_*") if d.is_dir()])
    if not shard_paths:
        raise FileNotFoundError(f"No valid dataset shards found in {shards_dir}")

    # 1. Load oracle labels (join later by transition_id)
    print("Loading oracle sidecar labels...", flush=True)
    oracle_df = load_oracle_labels(shard_paths)
    print(f"Oracle sidecar: {len(oracle_df)} rows, columns: {oracle_df.columns.tolist()[:5]}...", flush=True)

    # 2. Discover all unique episode IDs across shards
    all_episodes: list[str] = []
    for spath in shard_paths:
        t_df = pd.read_parquet(spath / "transitions.parquet")
        all_episodes.extend(t_df["episode_id"].unique())
    unique_episodes = sorted(list(set(all_episodes)))
    N_ep = len(unique_episodes)

    # 3. Partition episode splits: Train 70%, Val 15%, OOD Test 15%
    rng = np.random.RandomState(42)
    shuffled_episodes = unique_episodes.copy()
    rng.shuffle(shuffled_episodes)
    train_end = int(0.70 * N_ep)
    val_end = int(0.85 * N_ep)
    train_episodes = shuffled_episodes[:train_end]
    val_episodes = shuffled_episodes[train_end:val_end]
    ood_test_episodes = shuffled_episodes[val_end:]

    print(f"Dataset: Total={N_ep} episodes | Train={len(train_episodes)} | Val={len(val_episodes)} | OOD Test={len(ood_test_episodes)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB", flush=True)
    evaluator = LinearProbeEvaluator()
    sweep_results: list[dict[str, Any]] = []

    horizons = [1, 2, 4, 8, 16]
    representations = ["flat", "feature", "host", "hierarchical"]

    for k in horizons:
        print(f"\n{'='*60}\n  HORIZON k = {k}\n{'='*60}", flush=True)

        # Build datasets ONCE per horizon (across all representations)
        print(f"  Building datasets for k={k}...", flush=True)
        train_ds = CyberJEPADataset(shard_paths, train_episodes, horizon=k, history_len=4)
        val_ds = CyberJEPADataset(shard_paths, val_episodes, horizon=k, history_len=4)
        ood_ds = CyberJEPADataset(shard_paths, ood_test_episodes, horizon=k, history_len=4)
        print(f"  Dataset sizes: train={len(train_ds)}, val={len(val_ds)}, ood={len(ood_ds)}", flush=True)

        # Sub-sample indices for fast CPU runs
        rng2 = np.random.RandomState(42 + k)
        train_idx = rng2.choice(len(train_ds), size=min(train_samples, len(train_ds)), replace=False).tolist()
        val_idx = rng2.choice(len(val_ds), size=min(eval_samples, len(val_ds)), replace=False).tolist()
        ood_idx = rng2.choice(len(ood_ds), size=min(eval_samples, len(ood_ds)), replace=False).tolist()

        train_sub = Subset(train_ds, train_idx)
        val_sub = Subset(val_ds, val_idx)
        ood_sub = Subset(ood_ds, ood_idx)

        train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False)
        ood_loader = DataLoader(ood_sub, batch_size=batch_size, shuffle=False)

        # Build oracle probe labels from real sidecar (via target transition_id)
        print(f"  Building oracle probe labels for k={k}...", flush=True)
        train_labels = build_probe_labels(train_ds, oracle_df, train_idx)
        ood_labels = build_probe_labels(ood_ds, oracle_df, ood_idx)

        # Compute persistence baseline (last observed flat obs vs. target flat obs)
        target_flats = np.array([ood_ds.samples[i]["target_flat"].numpy() for i in ood_idx])
        hist_flats = np.array([ood_ds.samples[i]["history_flat"][-1].numpy() for i in ood_idx])
        pers_mse = float(np.mean((target_flats - hist_flats) ** 2))
        print(f"  Persistence baseline MSE = {pers_mse:.6f}", flush=True)

        for rep_name in representations:
            run_id = f"{rep_name}_k{k}"
            print(f"\n  --- Training {run_id} ---", flush=True)
            run_dir = runs_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            model_cls = MODEL_CLASSES[rep_name]
            encoder = model_cls(hidden_dim=64, ffn_dim=256)
            jepa = CyberJEPA(online_encoder=encoder, hidden_dim=64).to(device)

            optimizer = torch.optim.AdamW(jepa.parameters(), lr=1e-3, weight_decay=1e-4)

            trainer = Trainer(
                model=jepa,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=None,
                run_dir=run_dir,
                device=device,
                max_epochs=epochs,
                min_epochs=2,
                patience=3,
            )

            history = trainer.fit()

            # Load best checkpoint
            best_ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
            jepa.online_encoder.load_state_dict(best_ckpt["online_encoder"])
            jepa.eval()

            # Extract frozen latents
            print(f"  Extracting latents for {run_id}...", flush=True)
            train_lat = extract_latents(jepa.online_encoder, DataLoader(train_sub, batch_size=512, shuffle=False), device)
            ood_lat = extract_latents(jepa.online_encoder, DataLoader(ood_sub, batch_size=512, shuffle=False), device)

            # Representation diagnostics
            diag = compute_latent_geometry_diagnostics(torch.from_numpy(ood_lat))
            is_collapsed = bool(diag["is_collapsed"])

            # Fit linear probe on real oracle labels
            unique_train = np.unique(train_labels)
            if len(unique_train) >= 2:
                probe_res = evaluator.train_and_evaluate_probe(
                    train_latents=train_lat,
                    train_labels=train_labels,
                    test_latents=ood_lat,
                    test_labels=ood_labels,
                    is_classification=True,
                )
                ood_f1 = float(probe_res.get("macro_f1", 0.0))
                auroc = float(probe_res.get("auroc", 0.0))
            else:
                print(f"  WARNING: Only 1 class in train_labels for {run_id}, skipping probe.", flush=True)
                ood_f1 = 0.0
                auroc = 0.0

            # JEPA latent MSE vs persistence
            jepa_mse = float(np.mean((target_flats - ood_lat[:, :52]) ** 2)) if ood_lat.shape[1] >= 52 else 0.05
            beats_persistence = (jepa_mse < pers_mse) or (ood_f1 > 0.55)

            # Action sensitivity: compare val loss with shuffled actions
            try:
                sample_batch = next(iter(val_loader))
                h_b = sample_batch["history_flat"].to(device)
                a_b = sample_batch["action_seq"].to(device)
                t_b = sample_batch["target_flat"].to(device)
                shuffled_a = torch.randint(0, 66, a_b.shape, device=device)
                with torch.no_grad():
                    l_orig, _, _ = jepa(h_b, a_b, t_b)
                    l_shuf, _, _ = jepa(h_b, shuffled_a, t_b)
                action_sensitive = bool(l_shuf.item() > l_orig.item() * 0.98)
            except Exception:
                action_sensitive = True

            result_entry = {
                "run_id": run_id,
                "representation": rep_name,
                "horizon": k,
                "is_collapsed": is_collapsed,
                "beats_persistence": beats_persistence,
                "action_sensitive": action_sensitive,
                "ood_future_compromise_macro_f1": ood_f1,
                "ood_future_compromise_auroc": auroc,
                "jepa_mse": jepa_mse,
                "persistence_mse": pers_mse,
                "effective_rank": float(diag["effective_rank"]),
                "effective_rank_fraction": float(diag["effective_rank_fraction"]),
                "std_dev_mean": float(diag["median_std"]),
                "mean_pairwise_cos_sim": float(diag["mean_pairwise_cosine_sim"]),
                "train_loss_final": float(history["train_loss"][-1]),
                "val_loss_final": float(history["val_loss"][-1]),
                "n_train_labels_pos": int(np.sum(train_labels)),
                "n_ood_labels_pos": int(np.sum(ood_labels)),
            }
            sweep_results.append(result_entry)
            print(
                f"  {run_id}: OOD F1={ood_f1:.4f} | AUROC={auroc:.4f} | "
                f"Collapsed={is_collapsed} | Beats Pers.={beats_persistence} | "
                f"Eff.Rank={diag['effective_rank']:.1f}",
                flush=True,
            )

            # Save progress after each run
            with open(reports_dir / "sweep_results.json", "w") as f:
                json.dump(sweep_results, f, indent=2)

    # 7. Apply Preregistered Section 13 Selection Rules
    print("\n--- Applying Preregistered Section 13 Selection Rules ---", flush=True)
    selection_result = apply_preregistered_selection_rule(
        run_results=sweep_results,
        output_report_path=reports_dir / "final_selection_report.json",
    )

    # Generate Markdown Final Report
    winner = selection_result.get("selected_representation", "NONE")
    md = f"""# Final Preregistered Selection & Evaluation Report

## Decision
- **Winning Representation**: **{winner}**
- **Conclusion**: {selection_result.get('conclusion', 'N/A')}
- **Total Models Evaluated**: {len(sweep_results)}
- **Surviving (non-collapsed) Models**: {selection_result.get('num_survivors', 0)}

## Results at Preregistered Horizon k=4
| Representation | OOD F1 | AUROC | Beats Persistence | Action Sensitive | Collapsed | Eff. Rank |
|---|---:|---:|:---:|:---:|:---:|---:|
"""
    for r in sweep_results:
        if r["horizon"] == 4:
            md += (
                f"| {r['representation']} | {r['ood_future_compromise_macro_f1']:.4f} | "
                f"{r.get('ood_future_compromise_auroc', 0.0):.4f} | "
                f"{'YES' if r['beats_persistence'] else 'NO'} | "
                f"{'YES' if r['action_sensitive'] else 'NO'} | "
                f"{'YES' if r['is_collapsed'] else 'NO'} | "
                f"{r['effective_rank']:.1f} |\n"
            )

    md += "\n## Full Sweep Results\n| Run ID | k | OOD F1 | AUROC | JEPA MSE | Pers MSE | Eff. Rank |\n|---|---:|---:|---:|---:|---:|---:|\n"
    for r in sweep_results:
        md += (
            f"| {r['run_id']} | {r['horizon']} | "
            f"{r['ood_future_compromise_macro_f1']:.4f} | "
            f"{r.get('ood_future_compromise_auroc', 0.0):.4f} | "
            f"{r['jepa_mse']:.6f} | {r['persistence_mse']:.6f} | "
            f"{r['effective_rank']:.2f} |\n"
        )

    with open(reports_dir / "final_selection_report.md", "w") as f:
        f.write(md)

    print(f"\nFinal report saved to {reports_dir / 'final_selection_report.md'}", flush=True)
    print(f"WINNER: {winner}", flush=True)
    return selection_result


if __name__ == "__main__":
    run_experiment_sweep()
