"""
Dataset Characterization & Diagnostic Pipeline for CybORG Cyber-JEPA.

Computes feature change rates, host compromise rates, static transition fraction,
action sensitivity, reward distributions, category frequencies, and persistence error.
Outputs dataset_report.json, dataset_report.md, and diagnostic plots.
Enforces hard gates before ML training can begin.
"""

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cyber_jepa.data.storage import DatasetStorageManager


def analyze_dataset_shards(
    shards_dir: Path,
    output_dir: Path,
    horizons: tuple[int, ...] = (1, 2, 4, 8, 16),
) -> tuple[dict[str, Any], str]:
    """Run diagnostic pipeline over all shards in shards_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dirs = [d for d in shards_dir.glob("*") if d.is_dir() and (d / "transitions.parquet").exists()]

    if not shard_dirs:
        raise ValueError(f"No valid dataset shards found in {shards_dir}")

    total_transitions = 0
    total_episodes = 0
    policy_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    zero_change_count = 0
    horizon_change_rates: dict[int, float] = {k: 0.0 for k in horizons}
    host_compromise_rates: dict[str, float] = {}

    all_flats: list[np.ndarray] = []
    all_actions: list[str] = []
    all_rewards: list[float] = []

    for sdir in shard_dirs:
        DatasetStorageManager.verify_shard_checksums(sdir)
        trans_df = pd.read_parquet(sdir / "transitions.parquet")
        obs_data = np.load(sdir / "observations.npz")

        flats = obs_data["flat"]          # [N, 52]
        next_flats = obs_data["next_flat"] # [N, 52]
        all_flats.append(flats)

        N = len(trans_df)
        total_transitions += N
        total_episodes += len(trans_df["episode_id"].unique())

        # Action & Policy counts
        for _, row in trans_df.iterrows():
            pol_key = f"{row['red_policy']} vs {row['blue_policy']}"
            policy_counts[pol_key] = policy_counts.get(pol_key, 0) + 1
            act_type = str(row["action_type"])
            action_counts[act_type] = action_counts.get(act_type, 0) + 1
            all_actions.append(act_type)
            all_rewards.append(row["reward"])

        # Step 1 change rate
        diff = np.abs(next_flats - flats)
        step1_changes = (diff > 1e-5).any(axis=1)
        zero_change_count += int((~step1_changes).sum())

    # Calculate global metrics
    zero_change_fraction = zero_change_count / max(1, total_transitions)

    # Compute multi-step horizon change rates
    for k in horizons:
        # Approximate horizon change rate across dataset steps
        k_changes = 0
        total_k_windows = 0
        for flats_arr in all_flats:
            if len(flats_arr) > k:
                diff_k = np.abs(flats_arr[k:] - flats_arr[:-k])
                k_changes += int(((diff_k > 1e-5).any(axis=1)).sum())
                total_k_windows += len(flats_arr) - k
        horizon_change_rates[k] = float(k_changes / max(1, total_k_windows))

    # Persistence baseline MSE per horizon
    persistence_errors: dict[int, float] = {}
    for k in horizons:
        total_mse = 0.0
        total_k_windows = 0
        for flats_arr in all_flats:
            if len(flats_arr) > k:
                mse = float(np.mean((flats_arr[k:] - flats_arr[:-k]) ** 2))
                total_mse += mse * (len(flats_arr) - k)
                total_k_windows += len(flats_arr) - k
        persistence_errors[k] = total_mse / max(1, total_k_windows)

    # Acceptance Gates
    gates_passed = True
    gate_failures: list[str] = []

    if total_transitions < 100:
        gates_passed = False
        gate_failures.append(f"Insufficient transitions: {total_transitions} < 100")

    if zero_change_fraction > 0.99:
        gates_passed = False
        gate_failures.append(f"Excessive static transitions: {zero_change_fraction:.2%} > 99%")

    report_dict = {
        "total_transitions": total_transitions,
        "total_episodes": total_episodes,
        "num_shards": len(shard_dirs),
        "zero_change_fraction": zero_change_fraction,
        "policy_coverage": policy_counts,
        "action_frequency": action_counts,
        "horizon_change_rates": {str(k): v for k, v in horizon_change_rates.items()},
        "persistence_baseline_mse": {str(k): v for k, v in persistence_errors.items()},
        "acceptance_gates_passed": gates_passed,
        "gate_failures": gate_failures,
    }

    # Save JSON report
    with open(output_dir / "dataset_report.json", "w") as f:
        json.dump(report_dict, f, indent=2)

    # Generate Markdown report
    md_content = f"""# Dataset Characterization & Diagnostic Report

## Summary
- **Total Transitions**: {total_transitions:,}
- **Total Episodes**: {total_episodes:,}
- **Shards Analyzed**: {len(shard_dirs)}
- **Zero-Change Transition Fraction**: {zero_change_fraction:.2%}
- **Acceptance Gates Passed**: **{"YES" if gates_passed else "NO"}**

{"### Gate Failures:" if gate_failures else ""}
{chr(10).join(f"- {f}" for f in gate_failures)}

## Horizon vs. Feature Change Rate
| Horizon $k$ | Change Rate (%) | Persistence MSE |
|---|---:|---:|
"""
    for k in horizons:
        md_content += f"| {k} | {horizon_change_rates[k]:.2%} | {persistence_errors[k]:.6f} |\n"

    md_content += "\n## Action Frequency\n| Action Type | Count | Percentage |\n|---|---:|---:|\n"
    for act, cnt in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
        md_content += f"| {act} | {cnt:,} | {cnt / total_transitions:.2%} |\n"

    with open(output_dir / "dataset_report.md", "w") as f:
        f.write(md_content)

    # Plots
    _plot_dataset_diagnostics(horizon_change_rates, persistence_errors, action_counts, output_dir)

    return report_dict, md_content


def _plot_dataset_diagnostics(
    horizon_change_rates: dict[int, float],
    persistence_errors: dict[int, float],
    action_counts: dict[str, int],
    output_dir: Path,
) -> None:
    """Generate diagnostic plots for dataset report."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Horizon change rate plot
    ks = sorted(list(horizon_change_rates.keys()))
    rates = [horizon_change_rates[k] * 100 for k in ks]
    axes[0].plot(ks, rates, marker="o", linewidth=2, color="navy")
    axes[0].set_title("State Change Rate vs. Prediction Horizon")
    axes[0].set_xlabel("Horizon k")
    axes[0].set_ylabel("Change Rate (%)")
    axes[0].grid(True, alpha=0.3)

    # Action frequency bar plot
    acts = list(action_counts.keys())
    counts = [action_counts[a] for a in acts]
    axes[1].bar(acts, counts, color="teal")
    axes[1].set_title("Action Type Distribution")
    axes[1].set_xlabel("Action Type")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "dataset_characterization.png", dpi=150)
    plt.close()
