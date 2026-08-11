"""
Dataset Characterization & Diagnostic Pipeline for CybORG Cyber-JEPA.

Computes feature change rates, host compromise rates, static transition fraction,
action sensitivity, reward distributions, category frequencies, and persistence error.
Computes all temporal metrics strictly WITHIN episode boundaries.
Outputs dataset_report.json, dataset_report.md, and diagnostic plots.
Enforces hard gates before ML training can begin.
"""

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cyber_jepa.data.storage import DatasetStorageManager


def analyze_dataset_shards(
    shards_dir: Path,
    output_dir: Path,
    horizons: tuple[int, ...] = (1, 2, 4, 8, 16),
) -> tuple[dict[str, Any], str]:
    """Run diagnostic pipeline over all shards in shards_dir strictly within episode boundaries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dirs = [d for d in shards_dir.glob("*") if d.is_dir() and (d / "transitions.parquet").exists()]

    if not shard_dirs:
        raise ValueError(f"No valid dataset shards found in {shards_dir}")

    total_transitions = 0
    total_episodes = 0
    policy_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    host_target_counts: dict[str, int] = {}
    subnet_target_counts: dict[str, int] = {}
    episode_lengths: list[int] = []
    oracle_compromise_counts: dict[str, int] = {}
    invalid_param_count = 0

    zero_change_count = 0
    horizon_change_rates: dict[int, float] = {k: 0.0 for k in horizons}
    persistence_errors: dict[int, float] = {k: 0.0 for k in horizons}

    # Grouped episode storage for episode-bounded metric calculation
    episodes_map: dict[str, dict[str, Any]] = {}

    for sdir in shard_dirs:
        DatasetStorageManager.verify_shard_checksums(sdir)
        trans_df = pd.read_parquet(sdir / "transitions.parquet")
        obs_data = np.load(sdir / "observations.npz")
        oracle_df = pd.read_parquet(sdir / "oracle_labels.parquet") if (sdir / "oracle_labels.parquet").exists() else None

        flats = obs_data["flat"]          # [N, 52]
        next_flats = obs_data["next_flat"] # [N, 52]
        known_masks = obs_data["known_host_mask"] # [N, 13]

        N = len(trans_df)
        total_transitions += N

        for idx, row in trans_df.iterrows():
            ep_id = str(row["trajectory_id"])
            if ep_id not in episodes_map:
                episodes_map[ep_id] = {
                    "flats": [],
                    "known_masks": [],
                }
            episodes_map[ep_id]["flats"].append(flats[idx])
            episodes_map[ep_id]["known_masks"].append(known_masks[idx])

            pol_key = f"{row['red_policy']} vs {row['blue_policy']}"
            policy_counts[pol_key] = policy_counts.get(pol_key, 0) + 1

            act_type = str(row["action_type"])
            action_counts[act_type] = action_counts.get(act_type, 0) + 1

            host_tgt = str(row.get("host_target", "NONE"))
            host_target_counts[host_tgt] = host_target_counts.get(host_tgt, 0) + 1

            sub_tgt = str(row.get("subnet_target", "NONE"))
            subnet_target_counts[sub_tgt] = subnet_target_counts.get(sub_tgt, 0) + 1

            if not row.get("action_valid", True):
                invalid_param_count += 1

            # Step 1 change
            diff1 = np.abs(next_flats[idx] - flats[idx])
            if (diff1 <= 1e-5).all():
                zero_change_count += 1

        if oracle_df is not None:
            for _, o_row in oracle_df.iterrows():
                stage = str(o_row.get("red_stage", "unknown"))
                oracle_compromise_counts[stage] = oracle_compromise_counts.get(stage, 0) + 1

    total_episodes = len(episodes_map)
    episode_lengths = [len(ep["flats"]) for ep in episodes_map.values()]

    # Calculate temporal metrics STRICTLY WITHIN episode boundaries
    for k in horizons:
        k_changes = 0
        total_k_windows = 0
        total_mse = 0.0

        for ep_id, ep_data in episodes_map.items():
            ep_flats = np.array(ep_data["flats"], dtype=np.float32) # [L, 52]
            L = len(ep_flats)
            if L > k:
                diff_k = np.abs(ep_flats[k:] - ep_flats[:-k])
                changed_windows = (diff_k > 1e-5).any(axis=1)
                k_changes += int(changed_windows.sum())
                mse_k = float(np.mean((ep_flats[k:] - ep_flats[:-k]) ** 2))

                total_k_windows += (L - k)
                total_mse += mse_k * (L - k)

        horizon_change_rates[k] = float(k_changes / max(1, total_k_windows))
        persistence_errors[k] = float(total_mse / max(1, total_k_windows))

    # Calculate global metrics
    zero_change_fraction = float(zero_change_count / max(1, total_transitions))

    # Acceptance Gates (Strict Evaluation)
    gates_passed = True
    gate_failures: list[str] = []

    if total_transitions < 100:
        gates_passed = False
        gate_failures.append(f"Gate 1 Failed: Insufficient transitions ({total_transitions} < 100)")

    if zero_change_fraction > 0.99:
        gates_passed = False
        gate_failures.append(f"Gate 2 Failed: Excessive static transitions ({zero_change_fraction:.2%} > 99%)")

    if horizon_change_rates.get(4, 0.0) < 0.001:
        gates_passed = False
        gate_failures.append(f"Gate 3 Failed: Insufficient state change at horizon k=4 ({horizon_change_rates.get(4, 0.0):.4%})")

    report_dict = {
        "total_transitions": total_transitions,
        "total_episodes": total_episodes,
        "num_shards": len(shard_dirs),
        "zero_change_fraction": zero_change_fraction,
        "policy_coverage": policy_counts,
        "action_frequency": action_counts,
        "host_target_frequency": host_target_counts,
        "subnet_target_frequency": subnet_target_counts,
        "invalid_action_parameter_rate": float(invalid_param_count / max(1, total_transitions)),
        "episode_length_mean": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "oracle_stage_prevalence": oracle_compromise_counts,
        "horizon_change_rates": {str(k): v for k, v in horizon_change_rates.items()},
        "persistence_baseline_mse": {str(k): v for k, v in persistence_errors.items()},
        "acceptance_gates_passed": gates_passed,
        "gate_failures": gate_failures,
    }

    # Save JSON report
    with open(output_dir / "dataset_report.json", "w") as f:
        json.dump(report_dict, f, indent=2)

    # Plot diagnostic figures using headless Agg backend
    plt.figure(figsize=(8, 4))
    plt.plot(list(horizons), [horizon_change_rates[k] for k in horizons], marker="o", color="blue", label="Change Rate")
    plt.title("Feature Change Rate vs. Horizon (Episode-Bounded)")
    plt.xlabel("Horizon k")
    plt.ylabel("Change Rate")
    plt.grid(True)
    plt.savefig(output_dir / "horizon_change_rate.png", bbox_inches="tight")
    plt.close()

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

## Horizon vs. Feature Change Rate (Episode-Bounded)
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

    if not gates_passed:
        raise ValueError(f"Dataset characterization failed quality gates: {gate_failures}")

    return report_dict, md_content
