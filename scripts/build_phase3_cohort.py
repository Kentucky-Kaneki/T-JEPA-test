"""
Phase 3 Evaluation Cohort Generator.

Constructs, validates, and persists a single immutable Phase 3 evaluation cohort
(k=8, h=4) across unseen split groups to guarantee identical evaluation targets
across all models, seeds, and representations.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cyber_jepa.data.dataset import generate_group_splits, verify_dataset_integrity


def build_phase3_cohort(
    shards_dir: Path,
    output_dir: Path,
    horizon: int = 8,
    history_len: int = 4,
    cohort_seed: int = 4201,
    split_seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    """Generate and save persistent Phase 3 evaluation cohort and sidecars."""
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dirs = sorted([
        d for d in shards_dir.glob("*")
        if d.is_dir() and (d / "transitions.parquet").exists()
    ])

    if not shard_dirs:
        raise ValueError(f"No dataset shards found in {shards_dir}")

    # Verify integrity of all shards
    verify_dataset_integrity(shard_dirs)

    # 1. Collect all transitions and oracle labels
    trans_list: list[pd.DataFrame] = []
    oracle_list: list[pd.DataFrame] = []
    manifest_hashes: dict[str, str] = {}

    for sdir in shard_dirs:
        t_df = pd.read_parquet(sdir / "transitions.parquet")
        o_df = pd.read_parquet(sdir / "oracle_labels.parquet")
        trans_list.append(t_df)
        oracle_list.append(o_df)

        m_path = sdir / "manifest.json"
        if m_path.exists():
            manifest_hashes[sdir.name] = hashlib.sha256(m_path.read_bytes()).hexdigest()

    all_trans = pd.concat(trans_list, ignore_index=True)
    all_oracle = pd.concat(oracle_list, ignore_index=True)

    # Join transitions with oracle labels strictly on transition_id
    merged_df = pd.merge(
        all_trans,
        all_oracle[["transition_id", "critical_server_compromised"]],
        on="transition_id",
        how="inner",
    )

    # 2. Partition split_group_ids
    all_group_ids = sorted(merged_df["split_group_id"].unique().tolist())
    splits = generate_group_splits(all_group_ids, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    holdout_group_ids = set(splits["test"])

    # Filter for holdout group transitions
    holdout_df = merged_df[merged_df["split_group_id"].isin(holdout_group_ids)].copy()

    # 3. Build sliding windows (t_context, t_target = t_context + k) per trajectory
    cohort_rows: list[dict[str, Any]] = []
    grouped = holdout_df.groupby("trajectory_id", sort=True)

    for traj_id, group in grouped:
        group_sorted = group.sort_values(by="step_index").reset_index(drop=True)
        seq_len = len(group_sorted)

        # Sliding window requirement: history_len history steps + horizon target step
        for i in range(history_len - 1, seq_len - horizon):
            ctx_row = group_sorted.iloc[i]
            tgt_row = group_sorted.iloc[i + horizon]

            cohort_rows.append({
                "transition_id": str(tgt_row["transition_id"]),
                "trajectory_id": str(traj_id),
                "split_group_id": str(tgt_row["split_group_id"]),
                "t_context": int(ctx_row["step_index"]),
                "t_target": int(tgt_row["step_index"]),
                "red_policy": str(tgt_row["red_policy"]),
                "blue_policy": str(tgt_row["blue_policy"]),
                "collection_seed": int(tgt_row["seed"]),
                "future_critical_server_label": int(tgt_row["critical_server_compromised"]),
                "current_critical_server_persistence_label": int(
                    ctx_row["critical_server_compromised"]
                ),
            })

    cohort_df = pd.DataFrame(cohort_rows).sort_values(by="transition_id").reset_index(drop=True)

    # Subsample if necessary or keep full holdout evaluation cohort
    rng = np.random.RandomState(cohort_seed)
    if len(cohort_df) > 2000:
        sub_indices = rng.choice(len(cohort_df), size=2000, replace=False).tolist()
        cohort_df = cohort_df.iloc[sub_indices].sort_values(
            by="transition_id"
        ).reset_index(drop=True)

    # 4. Save cohort Parquet
    parquet_path = output_dir / "phase3_cohort.parquet"
    cohort_df.to_parquet(parquet_path, index=False)
    cohort_sha256 = hashlib.sha256(parquet_path.read_bytes()).hexdigest()

    # Write cohort_hash.txt
    with open(output_dir / "phase3_cohort.sha256", "w") as f:
        f.write(cohort_sha256 + "\n")

    # 5. Class prevalence & policy breakdowns
    total_samples = len(cohort_df)
    pos_samples = int(cohort_df["future_critical_server_label"].sum())
    prevalence = pos_samples / max(1, total_samples)

    red_prevalence = {}
    for r_pol, r_group in cohort_df.groupby("red_policy"):
        red_prevalence[r_pol] = {
            "total": len(r_group),
            "positive": int(r_group["future_critical_server_label"].sum()),
            "prevalence": float(r_group["future_critical_server_label"].mean()),
        }

    blue_prevalence = {}
    for b_pol, b_group in cohort_df.groupby("blue_policy"):
        blue_prevalence[b_pol] = {
            "total": len(b_group),
            "positive": int(b_group["future_critical_server_label"].sum()),
            "prevalence": float(b_group["future_critical_server_label"].mean()),
        }

    manifest_json: dict[str, Any] = {
        "cohort_generation_config": {
            "scenario": "Scenario1b",
            "horizon": horizon,
            "history_len": history_len,
            "cohort_seed": cohort_seed,
            "split_seed": split_seed,
            "holdout_type": "unseen_split_group_holdout",
        },
        "num_split_groups": len(cohort_df["split_group_id"].unique()),
        "num_trajectories": len(cohort_df["trajectory_id"].unique()),
        "num_transitions": total_samples,
        "class_prevalence": prevalence,
        "prevalence_by_red_policy": red_prevalence,
        "prevalence_by_blue_policy": blue_prevalence,
        "dataset_manifest_hashes": manifest_hashes,
        "cohort_sha256": cohort_sha256,
    }

    json_path = output_dir / "phase3_cohort.json"
    with open(json_path, "w") as f:
        json.dump(manifest_json, f, indent=2)

    print(
        f"[+] Phase 3 Cohort generated successfully! {total_samples} samples across "
        f"{manifest_json['num_trajectories']} trajectories. SHA256: {cohort_sha256[:16]}..."
    )

    return cohort_df, manifest_json, cohort_sha256


if __name__ == "__main__":
    shards_p = Path("data/shards")
    out_p = Path("experiments/phase3")
    build_phase3_cohort(shards_p, out_p)
