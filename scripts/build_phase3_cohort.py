"""Build the immutable Phase 3 k=8, h=4 holdout cohort from collected shards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cyber_jepa.data.dataset import generate_group_splits, verify_dataset_integrity


def build_phase3_cohort(
    shards_dir: Path = Path("data/shards"),
    output_dir: Path = Path("experiments/phase3"),
    horizon: int = 8,
    history_len: int = 4,
    cohort_seed: int = 4201,
) -> Path:
    shard_dirs = sorted(p for p in shards_dir.iterdir() if (p / "transitions.parquet").exists())
    if not shard_dirs:
        raise FileNotFoundError(f"No collected shards under {shards_dir}")
    verify_dataset_integrity(shard_dirs)

    transitions = pd.concat([pd.read_parquet(p / "transitions.parquet") for p in shard_dirs], ignore_index=True)
    labels = pd.concat([pd.read_parquet(p / "oracle_labels.parquet") for p in shard_dirs], ignore_index=True)
    merged = transitions.merge(
        labels[["transition_id", "critical_server_compromised"]], on="transition_id", how="inner"
    )
    splits = generate_group_splits(sorted(merged["split_group_id"].unique().tolist()))
    holdout = merged[merged["split_group_id"].isin(splits["test"])]

    rows: list[dict[str, object]] = []
    for trajectory_id, group in holdout.groupby("trajectory_id", sort=True):
        group = group.sort_values("step_index").reset_index(drop=True)
        for idx in range(history_len - 1, len(group) - horizon):
            context, target = group.iloc[idx], group.iloc[idx + horizon]
            rows.append({
                "transition_id": str(target.transition_id),
                "trajectory_id": str(trajectory_id),
                "split_group_id": str(target.split_group_id),
                "t_context": int(context.step_index),
                "t_target": int(target.step_index),
                "red_policy": str(target.red_policy),
                "blue_policy": str(target.blue_policy),
                "future_critical_server_label": int(target.critical_server_compromised),
            })
    cohort = pd.DataFrame(rows).sort_values("transition_id").reset_index(drop=True)
    if len(cohort) > 2000:
        cohort = cohort.iloc[np.random.RandomState(cohort_seed).choice(len(cohort), 2000, replace=False)]
        cohort = cohort.sort_values("transition_id").reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = output_dir / "phase3_cohort.parquet"
    cohort.to_parquet(cohort_path, index=False)
    digest = hashlib.sha256(cohort_path.read_bytes()).hexdigest()
    (output_dir / "phase3_cohort.sha256").write_text(digest + "\n", encoding="utf-8")
    (output_dir / "phase3_cohort.json").write_text(json.dumps({
        "horizon": horizon, "history_len": history_len, "cohort_seed": cohort_seed,
        "samples": len(cohort), "cohort_sha256": digest,
    }, indent=2), encoding="utf-8")
    print(f"Created {cohort_path} ({len(cohort)} samples; sha256 {digest[:16]}...)")
    return cohort_path


if __name__ == "__main__":
    build_phase3_cohort()
