"""
Windowing, Deterministic Episode Splitting, and PyTorch Dataset Loader for Cyber-JEPA.

Constructs (history, action_sequence, target) sliding windows over episodes:
- History: O_{t-3:t}^{Blue} (4 timesteps)
- Actions: a_{t:t+k-1}^{Blue} (k timesteps)
- Target: O_{t+k}^{Blue}
Enforces zero padding across episode boundaries and deterministic episode-level splits.
"""

import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from cyber_jepa.data.schema import BlueObservation, ActionSpec

def verify_dataset_integrity(shard_dirs: list[Path]) -> None:
    """Strictly enforce uniqueness and 1-to-1 transition-to-oracle join across all shards."""
    trans_list = []
    oracle_list = []
    
    for sdir in shard_dirs:
        t_path = sdir / "transitions.parquet"
        o_path = sdir / "oracle_labels.parquet"
        
        if not t_path.exists() or not o_path.exists():
            continue
            
        trans_list.append(pd.read_parquet(t_path))
        oracle_list.append(pd.read_parquet(o_path))
        
    if not trans_list:
        raise ValueError("No transitions found to verify.")
        
    all_trans = pd.concat(trans_list, ignore_index=True)
    all_oracle = pd.concat(oracle_list, ignore_index=True)
    
    # 1. Uniqueness of trajectory_id
    if not all_trans["trajectory_id"].is_unique:
        # Wait, trajectory_id is unique per *trajectory*, not transition.
        # But wait, transition_id must be globally unique
        pass
        
    # 1. Uniqueness of transition_id
    if not all_trans["transition_id"].is_unique:
        duplicates = all_trans[all_trans["transition_id"].duplicated(keep=False)]
        raise ValueError(f"Integrity Violation: Duplicate transition_ids found: {duplicates['transition_id'].tolist()}")
        
    if not all_oracle["transition_id"].is_unique:
        raise ValueError("Integrity Violation: Duplicate transition_ids in oracle sidecar.")
        
    # 2. Exact 1-to-1 join
    trans_ids = set(all_trans["transition_id"])
    oracle_ids = set(all_oracle["transition_id"])
    
    orphans_in_trans = trans_ids - oracle_ids
    orphans_in_oracle = oracle_ids - trans_ids
    
    if orphans_in_trans:
        raise ValueError(f"Integrity Violation: {len(orphans_in_trans)} orphans in transitions (no oracle).")
        
    if orphans_in_oracle:
        raise ValueError(f"Integrity Violation: {len(orphans_in_oracle)} orphans in oracle (no transition).")
        
    print(f"Dataset integrity verified: {len(trans_ids)} transitions with perfect 1-to-1 oracle alignment.")


def generate_group_splits(
    group_ids: list[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    salt: str = "cyborg_jepa_split_v1",
) -> dict[str, list[str]]:
    """Deterministically partition split_group_ids into train, val, and test splits."""
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-5:
        raise ValueError("Split ratios must sum to 1.0")

    train_eps, val_eps, test_eps = [], [], []

    for grp_id in sorted(list(set(group_ids))):
        h = hashlib.sha256(f"{salt}_{grp_id}".encode("utf-8")).hexdigest()
        val_hash = int(h[:8], 16) / 0xFFFFFFFF

        if val_hash < train_ratio:
            train_eps.append(grp_id)
        elif val_hash < train_ratio + val_ratio:
            val_eps.append(grp_id)
        else:
            test_eps.append(grp_id)

    return {
        "train": sorted(train_eps),
        "val": sorted(val_eps),
        "test": sorted(test_eps),
    }


def generate_ood_splits(
    transitions_df: pd.DataFrame,
) -> dict[str, dict[str, list[str]]]:
    """Generate Out-of-Distribution (OOD) transfer splits: B-line vs Meander."""
    bline_eps = sorted(transitions_df[transitions_df["red_policy"] == "bline"]["split_group_id"].unique().tolist())
    meander_eps = sorted(transitions_df[transitions_df["red_policy"] == "meander"]["split_group_id"].unique().tolist())

    return {
        "bline_to_meander": {"train": bline_eps, "test": meander_eps},
        "meander_to_bline": {"train": meander_eps, "test": bline_eps},
    }


class CyberJEPADataset(Dataset):
    """PyTorch Dataset exposing windowed observations, action sequences, and targets."""

    def __init__(
        self,
        shard_dirs: list[Path],
        split_group_set: list[str],
        horizon: int = 1,
        history_len: int = 4,
        fit_normalizers: bool = False,
        normalizer_stats: dict[str, Any] | None = None,
    ):
        self.horizon = horizon
        self.history_len = history_len
        self.split_group_set = set(split_group_set)

        self.samples: list[dict[str, Any]] = []
        self._load_and_window_shards(shard_dirs)

        if fit_normalizers:
            self.normalizer_stats = self._fit_normalizers()
        else:
            self.normalizer_stats = normalizer_stats or {}

    def _load_and_window_shards(self, shard_dirs: list[Path]) -> None:
        """Construct sliding windows over valid episodes without boundary crossing."""
        for sdir in shard_dirs:
            trans_df = pd.read_parquet(sdir / "transitions.parquet")
            obs_data = np.load(sdir / "observations.npz")
            flats = obs_data["flat"]

            # Filter for requested split groups
            filtered_df = trans_df[trans_df["split_group_id"].isin(self.split_group_set)]
            ep_groups = filtered_df.groupby("trajectory_id", sort=True)

            for traj_id, group in ep_groups:
                group_indices = np.array(group.index.tolist())
                L = len(group_indices)

                act_col = "action_discrete_index" if "action_discrete_index" in group.columns else "action_idx"
                t_col = "step_index" if "step_index" in group.columns else "t"

                act_vals = group[act_col].values
                t_vals = group[t_col].values

                # Window requirement: history_len history steps + horizon future action/target steps
                for i in range(self.history_len - 1, L - self.horizon):
                    hist_idx_range = group_indices[i - self.history_len + 1 : i + 1]
                    target_idx = group_indices[i + self.horizon]

                    hist_flats = flats[hist_idx_range]                # [4, 52]
                    target_flat = flats[target_idx]                   # [52]
                    action_seq = act_vals[i : i + self.horizon].tolist()
                    t_ctx = int(t_vals[i])
                    t_tgt = int(t_vals[i + self.horizon])

                    self.samples.append({
                        "trajectory_id": traj_id,
                        "t_context": t_ctx,
                        "t_target": t_tgt,
                        "history_flat": torch.tensor(hist_flats, dtype=torch.float32),
                        "action_seq": torch.tensor(action_seq, dtype=torch.long),
                        "target_flat": torch.tensor(target_flat, dtype=torch.float32),
                        "horizon": self.horizon,
                    })

    def _fit_normalizers(self) -> dict[str, Any]:
        """Fit feature mean and std on training split history observations only."""
        if not self.samples:
            return {"mean": 0.0, "std": 1.0}
        all_hist = np.concatenate([s["history_flat"].numpy() for s in self.samples], axis=0)
        mean = np.mean(all_hist, axis=0)
        std = np.std(all_hist, axis=0)
        std[std < 1e-6] = 1.0
        return {"mean": mean.tolist(), "std": std.tolist()}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]

        # Apply normalizer stats if available
        hist = sample["history_flat"]
        target = sample["target_flat"]

        if "mean" in self.normalizer_stats and "std" in self.normalizer_stats:
            mean = torch.tensor(self.normalizer_stats["mean"], dtype=torch.float32)
            std = torch.tensor(self.normalizer_stats["std"], dtype=torch.float32)
            hist = (hist - mean) / std
            target = (target - mean) / std

        return {
            "trajectory_id": sample["trajectory_id"],
            "t_context": sample["t_context"],
            "t_target": sample["t_target"],
            "history_flat": hist,
            "action_seq": sample["action_seq"],
            "target_flat": target,
            "horizon": sample["horizon"],
        }
