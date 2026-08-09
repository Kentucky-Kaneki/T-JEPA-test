"""
Unit and audit tests for deterministic episode splitting, windowing, and zero oracle leakage.

Verifies:
- Train, Val, Test episode splits are strictly disjoint
- Sliding windows drop boundary-crossing samples
- No model batch or dataset item contains oracle ground-truth fields or file references
"""

from pathlib import Path
from typing import Any
import numpy as np
import pytest
from torch.utils.data import DataLoader

from cyber_jepa.data.collector import collect_shard, get_scenario1b_path
from cyber_jepa.data.dataset import generate_episode_splits, CyberJEPADataset


ORACLE_LEAKAGE_KEYS = {
    "host_compromise_status",
    "attacker_present",
    "red_stage",
    "critical_server_compromised",
    "oracle_labels.parquet",
    "oracle_labels",
}


def test_episode_splits_disjointness():
    """Verify train, val, and test splits are strictly disjoint."""
    episode_ids = [f"ep_{i:03d}" for i in range(100)]
    splits = generate_episode_splits(episode_ids, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    assert train_set.isdisjoint(val_set), "Train and Val splits overlap!"
    assert train_set.isdisjoint(test_set), "Train and Test splits overlap!"
    assert val_set.isdisjoint(test_set), "Val and Test splits overlap!"
    assert len(train_set) + len(val_set) + len(test_set) == 100


def test_windowing_boundary_rules(tmp_path: Path):
    """Verify windowing drops boundary-crossing samples and enforces 4 history timesteps."""
    scen_path = get_scenario1b_path()
    shard_dir = tmp_path / "shard_win"

    transitions, _, _ = collect_shard(
        scenario_path=scen_path,
        red_policy="bline",
        blue_policy="random",
        episodes=2,
        max_steps=10,
        seed=1001,
        dataset_id="win_test",
        output_dir=shard_dir,
    )

    ep_ids = sorted(list(set(t.episode_id for t in transitions)))

    # Dataset at horizon k=4
    ds = CyberJEPADataset(
        shard_dirs=[shard_dir],
        episode_split=ep_ids,
        horizon=4,
        history_len=4,
        fit_normalizers=True,
    )

    for item in ds:
        # History shape must be [4, 52]
        assert item["history_flat"].shape == (4, 52)
        # Action sequence length must equal horizon (4)
        assert item["action_seq"].shape == (4,)
        # Target flat shape must be [52]
        assert item["target_flat"].shape == (52,)
        assert item["horizon"] == 4

        # Context timestep vs Target timestep must equal horizon (4)
        assert item["t_target"] - item["t_context"] == 4


def test_strict_oracle_leakage_audit(tmp_path: Path):
    """Recursively scan model batches to assert ZERO oracle ground-truth leakage."""
    scen_path = get_scenario1b_path()
    shard_dir = tmp_path / "shard_leak"

    transitions, _, _ = collect_shard(
        scenario_path=scen_path,
        red_policy="meander",
        blue_policy="coverage",
        episodes=2,
        max_steps=10,
        seed=2003,
        dataset_id="leak_test",
        output_dir=shard_dir,
    )

    ep_ids = sorted(list(set(t.episode_id for t in transitions)))
    ds = CyberJEPADataset(
        shard_dirs=[shard_dir],
        episode_split=ep_ids,
        horizon=2,
        history_len=4,
        fit_normalizers=True,
    )

    loader = DataLoader(ds, batch_size=4, shuffle=False)

    for batch in loader:
        _assert_no_oracle_leakage_recursive(batch)


def _assert_no_oracle_leakage_recursive(obj: Any) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert str(k) not in ORACLE_LEAKAGE_KEYS, f"Oracle key '{k}' detected in batch dict!"
            _assert_no_oracle_leakage_recursive(v)
    elif isinstance(obj, (list, tuple)):
        for elem in obj:
            _assert_no_oracle_leakage_recursive(elem)
    elif isinstance(obj, str):
        for leak_key in ORACLE_LEAKAGE_KEYS:
            assert leak_key not in obj, f"Oracle key string '{leak_key}' detected in batch value!"
