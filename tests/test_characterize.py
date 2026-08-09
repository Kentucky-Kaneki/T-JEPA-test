"""
Unit and integration tests for Dataset Characterization Pipeline.

Verifies report generation, JSON/MD export, diagnostic plot creation,
episode-bounded metrics, and acceptance gate enforcement.
"""

from pathlib import Path
import pytest
import CybORG as cyborg_pkg

from cyber_jepa.data.collector import collect_shard, get_scenario1b_path
from cyber_jepa.data.characterize import analyze_dataset_shards


def test_characterization_pipeline(tmp_path: Path):
    """Verify dataset characterization pipeline over synthetic shards."""
    scen_path = get_scenario1b_path()
    shards_dir = tmp_path / "shards"
    reports_dir = tmp_path / "reports"

    # Collect 2 shards totaling 100 transitions so quality gates pass
    collect_shard(
        scenario_path=scen_path,
        red_policy="bline",
        blue_policy="random",
        episodes=5,
        max_steps=10,
        seed=1001,
        dataset_id="test_char",
        output_dir=shards_dir / "shard1",
    )

    collect_shard(
        scenario_path=scen_path,
        red_policy="meander",
        blue_policy="coverage",
        episodes=5,
        max_steps=10,
        seed=2003,
        dataset_id="test_char",
        output_dir=shards_dir / "shard2",
    )

    report, md = analyze_dataset_shards(shards_dir, reports_dir)

    assert report["total_transitions"] == 100
    assert report["total_episodes"] == 10
    assert report["num_shards"] == 2

    assert (reports_dir / "dataset_report.json").exists()
    assert (reports_dir / "dataset_report.md").exists()
    assert (reports_dir / "horizon_change_rate.png").exists()

    assert "Dataset Characterization & Diagnostic Report" in md
