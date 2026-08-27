"""Collect and validate deterministic CybORG Scenario1b dataset shards.

The collection matrix is read from ``configs/data/collection_full.yaml``.  A fresh
default run creates 2 red policies x 3 blue policies x 3 seeds = 18 immutable
shards in ``data/shards``. Existing shards are never overwritten: use ``--resume``
to verify and skip them, or choose a new output directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from cyber_jepa.data.characterize import analyze_dataset_shards
from cyber_jepa.data.collector import collect_shard, get_scenario1b_path
from cyber_jepa.data.storage import DatasetStorageManager


DEFAULT_CONFIG = Path("configs/data/collection_full.yaml")


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate the declared collection matrix."""
    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    required = {
        "episodes_per_shard",
        "max_steps_per_episode",
        "red_policies",
        "blue_policies",
        "seeds",
        "shards_dir",
    }
    missing = required - set(config or {})
    if missing:
        raise ValueError(f"Collection config missing required keys: {sorted(missing)}")
    return config


def run_collection(
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path | None = None,
    reports_dir: Path = Path("reports"),
    resume: bool = False,
    dry_run: bool = False,
) -> None:
    config = load_config(config_path)
    shard_root = output_dir or Path(config["shards_dir"])
    scenario_path = get_scenario1b_path()

    red_policies: list[str] = config["red_policies"]
    blue_policies: list[str] = config["blue_policies"]
    seeds: list[int] = config["seeds"]
    episodes = int(config["episodes_per_shard"])
    max_steps = int(config["max_steps_per_episode"])
    total = len(red_policies) * len(blue_policies) * len(seeds)

    print(f"Scenario: {scenario_path}")
    print(f"Collection matrix: {len(red_policies)} red x {len(blue_policies)} blue x {len(seeds)} seeds = {total} shards")
    print(f"Per shard: {episodes} episodes x at most {max_steps} steps")
    print(f"Output: {shard_root}")

    if dry_run:
        return

    shard_root.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    completed = 0

    for red_policy in red_policies:
        for blue_policy in blue_policies:
            for seed in seeds:
                completed += 1
                shard_dir = shard_root / f"shard_{red_policy}_{blue_policy}_{seed}"
                if shard_dir.exists():
                    if not resume:
                        raise FileExistsError(
                            f"Refusing to overwrite existing shard: {shard_dir}. "
                            "Use --resume to checksum and skip it, or select a new --output-dir."
                        )
                    DatasetStorageManager.verify_shard_checksums(shard_dir)
                    print(f"[{completed:02d}/{total}] verified existing {shard_dir.name}; skipping")
                    continue

                print(f"[{completed:02d}/{total}] collecting {shard_dir.name}")
                collect_shard(
                    scenario_path=scenario_path,
                    red_policy=red_policy,
                    blue_policy=blue_policy,
                    episodes=episodes,
                    max_steps=max_steps,
                    seed=seed,
                    dataset_id="cyborg_scenario1b_full",
                    output_dir=shard_dir,
                )

    report, _ = analyze_dataset_shards(shard_root, reports_dir)
    if not report["acceptance_gates_passed"]:
        raise RuntimeError(f"Dataset acceptance gates failed: {report['gate_failures']}")
    print(f"Collection and characterization passed: {report['total_transitions']:,} transitions.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_collection(
        config_path=args.config,
        output_dir=args.output_dir,
        reports_dir=args.reports_dir,
        resume=args.resume,
        dry_run=args.dry_run,
    )
