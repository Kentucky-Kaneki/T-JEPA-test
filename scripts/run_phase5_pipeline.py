"""Collect/resume Scenario1b shards, then execute one Phase 5 stage.

The pinned CybORG source is installed separately through ``requirements/sim.in``.
This wrapper checksum-verifies completed shards, collects only missing ones, and
then starts Phase 5 so the standard flow needs one execution command.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from run_collection import run_collection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["stage1", "stage2", "stage3"], default="stage1")
    parser.add_argument("--selected-ratio")
    parser.add_argument("--shards-dir", type=Path, default=Path("data/shards"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/phase5_runs"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-collection", action="store_true")
    args = parser.parse_args()

    if not args.skip_collection:
        run_collection(output_dir=args.shards_dir, resume=True, dry_run=args.dry_run)

    command = [
        sys.executable, "scripts/run_phase5.py", "--stage", args.stage,
        "--shards-dir", str(args.shards_dir), "--output-dir", str(args.output_dir),
    ]
    if args.selected_ratio:
        command.extend(["--selected-ratio", args.selected_ratio])
    if args.dry_run:
        command.append("--dry-run")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
