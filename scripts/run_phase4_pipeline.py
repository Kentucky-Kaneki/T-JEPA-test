"""One-command, non-overwriting Scenario1b collection and Phase 4 execution.

Existing shards are checksum-verified and skipped.  The immutable Phase 3 cohort
is created only when absent; it is never overwritten by this script.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

# Allow direct execution with ``python scripts/run_phase4_pipeline.py`` while
# keeping imports rooted at the repository.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.run_collection import run_collection
from run_phase4 import main as run_phase4


def main() -> None:
    shards_dir = Path("data/shards")
    phase3_dir = Path("experiments/phase3")
    cohort = phase3_dir / "phase3_cohort.parquet"
    cohort_hash = phase3_dir / "phase3_cohort.sha256"

    # The configured matrix is the Scenario1b source of truth. --resume makes
    # this safe for reruns: validated shards are reused and never overwritten.
    run_collection(
        output_dir=shards_dir,
        reports_dir=Path("experiments/phase4/collection_reports"),
        resume=True,
    )

    if not cohort.exists() or not cohort_hash.exists():
        raise RuntimeError(
            "The immutable Phase 3 cohort and checksum are required for a valid comparison; "
            "do not rebuild them from a new collection."
        )

    # The archived Phase 3 F1 values are valid only for the Phase 3 source
    # matrix. A manifest has a collection timestamp, so compare its stable
    # identity fields rather than its byte hash after a valid regeneration.
    cohort_metadata = json.loads((phase3_dir / "phase3_cohort.json").read_text(encoding="utf-8"))
    expected_manifests = cohort_metadata.get("dataset_manifest_hashes", {})
    if len(expected_manifests) != 18:
        raise RuntimeError("Phase 3 cohort must declare hashes for exactly 18 source shards.")
    actual_names = {path.name for path in shards_dir.iterdir() if (path / "manifest.json").exists()}
    if actual_names != set(expected_manifests):
        raise RuntimeError("Collected shard names do not match the immutable Phase 3 dataset contract.")
    all_transition_ids: set[str] = set()
    for shard_name in expected_manifests:
        manifest = shards_dir / shard_name / "manifest.json"
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        _, red_policy, blue_policy, collection_seed = shard_name.split("_", 3)
        if (
            metadata.get("scenario_name") != "Scenario1b"
            or metadata.get("cyborg_version") != "v3.1 (dd586a3)"
            or metadata.get("red_policy") != red_policy
            or metadata.get("blue_policy") != blue_policy
            or int(metadata.get("collection_seed", -1)) != int(collection_seed)
            or int(metadata.get("requested_episodes", -1)) != 100
            or int(metadata.get("max_steps", -1)) != 50
        ):
            raise RuntimeError(f"Phase 3 collection contract mismatch for {shard_name}.")
        transitions = pd.read_parquet(shards_dir / shard_name / "transitions.parquet")
        all_transition_ids.update(transitions["transition_id"].astype(str))
    cohort_transitions = set(pd.read_parquet(cohort)["transition_id"].astype(str))
    if not cohort_transitions.issubset(all_transition_ids):
        raise RuntimeError("Regenerated shards do not contain every archived Phase 3 cohort transition.")

    run_phase4(["--shards-dir", str(shards_dir)])


if __name__ == "__main__":
    main()
