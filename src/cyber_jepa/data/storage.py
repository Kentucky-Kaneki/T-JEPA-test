"""
Dataset storage manager for CybORG Cyber-JEPA datasets.

Manages saving and loading compressed NumPy arrays (observations), Parquet tables
(transitions and oracle sidecars), zstandard-compressed JSONL (raw blue observations),
JSON manifests, and SHA256 checksums with atomic temporary file renaming.
"""

import datetime
import hashlib
import json
from collections.abc import Hashable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zstandard as zstd

from cyber_jepa.data.schema import (
    SCENARIO1B_ACTION_TYPES,
    SCENARIO1B_HOST_SLOTS,
    SCENARIO1B_SUBNET_SLOTS,
    OracleLabels,
    Transition,
)


class DatasetStorageManager:
    """Manages atomic dataset shard reads/writes and checksum verification."""

    SCHEMA_VERSION = "2.0.0"

    @classmethod
    def save_shard(
        cls,
        shard_dir: Path,
        transitions: list[Transition],
        oracle_labels: list[OracleLabels],
        manifest: dict[str, Any],
    ) -> dict[str, str]:
        """Save a dataset shard atomically using temporary files and atomic rename."""
        shard_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = shard_dir / "_tmp_write"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Observations NPZ (Structured, directly loadable arrays)
            flat_obs = np.array([t.flat_obs for t in transitions], dtype=np.float32)
            flat_next_obs = np.array([t.next_flat_obs for t in transitions], dtype=np.float32)
            known_masks = np.array([t.known_host_mask for t in transitions], dtype=bool)
            next_known_masks = np.array([t.next_known_host_mask for t in transitions], dtype=bool)

            obs_npz_tmp = tmp_dir / "observations_tmp.npz"
            np.savez_compressed(
                obs_npz_tmp,
                flat=flat_obs,
                next_flat=flat_next_obs,
                known_host_mask=known_masks,
                next_known_host_mask=next_known_masks,
            )

            # 2. Transitions Parquet (Canonical transition schema)
            trans_rows = []
            for t in transitions:
                trans_rows.append({
                    "dataset_id": t.dataset_id,
                    "trajectory_id": t.trajectory_id,
                    "split_group_id": t.split_group_id,
                    "transition_id": t.transition_id,
                    "seed": t.seed,
                    "step_index": t.step_index,
                    "t": t.step_index,
                    "terminated": t.terminated,
                    "truncated": t.truncated,
                    "action_discrete_index": t.action_discrete_index,
                    "action_idx": t.action_discrete_index,
                    "action_type": t.action_type,
                    "action_type_id": t.action_type_id,
                    "host_target": t.host_target,
                    "host_target_id": t.host_target_id,
                    "subnet_target": t.subnet_target,
                    "subnet_target_id": t.subnet_target_id,
                    "action_valid": t.action_valid,
                    "reward": t.reward,
                    "oracle_transition_id": t.oracle_transition_id,
                    "scenario_name": t.scenario_name,
                    "scenario_hash": t.scenario_hash,
                    "red_policy": t.red_policy,
                    "blue_policy": t.blue_policy,
                })
            trans_df = pd.DataFrame(trans_rows)
            trans_parquet_tmp = tmp_dir / "transitions_tmp.parquet"
            trans_df.to_parquet(trans_parquet_tmp, index=False)

            # 3. Oracle Labels Parquet (Physically Separate Sidecar)
            oracle_rows = []
            for o in oracle_labels:
                row = {
                    "transition_id": o.transition_id,
                    "trajectory_id": o.trajectory_id,
                    "split_group_id": o.split_group_id,
                    "t": o.t,
                    "red_stage": o.red_stage,
                    "critical_server_compromised": o.critical_server_compromised,
                }
                for host, status in o.host_compromise_status.items():
                    row[f"compromise_{host}"] = status
                for host, present in o.attacker_present.items():
                    row[f"attacker_{host}"] = present
                oracle_rows.append(row)
            oracle_df = pd.DataFrame(oracle_rows)
            oracle_parquet_tmp = tmp_dir / "oracle_labels_tmp.parquet"
            oracle_df.to_parquet(oracle_parquet_tmp, index=False)

            # 4. Raw Blue JSONL.ZST (Audit only)
            raw_jsonl_tmp = tmp_dir / "raw_blue_tmp.jsonl.zst"
            cctx = zstd.ZstdCompressor(level=3)
            with open(raw_jsonl_tmp, "wb") as f_out, cctx.stream_writer(f_out) as writer:
                for t in transitions:
                    line_data = {
                        "transition_id": t.transition_id,
                        "trajectory_id": t.trajectory_id,
                        "split_group_id": t.split_group_id,
                        "step_index": t.step_index,
                        "host_features": t.host_features,
                        "next_host_features": t.next_host_features,
                        "blue_events": t.blue_events,
                    }
                    writer.write((json.dumps(line_data) + "\n").encode("utf-8"))

            # 5. Schema JSON
            schema_data = {
                "schema_version": cls.SCHEMA_VERSION,
                "obs_flat_dim": 52,
                "num_hosts": 13,
                "host_slots": SCENARIO1B_HOST_SLOTS,
                "subnet_slots": SCENARIO1B_SUBNET_SLOTS,
                "action_types": SCENARIO1B_ACTION_TYPES,
                "dtypes": {
                    "flat": "float32",
                    "known_host_mask": "bool",
                    "action_discrete_index": "int64",
                    "action_type_id": "int64",
                },
            }
            schema_tmp = tmp_dir / "schema_tmp.json"
            with open(schema_tmp, "w") as f:
                json.dump(schema_data, f, indent=2)

            # 6. Manifest JSON
            manifest_data = dict(manifest)
            manifest_data["schema_version"] = cls.SCHEMA_VERSION
            manifest_data["cyborg_version"] = "v3.1 (dd586a3)"
            manifest_data["collection_timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            manifest_data["action_vocabulary"] = SCENARIO1B_ACTION_TYPES
            manifest_data["host_vocabulary"] = SCENARIO1B_HOST_SLOTS
            manifest_data["num_transitions"] = len(transitions)
            manifest_data["num_trajectories"] = len(set(t.trajectory_id for t in transitions))

            manifest_tmp = tmp_dir / "manifest_tmp.json"
            with open(manifest_tmp, "w") as f:
                json.dump(manifest_data, f, indent=2)

            # 7. Atomic Publication (Rename .tmp files to final destination)
            target_map = {
                obs_npz_tmp: shard_dir / "observations.npz",
                trans_parquet_tmp: shard_dir / "transitions.parquet",
                oracle_parquet_tmp: shard_dir / "oracle_labels.parquet",
                raw_jsonl_tmp: shard_dir / "raw_blue.jsonl.zst",
                schema_tmp: shard_dir / "schema.json",
                manifest_tmp: shard_dir / "manifest.json",
            }
            for tmp_p, final_p in target_map.items():
                tmp_p.replace(final_p)

            # 8. Checksums SHA256
            checksums = cls._generate_checksums(shard_dir)
            checksum_tmp = tmp_dir / "checksums_tmp.sha256"
            with open(checksum_tmp, "w") as f:
                for fname, sha in sorted(checksums.items()):
                    f.write(f"{sha}  {fname}\n")

            checksum_final = shard_dir / "checksums.sha256"
            checksum_tmp.replace(checksum_final)

            return checksums

        finally:
            if tmp_dir.exists():
                for p in tmp_dir.glob("*"):
                    try:
                        p.unlink()
                    except Exception:
                        pass
                try:
                    tmp_dir.rmdir()
                except Exception:
                    pass

    @classmethod
    def load_shard(cls, shard_dir: Path, verify_checksums: bool = True) -> tuple[list[dict[Hashable, Any]], dict[str, Any]]:
        """Load shard data and verify checksums."""
        if verify_checksums:
            cls.verify_shard_checksums(shard_dir)

        trans_df = pd.read_parquet(shard_dir / "transitions.parquet")
        with open(shard_dir / "manifest.json") as f:
            manifest = json.load(f)

        transitions_list = trans_df.to_dict(orient="records")
        return transitions_list, manifest

    @classmethod
    def verify_shard_checksums(cls, shard_dir: Path) -> bool:
        """Verify SHA256 checksums of all shard files."""
        checksum_path = shard_dir / "checksums.sha256"
        if not checksum_path.exists():
            raise FileNotFoundError(f"Checksum file missing: {checksum_path}")

        expected: dict[str, str] = {}
        with open(checksum_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        expected[parts[1].strip()] = parts[0].strip()

        for fname, exp_sha in expected.items():
            if fname == "checksums.sha256":
                continue
            fpath = shard_dir / fname
            if not fpath.exists():
                raise FileNotFoundError(f"Shard file missing: {fpath}")
            act_sha = cls._hash_file(fpath)
            if act_sha != exp_sha:
                raise ValueError(f"Checksum mismatch for {fname}: expected {exp_sha}, got {act_sha}")

        return True

    @classmethod
    def _generate_checksums(cls, shard_dir: Path) -> dict[str, str]:
        checksums = {}
        for p in shard_dir.glob("*"):
            if p.is_file() and p.name != "checksums.sha256" and not p.name.endswith(".tmp"):
                checksums[p.name] = cls._hash_file(p)
        return checksums

    @classmethod
    def _hash_file(cls, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()
