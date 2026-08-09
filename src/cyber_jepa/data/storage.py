"""
Dataset storage manager for CybORG Cyber-JEPA datasets.

Manages saving and loading compressed NumPy arrays (observations), Parquet tables
(transitions and oracle sidecars), zstandard-compressed JSONL (raw blue observations),
JSON manifests, and SHA256 checksums.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Hashable
import numpy as np
import pandas as pd
import zstandard as zstd

from cyber_jepa.data.schema import BlueObservation, ActionSpec, OracleLabels, Transition, UNKNOWN_TOKEN


class DatasetStorageManager:
    """Manages atomic dataset shard reads/writes and checksum verification."""

    SCHEMA_VERSION = "1.0.0"

    @classmethod
    def save_shard(
        self,
        shard_dir: Path,
        transitions: list[Transition],
        oracle_labels: list[OracleLabels],
        manifest: dict[str, Any],
    ) -> dict[str, str]:
        """Save a dataset shard to directory format with checksums."""
        shard_dir.mkdir(parents=True, exist_ok=True)

        # 1. Observations NPZ
        flat_obs = np.array([t.obs.flat for t in transitions], dtype=np.float32)
        flat_next_obs = np.array([t.next_obs.flat for t in transitions], dtype=np.float32)
        host_known_masks = np.array([t.obs.host_known_mask for t in transitions], dtype=bool)
        next_host_known_masks = np.array([t.next_obs.host_known_mask for t in transitions], dtype=bool)

        obs_npz_path = shard_dir / "observations.npz"
        np.savez_compressed(
            obs_npz_path,
            flat=flat_obs,
            next_flat=flat_next_obs,
            host_known_mask=host_known_masks,
            next_host_known_mask=next_host_known_masks,
        )

        # 2. Transitions Parquet
        trans_rows = []
        for t in transitions:
            trans_rows.append({
                "dataset_id": t.dataset_id,
                "episode_id": t.episode_id,
                "transition_id": t.transition_id,
                "t": t.t,
                "collection_seed": t.collection_seed,
                "scenario_name": t.scenario_name,
                "scenario_hash": t.scenario_hash,
                "red_policy": t.red_policy,
                "blue_policy": t.blue_policy,
                "action_idx": t.action.discrete_index,
                "action_type": t.action.action_type,
                "target_kind": t.action.target_kind,
                "target_host": t.action.target_host,
                "target_subnet": t.action.target_subnet,
                "reward": t.reward,
                "done": t.done,
            })
        trans_df = pd.DataFrame(trans_rows)
        trans_parquet_path = shard_dir / "transitions.parquet"
        trans_df.to_parquet(trans_parquet_path, index=False)

        # 3. Oracle Labels Parquet (Physically Separate Sidecar)
        oracle_rows = []
        for o in oracle_labels:
            row = {
                "transition_id": o.transition_id,
                "episode_id": o.episode_id,
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
        oracle_parquet_path = shard_dir / "oracle_labels.parquet"
        oracle_df.to_parquet(oracle_parquet_path, index=False)

        # 4. Raw Blue JSONL.ZST
        raw_jsonl_path = shard_dir / "raw_blue.jsonl.zst"
        cctx = zstd.ZstdCompressor(level=3)
        with open(raw_jsonl_path, "wb") as f_out, cctx.stream_writer(f_out) as writer:
            for t in transitions:
                line_data = {
                    "transition_id": t.transition_id,
                    "t": t.t,
                    "raw_blue": t.obs.raw_blue,
                    "host_features": t.obs.host_features,
                    "blue_events": t.obs.blue_events,
                }
                writer.write((json.dumps(line_data) + "\n").encode("utf-8"))

        # 5. Schema JSON
        schema_data = {
            "schema_version": self.SCHEMA_VERSION,
            "obs_flat_dim": 52,
            "num_hosts": 13,
            "host_slots": transitions[0].obs.host_ids if transitions else [],
        }
        schema_path = shard_dir / "schema.json"
        with open(schema_path, "w") as f:
            json.dump(schema_data, f, indent=2)

        # 6. Manifest JSON
        manifest_data = dict(manifest)
        manifest_data["schema_version"] = self.SCHEMA_VERSION
        manifest_data["num_transitions"] = len(transitions)
        manifest_data["num_episodes"] = len(set(t.episode_id for t in transitions))
        manifest_path = shard_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f, indent=2)

        # 7. Checksums SHA256
        checksums = self._generate_checksums(shard_dir)
        checksum_path = shard_dir / "checksums.sha256"
        with open(checksum_path, "w") as f:
            for fname, sha in sorted(checksums.items()):
                f.write(f"{sha}  {fname}\n")

        return checksums

    @classmethod
    def load_shard(self, shard_dir: Path, verify_checksums: bool = True) -> tuple[list[dict[Hashable, Any]], dict[str, Any]]:
        """Load shard data and verify checksums."""
        if verify_checksums:
            self.verify_shard_checksums(shard_dir)

        trans_df = pd.read_parquet(shard_dir / "transitions.parquet")
        obs_data = np.load(shard_dir / "observations.npz")
        with open(shard_dir / "manifest.json", "r") as f:
            manifest = json.load(f)

        transitions_list = trans_df.to_dict(orient="records")
        return transitions_list, manifest

    @classmethod
    def verify_shard_checksums(self, shard_dir: Path) -> bool:
        """Verify SHA256 checksums of all shard files."""
        checksum_path = shard_dir / "checksums.sha256"
        if not checksum_path.exists():
            raise FileNotFoundError(f"Checksum file missing: {checksum_path}")

        expected: dict[str, str] = {}
        with open(checksum_path, "r") as f:
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
            act_sha = self._hash_file(fpath)
            if act_sha != exp_sha:
                raise ValueError(f"Checksum mismatch for {fname}: expected {exp_sha}, got {act_sha}")

        return True

    @classmethod
    def _generate_checksums(self, shard_dir: Path) -> dict[str, str]:
        checksums = {}
        for p in shard_dir.glob("*"):
            if p.is_file() and p.name != "checksums.sha256":
                checksums[p.name] = self._hash_file(p)
        return checksums

    @classmethod
    def _hash_file(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()
