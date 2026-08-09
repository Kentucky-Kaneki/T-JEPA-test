"""
Trajectory Data Collector for CybORG Cyber-JEPA.

Collects aligned transitions across Red x Blue x Seed matrix using single-step
ObservationMultiplexer and saves immutable dataset shards with checksums.
"""

import hashlib
import inspect
import random
from pathlib import Path
from typing import Any

import numpy as np

from cyber_jepa.data.schema import OracleLabels, Transition
from cyber_jepa.data.storage import DatasetStorageManager
from cyber_jepa.env.action_mapper import ActionMapper
from cyber_jepa.env.observation_multiplexer import ObservationMultiplexer


class CoverageDirectedPolicy:
    """Blue policy balancing action types before balancing host/subnet targets."""

    def __init__(self, action_mapper: ActionMapper, seed: int = 42):
        self.action_mapper = action_mapper
        self.rng = random.Random(seed)

        # Group discrete indices by action_type
        self.type_groups: dict[str, list[int]] = {}
        for i in range(action_mapper.num_actions):
            spec = action_mapper.resolve(i)
            self.type_groups.setdefault(spec.action_type, []).append(i)

        self.action_types = sorted(list(self.type_groups.keys()))

    def sample(self) -> int:
        """Sample action type first, then sample target within that action type."""
        chosen_type = self.rng.choice(self.action_types)
        chosen_index = self.rng.choice(self.type_groups[chosen_type])
        return chosen_index


def get_scenario1b_path() -> str:
    """Return absolute path to pinned Scenario1b.yaml scenario file."""
    import CybORG as cyborg_pkg
    cyborg_dir = Path(inspect.getfile(cyborg_pkg)).parent
    path = cyborg_dir / "Simulator" / "Scenarios" / "scenario_files" / "Scenario1b.yaml"
    assert path.exists(), f"Scenario1b.yaml not found at {path}"
    return str(path)


def compute_scenario_hash(scenario_path: str) -> str:
    """Compute SHA256 hash of the scenario file."""
    p = Path(scenario_path)
    if not p.exists():
        return "UNKNOWN_HASH"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def collect_shard(
    scenario_path: str,
    red_policy: str = "bline",
    blue_policy: str = "random",
    episodes: int = 100,
    max_steps: int = 50,
    seed: int = 1001,
    dataset_id: str = "dataset_full",
    output_dir: Path | None = None,
) -> tuple[list[Transition], list[OracleLabels], dict[str, Any]]:
    """Collect trajectory transitions for one Red x Blue x Seed shard."""
    random.seed(seed)
    np.random.seed(seed)

    scenario_hash = compute_scenario_hash(scenario_path)
    transitions: list[Transition] = []
    oracle_labels: list[OracleLabels] = []

    # Initialize multiplexer
    mux = ObservationMultiplexer(scenario_path=scenario_path, red_agent_type=red_policy, seed=seed)
    action_mapper = mux.action_mapper
    sleep_idx = action_mapper.get_sleep_index()

    # Blue policy sampler setup
    coverage_policy = CoverageDirectedPolicy(action_mapper, seed=seed * 31 + 7)
    random_policy_rng = random.Random(seed * 17 + 3)

    for ep in range(episodes):
        split_group_id = f"group_{seed}_{ep:03d}"
        trajectory_id = f"traj_{red_policy}_{blue_policy}_{seed}_{ep:03d}"
        ep_seed = seed + ep

        # Reset simulator ONCE per episode
        obs0, oracle0 = mux.reset(seed=ep_seed, trajectory_id=trajectory_id, split_group_id=split_group_id)

        current_obs = obs0

        for t in range(1, max_steps + 1):
            trans_id = f"{trajectory_id}_t{t:02d}"

            # Sample Blue action
            if blue_policy.lower() == "sleep":
                action_idx = sleep_idx
            elif blue_policy.lower() == "random":
                action_idx = random_policy_rng.randint(0, action_mapper.num_actions - 1)
            elif blue_policy.lower() in ("coverage", "coverage_directed"):
                action_idx = coverage_policy.sample()
            else:
                raise ValueError(f"Unknown blue_policy: {blue_policy}")

            # Single simulator step
            next_obs, reward, act_spec, done, info, next_oracle = mux.step(
                action=action_idx,
                trajectory_id=trajectory_id,
                split_group_id=split_group_id,
            )
            oracle_labels.append(next_oracle)

            trans = Transition(
                dataset_id=dataset_id,
                trajectory_id=trajectory_id,
                split_group_id=split_group_id,
                transition_id=trans_id,
                seed=ep_seed,
                step_index=t,
                terminated=done and t < max_steps,
                truncated=done and t >= max_steps,
                flat_obs=list(current_obs.flat),
                next_flat_obs=list(next_obs.flat),
                host_features=current_obs.host_features,
                next_host_features=next_obs.host_features,
                known_host_mask=list(current_obs.host_known_mask),
                next_known_host_mask=list(next_obs.host_known_mask),
                blue_events=next_obs.blue_events,
                action_discrete_index=act_spec.discrete_index,
                action_type=act_spec.action_type,
                action_type_id=act_spec.action_type_id,
                host_target=act_spec.target_host,
                host_target_id=act_spec.target_host_id,
                subnet_target=act_spec.target_subnet,
                subnet_target_id=act_spec.target_subnet_id,
                action_parameters=act_spec.parameters,
                action_valid=act_spec.is_valid,
                reward=reward,
                oracle_transition_id=trans_id,
                scenario_name="Scenario1b",
                scenario_hash=scenario_hash,
                red_policy=red_policy,
                blue_policy=blue_policy,
                obs=current_obs,
                action=act_spec,
                next_obs=next_obs,
                done=done,
            )
            transitions.append(trans)

            current_obs = next_obs
            if done:
                break

    manifest: dict[str, Any] = {
        "dataset_id": dataset_id,
        "scenario_name": "Scenario1b",
        "scenario_hash": scenario_hash,
        "red_policy": red_policy,
        "blue_policy": blue_policy,
        "collection_seed": seed,
        "requested_episodes": episodes,
        "max_steps": max_steps,
        "num_transitions": len(transitions),
        "num_oracle_records": len(oracle_labels),
    }

    if output_dir is not None:
        if output_dir.name.startswith("shard"):
            shard_dir = output_dir
        else:
            shard_dir = output_dir / f"shard_{red_policy}_{blue_policy}_{seed}"

        checksums = DatasetStorageManager.save_shard(
            shard_dir=shard_dir,
            transitions=transitions,
            oracle_labels=oracle_labels,
            manifest=manifest,
        )
        manifest["checksums"] = checksums

    return transitions, oracle_labels, manifest
