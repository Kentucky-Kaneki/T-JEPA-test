"""
Data contracts and schema definitions for CybORG Cyber-JEPA.

Strict dataclass contracts enforcing zero simulator leakage, explicit unknown/visibility
encoding, and unambiguous state/action representations.
"""

from dataclasses import dataclass, field
from typing import Any


UNKNOWN_TOKEN = "UNKNOWN"
NONE_TOKEN = "NONE"

# Fixed Scenario1b Host Slots in canonical order
SCENARIO1B_HOST_SLOTS = [
    "Defender",
    "Enterprise0",
    "Enterprise1",
    "Enterprise2",
    "Op_Host0",
    "Op_Host1",
    "Op_Host2",
    "Op_Server0",
    "User0",
    "User1",
    "User2",
    "User3",
    "User4",
]

# Fixed Scenario1b Subnets
SCENARIO1B_SUBNET_SLOTS = [
    "Enterprise",
    "Operational",
    "User",
]


@dataclass
class HostFeatureState:
    """Canonical Blue-visible host feature representation."""
    hostname: str
    subnet: str
    ip_address: str
    activity: str           # 'None', 'Scan', 'Exploit', 'UNKNOWN'
    compromised: str        # 'No', 'Unknown', 'User', 'Privileged', 'UNKNOWN'
    is_known: bool          # True if host has been observed by Blue, False if unobserved


@dataclass
class BlueObservation:
    """Canonical Blue observation contract (Section 4.1)."""
    flat: list[float]                        # Fixed 52-dim vector
    host_features: list[dict[str, Any]]      # 13 host dicts
    host_known_mask: list[bool]              # 13-element boolean visibility mask
    host_ids: list[str]                      # Fixed 13 hostnames
    subnet_ids: list[str]                    # Categorical subnets per host
    blue_events: list[dict[str, Any]]        # Legitimate Blue-visible events
    raw_blue: dict[str, Any]                 # JSON-normalized Blue observation
    timestep: int

    def __post_init__(self) -> None:
        if len(self.flat) != 52:
            raise ValueError(f"Flat observation vector must be 52-dim, got {len(self.flat)}")
        if len(self.host_ids) != 13:
            raise ValueError(f"Host IDs must contain 13 slots, got {len(self.host_ids)}")
        if len(self.host_known_mask) != 13:
            raise ValueError(f"Host known mask must contain 13 slots, got {len(self.host_known_mask)}")


@dataclass
class ActionSpec:
    """Semantic Blue Action contract (Section 4.2)."""
    discrete_index: int
    action_type: str                         # 'Sleep', 'Monitor', 'Analyse', 'Remove', etc.
    target_kind: str                         # 'none', 'host', 'subnet', 'session'
    target_host: str                         # hostname or UNKNOWN/NONE
    target_subnet: str                       # subnet or UNKNOWN/NONE
    parameters: dict[str, Any]
    is_valid: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "discrete_index": self.discrete_index,
            "action_type": self.action_type,
            "target_kind": self.target_kind,
            "target_host": self.target_host,
            "target_subnet": self.target_subnet,
            "parameters": self.parameters,
            "is_valid": self.is_valid,
        }


@dataclass
class OracleLabels:
    """Physically isolated simulator ground truth sidecar (Section 4.3)."""
    transition_id: str
    episode_id: str
    t: int
    host_compromise_status: dict[str, str]   # hostname -> 'clean', 'user', 'system'
    attacker_present: dict[str, bool]        # hostname -> bool
    red_stage: str                           # 'recon', 'access', 'priv_esc', 'impact', 'unknown'
    critical_server_compromised: bool


@dataclass
class Transition:
    """Single-step environment transition contract (Section 4.3)."""
    dataset_id: str
    episode_id: str
    transition_id: str
    t: int
    collection_seed: int
    scenario_name: str
    scenario_hash: str
    red_policy: str
    blue_policy: str
    obs: BlueObservation
    action: ActionSpec
    reward: float
    next_obs: BlueObservation
    done: bool
