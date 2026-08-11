"""
Data contracts and schema definitions for CybORG Cyber-JEPA.

Strict dataclass contracts enforcing zero simulator leakage, explicit unknown/visibility
encoding, canonical transition representations, and sidecar oracle linkage.
"""

from dataclasses import dataclass
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

# Fixed Action Types mapping to integer IDs (0..15)
SCENARIO1B_ACTION_TYPES = [
    "Sleep",
    "Monitor",
    "Analyse",
    "Remove",
    "Restore",
    "DecoyDefender",
    "DecoyEnterprise0",
    "DecoyEnterprise1",
    "DecoyEnterprise2",
    "DecoyOp_Host0",
    "DecoyOp_Host1",
    "DecoyOp_Host2",
    "DecoyOp_Server0",
    "DecoyUser0",
    "DecoyUser1",
    "DecoyUser2",
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
    action_type_id: int = 0
    target_host_id: int = 0
    target_subnet_id: int = 0

    def __post_init__(self) -> None:
        if self.action_type in SCENARIO1B_ACTION_TYPES:
            self.action_type_id = SCENARIO1B_ACTION_TYPES.index(self.action_type)
        else:
            self.action_type_id = 0

        if self.target_host in SCENARIO1B_HOST_SLOTS:
            self.target_host_id = SCENARIO1B_HOST_SLOTS.index(self.target_host) + 1
        else:
            self.target_host_id = 0 # 0 for NONE / UNKNOWN

        if self.target_subnet in SCENARIO1B_SUBNET_SLOTS:
            self.target_subnet_id = SCENARIO1B_SUBNET_SLOTS.index(self.target_subnet) + 1
        else:
            self.target_subnet_id = 0 # 0 for NONE / UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        return {
            "discrete_index": self.discrete_index,
            "action_type": self.action_type,
            "action_type_id": self.action_type_id,
            "target_kind": self.target_kind,
            "target_host": self.target_host,
            "target_host_id": self.target_host_id,
            "target_subnet": self.target_subnet,
            "target_subnet_id": self.target_subnet_id,
            "parameters": self.parameters,
            "is_valid": self.is_valid,
        }


@dataclass
class OracleLabels:
    """Physically isolated simulator ground truth sidecar (Section 4.3)."""
    transition_id: str
    trajectory_id: str
    split_group_id: str
    t: int
    host_compromise_status: dict[str, str]   # hostname -> 'clean', 'user', 'system'
    attacker_present: dict[str, bool]        # hostname -> bool
    red_stage: str                           # 'recon', 'access', 'priv_esc', 'impact', 'unknown'
    critical_server_compromised: bool


@dataclass
class Transition:
    """Single-step environment transition contract (Phase 2 canonical schema)."""
    dataset_id: str
    trajectory_id: str
    split_group_id: str
    transition_id: str
    seed: int
    step_index: int
    terminated: bool
    truncated: bool
    flat_obs: list[float]                    # 52-dim
    next_flat_obs: list[float]               # 52-dim
    host_features: list[dict[str, Any]]      # 13 host dicts
    next_host_features: list[dict[str, Any]] # 13 host dicts
    known_host_mask: list[bool]              # 13 bools
    next_known_host_mask: list[bool]         # 13 bools
    blue_events: list[dict[str, Any]]
    action_discrete_index: int
    action_type: str
    action_type_id: int
    host_target: str
    host_target_id: int
    subnet_target: str
    subnet_target_id: int
    action_parameters: dict[str, Any]
    action_valid: bool
    reward: float                            # Diagnostics only
    oracle_transition_id: str                # Link to sidecar
    scenario_name: str = "Scenario1b"
    scenario_hash: str = ""
    red_policy: str = "bline"
    blue_policy: str = "random"
    obs: BlueObservation | None = None
    action: ActionSpec | None = None
    next_obs: BlueObservation | None = None
    done: bool = False

    @property
    def t(self) -> int:
        return self.step_index
