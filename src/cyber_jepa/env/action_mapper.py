"""
Action space mapping and semantic resolution for CybORG Blue actions.

Resolves discrete action indices to ActionSpec dataclasses dynamically
without hardcoding action index 0 as Sleep.
"""

from typing import Any
from CybORG.Simulator.Actions import Action, Sleep
from cyber_jepa.data.schema import ActionSpec, NONE_TOKEN


class ActionMapper:
    """Dynamic discrete-to-semantic action mapper for CybORG wrappers."""

    def __init__(self, possible_actions: list[Action]):
        if not possible_actions:
            raise ValueError("possible_actions list cannot be empty")

        self.possible_actions = possible_actions
        self.num_actions = len(possible_actions)
        self.sleep_index = self._find_sleep_index()

    def _find_sleep_index(self) -> int:
        """Find index of Sleep action dynamically."""
        for idx, act in enumerate(self.possible_actions):
            if isinstance(act, Sleep) or type(act).__name__ == "Sleep" or getattr(act, "name", "") == "Sleep":
                return idx
        raise RuntimeError("Could not resolve 'Sleep' action in possible_actions list")

    def get_sleep_index(self) -> int:
        """Return the dynamic discrete index of the Sleep action."""
        return self.sleep_index

    def resolve(self, action_idx: int) -> ActionSpec:
        """Resolve a discrete action index to an ActionSpec contract."""
        if not (0 <= action_idx < self.num_actions):
            raise ValueError(f"Action index {action_idx} out of bounds [0, {self.num_actions - 1}]")

        act = self.possible_actions[action_idx]
        action_type = type(act).__name__
        params: dict[str, Any] = {}

        # Extract non-private attributes as parameters
        if hasattr(act, "__dict__"):
            for k, v in act.__dict__.items():
                if not k.startswith("_") and k != "priority":
                    params[k] = str(v) if not isinstance(v, (int, float, bool)) else v

        target_host = str(params.get("hostname", params.get("ip_address", NONE_TOKEN)))
        target_subnet = str(params.get("subnet", NONE_TOKEN))

        if target_host != NONE_TOKEN:
            target_kind = "host"
        elif target_subnet != NONE_TOKEN:
            target_kind = "subnet"
        elif "session" in params:
            target_kind = "session"
        else:
            target_kind = "none"

        return ActionSpec(
            discrete_index=action_idx,
            action_type=action_type,
            target_kind=target_kind,
            target_host=target_host,
            target_subnet=target_subnet,
            parameters=params,
            is_valid=True,
        )

    def get_action_object(self, action_idx: int) -> Action:
        """Return the underlying CybORG Action instance."""
        if not (0 <= action_idx < self.num_actions):
            raise ValueError(f"Action index {action_idx} out of bounds [0, {self.num_actions - 1}]")
        return self.possible_actions[action_idx]
