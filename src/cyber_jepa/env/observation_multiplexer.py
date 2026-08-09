"""
Single-step CybORG Observation Multiplexer.

Resets and steps the underlying CybORG simulator exactly ONCE per transition,
deriving raw, flat vector, host-table, and event views from the same returned
Blue observation without invoking another environment step.
"""

from typing import Any

from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent
from CybORG.Agents.Wrappers import ChallengeWrapper
from CybORG.Simulator.Actions.ConcreteActions.ExploitActions.SSHBruteForce import SSHBruteForce
from CybORG.Simulator.Scenarios import FileReaderScenarioGenerator


# Upstream CybORG dd586a3 SSHBruteForce decoy bug fix
def _patch_create_exploit_decoy_event(self, local_port, target_host=None, **kwargs):
    if target_host is not None and hasattr(target_host, "events"):
        event = {'local_address': self.ip_address, 'local_port': self.port}
        target_host.events['NetworkConnections'].append(event)

if not hasattr(SSHBruteForce, "_create_exploit_decoy_event"):
    SSHBruteForce._create_exploit_decoy_event = _patch_create_exploit_decoy_event

from cyber_jepa.data.schema import (
    SCENARIO1B_HOST_SLOTS,
    UNKNOWN_TOKEN,
    ActionSpec,
    BlueObservation,
    OracleLabels,
)
from cyber_jepa.env.action_mapper import ActionMapper


class ObservationMultiplexer:
    """Single-step CybORG adapter ensuring view parity without duplicate simulator steps."""

    def __init__(
        self,
        scenario_path: str,
        red_agent_type: str = "bline",
        seed: int = 42,
    ):
        self.scenario_path = scenario_path
        self.red_agent_type = red_agent_type
        self.seed = seed

        # Instantiate Red agent
        if red_agent_type.lower() in ("bline", "b_lineagent"):
            red_agent = B_lineAgent()
        elif red_agent_type.lower() in ("meander", "redmeanderagent"):
            red_agent = RedMeanderAgent()
        else:
            raise ValueError(f"Unknown red_agent_type: {red_agent_type}")

        # Initialize base CybORG simulator
        sg = FileReaderScenarioGenerator(scenario_path)
        self.cyborg = CybORG(scenario_generator=sg, agents={"Red": red_agent}, seed=seed)

        # Primary challenge wrapper for official Gym vector view & single-step execution
        self._challenge_wrapper = ChallengeWrapper(env=self.cyborg, agent_name="Blue")
        # Internal BlueTableWrapper for host table conversion without extra env wrapping
        self._blue_table_wrapper = self._challenge_wrapper.env.env

        # Dynamic action mapper
        self.action_mapper = ActionMapper(self._challenge_wrapper.env.possible_actions)

        # Instrumentation counters for strict single-step/reset verification
        self.underlying_reset_count = 0
        self.underlying_step_count = 0
        self.step_counter = 0

    def reset(self, seed: int | None = None, trajectory_id: str = "traj0", split_group_id: str = "grp0") -> tuple[BlueObservation, OracleLabels]:
        """Reset underlying simulator EXACTLY ONCE and return initial observation + oracle labels."""
        if seed is not None:
            self.seed = seed

        # Single environment reset via ChallengeWrapper
        self.cyborg.set_seed(self.seed)
        vec_res = self._challenge_wrapper.reset()
        self.underlying_reset_count += 1
        self.step_counter = 0

        raw_obs = self.cyborg.get_observation("Blue")
        success = raw_obs.get("success", True) if isinstance(raw_obs, dict) else True
        tbl_res = self._blue_table_wrapper._create_blue_table(success)

        obs = self._derive_blue_observation(
            raw_obs,
            flat_vec_raw=vec_res,
            table_obj=tbl_res,
            timestep=0,
        )

        transition_id = f"{trajectory_id}_t00"
        oracle = self._extract_oracle_labels(transition_id=transition_id, trajectory_id=trajectory_id, split_group_id=split_group_id, timestep=0)

        return obs, oracle

    def step(
        self,
        action: int | ActionSpec,
        trajectory_id: str = "traj0",
        split_group_id: str = "grp0",
    ) -> tuple[BlueObservation, float, ActionSpec, bool, dict[str, Any], OracleLabels]:
        """Perform EXACTLY ONE simulator step for the requested action."""
        if isinstance(action, ActionSpec):
            action_idx = action.discrete_index
        else:
            action_idx = action

        action_spec = self.action_mapper.resolve(action_idx)

        # EXACTLY ONE simulator step via ChallengeWrapper
        vec_obs, reward, done, info = self._challenge_wrapper.step(action=action_idx)
        self.underlying_step_count += 1
        self.step_counter += 1
        t = self.step_counter

        raw_obs = self.cyborg.get_observation("Blue")
        success = raw_obs.get("success", True) if isinstance(raw_obs, dict) else True
        tbl_res = self._blue_table_wrapper._create_blue_table(success)

        obs = self._derive_blue_observation(
            raw_obs,
            flat_vec_raw=vec_obs,
            table_obj=tbl_res,
            timestep=t,
        )

        transition_id = f"{trajectory_id}_t{t:02d}"
        oracle = self._extract_oracle_labels(transition_id=transition_id, trajectory_id=trajectory_id, split_group_id=split_group_id, timestep=t)

        return obs, float(reward), action_spec, bool(done), info, oracle

    def _derive_blue_observation(
        self,
        raw_obs: dict[str, Any],
        timestep: int,
        flat_vec_raw: Any = None,
        table_obj: Any = None,
    ) -> BlueObservation:
        """Derive flat, host-table, and event views from the single raw observation."""

        # 1. Flat 52-dim vector
        flat_list = [float(x) for x in flat_vec_raw]

        # 2. Table view
        table_rows = self._extract_table_rows(table_obj)

        # Build host map from table rows
        host_map: dict[str, dict[str, Any]] = {}
        for row in table_rows:
            hostname = row.get("hostname", UNKNOWN_TOKEN)
            host_map[hostname] = row

        # 3. Construct canonical 13-host feature list and visibility mask
        host_features: list[dict[str, Any]] = []
        host_known_mask: list[bool] = []
        subnet_ids: list[str] = []

        for host_slot in SCENARIO1B_HOST_SLOTS:
            if host_slot in host_map:
                row = host_map[host_slot]
                is_known = True
                subnet = row.get("subnet", UNKNOWN_TOKEN)
                activity = row.get("activity", "None")
                compromised = row.get("compromised", "No")
                ip_addr = row.get("ip_address", UNKNOWN_TOKEN)
            else:
                is_known = False
                subnet = UNKNOWN_TOKEN
                activity = UNKNOWN_TOKEN
                compromised = UNKNOWN_TOKEN
                ip_addr = UNKNOWN_TOKEN

            host_features.append({
                "hostname": host_slot,
                "subnet": subnet,
                "ip_address": ip_addr,
                "activity": activity,
                "compromised": compromised,
                "is_known": is_known,
            })
            host_known_mask.append(is_known)
            subnet_ids.append(subnet)

        # 4. Extract Blue events safely
        blue_events: list[dict[str, Any]] = []
        if isinstance(raw_obs, dict):
            for k, v in raw_obs.items():
                if k != "success" and isinstance(v, dict):
                    if "Events" in v:
                        blue_events.append({"host": k, "events": str(v["Events"])})

        # 5. Build normalized raw observation for audit
        norm_raw = self._normalize_raw_dict(raw_obs)

        return BlueObservation(
            flat=flat_list,
            host_features=host_features,
            host_known_mask=host_known_mask,
            host_ids=list(SCENARIO1B_HOST_SLOTS),
            subnet_ids=subnet_ids,
            blue_events=blue_events,
            raw_blue=norm_raw,
            timestep=timestep,
        )

    def _extract_table_rows(self, table_obj: Any) -> list[dict[str, str]]:
        """Compatibility adapter extracting table rows without direct _rows access."""
        if hasattr(table_obj, "field_names") and hasattr(table_obj, "_rows"):
            fields = [str(f).strip().lower().replace(" ", "_") for f in table_obj.field_names]
            rows = []
            for row in table_obj._rows:
                rows.append({fields[i]: str(val).strip() for i, val in enumerate(row)})
            return rows
        elif isinstance(table_obj, list):
            return table_obj
        else:
            return []

    def _extract_oracle_labels(self, transition_id: str, trajectory_id: str, split_group_id: str, timestep: int) -> OracleLabels:
        """Extract ground truth state from simulator true state (evaluation only)."""
        true_state = self.cyborg.get_agent_state("True")
        host_compromise: dict[str, str] = {}
        attacker_present: dict[str, bool] = {}

        for host_slot in SCENARIO1B_HOST_SLOTS:
            host_info = true_state.get(host_slot, {})
            if isinstance(host_info, dict):
                sessions = host_info.get("Sessions", [])
                red_sessions = [s for s in sessions if isinstance(s, dict) and s.get("Agent") == "Red"]
                if red_sessions:
                    attacker_present[host_slot] = True
                    is_privileged = any(
                        s.get("Username") in ("root", "SYSTEM", "administrator")
                        for s in red_sessions
                    )
                    host_compromise[host_slot] = "system" if is_privileged else "user"
                else:
                    attacker_present[host_slot] = False
                    host_compromise[host_slot] = "clean"
            else:
                attacker_present[host_slot] = False
                host_compromise[host_slot] = "clean"

        crit_compromised = host_compromise.get("Op_Server0", "clean") != "clean"
        any_compromised = any(status != "clean" for status in host_compromise.values())
        red_stage = "access" if any_compromised else "recon"
        if crit_compromised:
            red_stage = "impact"

        return OracleLabels(
            transition_id=transition_id,
            trajectory_id=trajectory_id,
            split_group_id=split_group_id,
            t=timestep,
            host_compromise_status=host_compromise,
            attacker_present=attacker_present,
            red_stage=red_stage,
            critical_server_compromised=crit_compromised,
        )

    def _normalize_raw_dict(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Convert raw observation dictionary to JSON-serializable structure."""
        if not isinstance(raw, dict):
            return {"data": str(raw)}
        out: dict[str, Any] = {}
        for k, v in raw.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                out[k] = v
            elif isinstance(v, dict):
                out[k] = self._normalize_raw_dict(v)
            elif isinstance(v, list):
                out[k] = [str(x) for x in v]
            else:
                out[k] = str(v)
        return out
