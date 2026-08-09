"""
Unit and integration tests for ObservationMultiplexer and ActionMapper.

Verifies single-step simulator stepping, view parity against official wrappers,
dynamic Sleep action index resolution, and strict oracle data separation.
"""

import inspect
from pathlib import Path
import numpy as np
import pytest
import CybORG as cyborg_pkg

from cyber_jepa.env.observation_multiplexer import ObservationMultiplexer
from cyber_jepa.env.action_mapper import ActionMapper
from CybORG import CybORG
from CybORG.Agents.Wrappers import ChallengeWrapper
from CybORG.Simulator.Scenarios import FileReaderScenarioGenerator
from CybORG.Agents import B_lineAgent, RedMeanderAgent


def get_scenario1b_path() -> str:
    cyborg_dir = Path(inspect.getfile(cyborg_pkg)).parent
    path = cyborg_dir / "Simulator" / "Scenarios" / "scenario_files" / "Scenario1b.yaml"
    assert path.exists(), f"Scenario1b.yaml not found at {path}"
    return str(path)


def test_dynamic_action_mapper():
    """Verify ActionMapper resolves Sleep index dynamically without assuming 0."""
    scen_path = get_scenario1b_path()
    sg = FileReaderScenarioGenerator(scen_path)
    cyborg = CybORG(scenario_generator=sg, agents={"Red": B_lineAgent()}, seed=123)
    cw = ChallengeWrapper(env=cyborg, agent_name="Blue")
    cw.reset()

    mapper = ActionMapper(cw.env.possible_actions)
    sleep_idx = mapper.get_sleep_index()
    assert 0 <= sleep_idx < mapper.num_actions

    sleep_spec = mapper.resolve(sleep_idx)
    assert sleep_spec.action_type == "Sleep"
    assert sleep_spec.discrete_index == sleep_idx

    # Resolve another action (e.g., Analyse)
    for i in range(mapper.num_actions):
        spec = mapper.resolve(i)
        assert spec.discrete_index == i
        assert isinstance(spec.action_type, str)


def test_single_step_execution():
    """Verify ObservationMultiplexer performs exactly one simulator step per transition."""
    scen_path = get_scenario1b_path()
    mux = ObservationMultiplexer(scenario_path=scen_path, red_agent_type="bline", seed=42)

    obs0, oracle0 = mux.reset()
    assert obs0.timestep == 0
    assert mux.step_counter == 0

    for step_num in range(1, 6):
        obs, reward, act_spec, done, info, oracle = mux.step(action=mux.action_mapper.get_sleep_index())
        assert obs.timestep == step_num
        assert mux.step_counter == step_num
        assert len(obs.flat) == 52
        assert len(obs.host_features) == 13
        assert oracle.transition_id == f"ep0_t{step_num}"


def test_view_parity_with_official_wrapper():
    """Verify ObservationMultiplexer flat vector matches official ChallengeWrapper output."""
    scen_path = get_scenario1b_path()
    seed = 999

    # 1. Standard ChallengeWrapper
    sg1 = FileReaderScenarioGenerator(scen_path)
    cyborg1 = CybORG(scenario_generator=sg1, agents={"Red": B_lineAgent()}, seed=seed)
    cw1 = ChallengeWrapper(env=cyborg1, agent_name="Blue")
    cyborg1.set_seed(seed)
    vec1 = cw1.reset()

    # 2. ObservationMultiplexer
    mux2 = ObservationMultiplexer(scenario_path=scen_path, red_agent_type="bline", seed=seed)
    obs2, _ = mux2.reset(seed=seed)

    np.testing.assert_allclose(
        vec1, np.array(obs2.flat, dtype=np.float32),
        err_msg="Reset flat vector parity failure between Multiplexer and ChallengeWrapper"
    )

    # Step 5 transitions under Sleep
    sleep_idx = mux2.action_mapper.get_sleep_index()
    for _ in range(5):
        vec1, r1, done1, _ = cw1.step(action=sleep_idx)
        obs2, r2, _, done2, _, _ = mux2.step(action=sleep_idx)

        np.testing.assert_allclose(
            vec1, np.array(obs2.flat, dtype=np.float32),
            err_msg="Step flat vector parity failure between Multiplexer and ChallengeWrapper"
        )
        assert r1 == r2
        assert done1 == done2


def test_oracle_isolation():
    """Verify BlueObservation contains no oracle ground-truth field names."""
    scen_path = get_scenario1b_path()
    mux = ObservationMultiplexer(scenario_path=scen_path, red_agent_type="meander", seed=777)
    obs, oracle = mux.reset()

    obs_dict = vars(obs)
    oracle_keys = {"host_compromise_status", "attacker_present", "red_stage", "critical_server_compromised"}

    for key in obs_dict.keys():
        assert key not in oracle_keys, f"Oracle key '{key}' leaked into BlueObservation!"

    # Ensure oracle object contains valid ground truth
    assert len(oracle.host_compromise_status) == 13
    assert len(oracle.attacker_present) == 13
    assert oracle.red_stage in ("recon", "access", "priv_esc", "impact", "unknown")


def test_underlying_simulator_instrumentation():
    """Verify exactly 1 reset per episode, 1 step per action, shared transition IDs, and zero orphan oracles."""
    scen_path = get_scenario1b_path()
    mux = ObservationMultiplexer(scenario_path=scen_path, red_agent_type="bline", seed=100)

    transition_ids = set()

    for ep_idx in range(2):
        ep_id = f"ep_{ep_idx}"
        obs0, oracle0 = mux.reset(seed=100 + ep_idx, episode_id=ep_id)
        assert oracle0.transition_id == f"{ep_id}_t0"
        assert oracle0.transition_id not in transition_ids
        transition_ids.add(oracle0.transition_id)

        for step_idx in range(1, 6):
            obs, r, act, done, info, oracle = mux.step(
                action=mux.action_mapper.get_sleep_index(),
                episode_id=ep_id,
            )
            t_id = f"{ep_id}_t{step_idx}"
            assert obs.timestep == step_idx
            assert oracle.t == step_idx
            assert oracle.episode_id == ep_id
            assert oracle.transition_id == t_id
            assert t_id not in transition_ids
            transition_ids.add(t_id)

    assert mux.underlying_reset_count == 2
    assert mux.underlying_step_count == 10
    assert len(transition_ids) == 12 # 2 reset obs + 10 step obs


def test_seed_replay_determinism():
    """Verify identical seeds produce deterministic observations and oracle states."""
    scen_path = get_scenario1b_path()
    mux1 = ObservationMultiplexer(scenario_path=scen_path, red_agent_type="meander", seed=999)
    mux2 = ObservationMultiplexer(scenario_path=scen_path, red_agent_type="meander", seed=999)

    obs1, oracle1 = mux1.reset(seed=999, episode_id="ep_det")
    obs2, oracle2 = mux2.reset(seed=999, episode_id="ep_det")

    assert obs1.flat == obs2.flat
    assert oracle1.host_compromise_status == oracle2.host_compromise_status

    for step_num in range(1, 6):
        o1, r1, a1, d1, _, or1 = mux1.step(action=step_num % mux1.action_mapper.num_actions, episode_id="ep_det")
        o2, r2, a2, d2, _, or2 = mux2.step(action=step_num % mux2.action_mapper.num_actions, episode_id="ep_det")

        assert o1.flat == o2.flat
        assert r1 == r2
        assert a1.discrete_index == a2.discrete_index
        assert or1.host_compromise_status == or2.host_compromise_status

