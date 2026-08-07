"""
CybORG Environment Inspection Utility.

Usage:
    python cyborg/inspect_env.py [--scenario Scenario1b] [--steps 10]
"""

import argparse
import os
import inspect
from CybORG import CybORG
from CybORG.Simulator.Scenarios import FileReaderScenarioGenerator
from CybORG.Agents.Wrappers import BlueTableWrapper, ChallengeWrapper
from CybORG.Agents import B_lineAgent, RedMeanderAgent
from CybORG.Simulator.Actions import Sleep


def get_default_scenario_path(scenario_name: str = "Scenario1b") -> str:
    cyborg_dir = os.path.dirname(inspect.getfile(CybORG))
    scen_path = os.path.join(cyborg_dir, 'Simulator', 'Scenarios', 'scenario_files', f'{scenario_name}.yaml')
    if not os.path.exists(scen_path):
        raise FileNotFoundError(f"Scenario file not found at: {scen_path}")
    return scen_path


def main():
    parser = argparse.ArgumentParser(description="Inspect CybORG observations and action spaces.")
    parser.add_argument("--scenario", type=str, default="Scenario1b", help="Scenario name (e.g. Scenario1b, Scenario2)")
    parser.add_argument("--red-agent", type=str, default="bline", choices=["bline", "meander"], help="Red agent policy")
    parser.add_argument("--steps", type=int, default=10, help="Number of timesteps to run")
    args = parser.parse_args()

    scen_path = get_default_scenario_path(args.scenario)
    sg = FileReaderScenarioGenerator(scen_path)

    red_policy = B_lineAgent() if args.red_agent == "bline" else RedMeanderAgent()
    agents = {'Red': red_policy}

    print(f"\n{'='*60}")
    print(f" CybORG Environment Inspection | Scenario: {args.scenario} | Red: {args.red_agent.upper()}")
    print(f"{'='*60}\n")

    cyborg = CybORG(scenario_generator=sg, agents=agents)
    btw = BlueTableWrapper(env=cyborg, output_mode='table')

    res = btw.reset(agent='Blue')
    print("Initial Host State Table (t=0):")
    print(res.observation)

    for t in range(1, args.steps + 1):
        step_res = btw.step(agent='Blue', action=Sleep())
        print(f"\n--- Timestep {t:2d} | Reward: {step_res.reward} ---")
        print(step_res.observation)


if __name__ == "__main__":
    main()
