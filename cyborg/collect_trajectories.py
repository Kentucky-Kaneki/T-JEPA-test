"""
Trajectory Data Collection Script for CybORG JEPA Training.

Collects transitions:
  (t, obs_vector, obs_table, action, reward, next_obs_vector, next_obs_table, done, ground_truth)
across multiple episodes and policy combinations.

Usage:
    python cyborg/collect_trajectories.py --episodes 20 --red-agent bline --blue-policy random --output data/trajectories_bline_random.pt
"""

import argparse
import os
import inspect
import random
import torch
import numpy as np
from tqdm import tqdm

from CybORG import CybORG
from CybORG.Simulator.Scenarios import FileReaderScenarioGenerator
from CybORG.Agents.Wrappers import ChallengeWrapper, BlueTableWrapper
from CybORG.Agents import B_lineAgent, RedMeanderAgent
from CybORG.Simulator.Actions import Sleep


def get_default_scenario_path(scenario_name: str = "Scenario1b") -> str:
    cyborg_dir = os.path.dirname(inspect.getfile(CybORG))
    return os.path.join(cyborg_dir, 'Simulator', 'Scenarios', 'scenario_files', f'{scenario_name}.yaml')


def parse_table_obs(table_obs) -> list[dict]:
    """Extract list of host dicts from BlueTableWrapper observation."""
    hosts = []
    # table_obs is a PrettyTable instance
    for row in table_obs._rows:
        hosts.append({
            "subnet": str(row[0]),
            "ip_address": str(row[1]),
            "hostname": str(row[2]),
            "activity": str(row[3]),
            "compromised": str(row[4]),
        })
    return hosts


def collect_trajectories(
    scenario_name: str = "Scenario1b",
    red_agent_type: str = "bline",
    blue_policy_type: str = "random",
    episodes: int = 20,
    max_steps: int = 50,
    seed: int = 42,
) -> list[dict]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    scen_path = get_default_scenario_path(scenario_name)
    sg = FileReaderScenarioGenerator(scen_path)

    red_agent = B_lineAgent() if red_agent_type == "bline" else RedMeanderAgent()
    agents = {'Red': red_agent}

    dataset = []

    for ep in tqdm(range(episodes), desc=f"Collecting ({red_agent_type} vs {blue_policy_type})"):
        cyborg = CybORG(scenario_generator=sg, agents=agents, seed=seed + ep)
        cw = ChallengeWrapper(env=cyborg, agent_name='Blue')
        btw = BlueTableWrapper(env=cyborg, output_mode='table')

        obs_vec = cw.reset()
        table_res = btw.reset(agent='Blue')
        obs_table = parse_table_obs(table_res.observation)

        ep_transitions = []

        for t in range(max_steps):
            if blue_policy_type == "sleep":
                action_idx = 0  # Sleep action in discrete space
            elif blue_policy_type == "random":
                action_idx = cw.action_space.sample()
            else:
                action_idx = 0

            next_obs_vec, reward, done, info = cw.step(action=action_idx)
            next_table_res = btw.step(agent='Blue', action=Sleep())
            next_obs_table = parse_table_obs(next_table_res.observation)

            transition = {
                "episode": ep,
                "step": t,
                "obs_vector": torch.tensor(obs_vec, dtype=torch.float32),
                "obs_table": obs_table,
                "action": action_idx,
                "reward": float(reward),
                "next_obs_vector": torch.tensor(next_obs_vec, dtype=torch.float32),
                "next_obs_table": next_obs_table,
                "done": bool(done),
            }
            ep_transitions.append(transition)

            obs_vec = next_obs_vec
            obs_table = next_obs_table

            if done:
                break

        dataset.extend(ep_transitions)

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Collect CybORG defensive trajectory datasets.")
    parser.add_argument("--scenario", type=str, default="Scenario1b")
    parser.add_argument("--red-agent", type=str, default="bline", choices=["bline", "meander"])
    parser.add_argument("--blue-policy", type=str, default="random", choices=["sleep", "random"])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--output", type=str, default="data/trajectories_bline_random.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"\nCollecting {args.episodes} episodes under {args.red_agent} (Red) vs {args.blue_policy} (Blue) …")
    dataset = collect_trajectories(
        scenario_name=args.scenario,
        red_agent_type=args.red_agent,
        blue_policy_type=args.blue_policy,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    torch.save(dataset, args.output)
    print(f"[+] Saved {len(dataset)} transitions to {args.output}\n")


if __name__ == "__main__":
    main()
