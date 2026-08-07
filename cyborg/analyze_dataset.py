"""
Trajectory Dataset Diagnostic and Sparsity Analysis Script.

Analyzes collected trajectory datasets for:
  1. Step-to-step state change sparsity (percentage of features changed between t and t+k).
  2. Host-level compromise/activity state changes across horizons k in {1, 2, 4, 8, 16}.
  3. Action distributions and reward statistics.

Usage:
    python cyborg/analyze_dataset.py --dataset data/test_trajectories.pt
"""

import argparse
import os
import torch
import numpy as np


def analyze_trajectory_dataset(dataset_path: str):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    dataset = torch.load(dataset_path, weights_only=False)
    num_transitions = len(dataset)

    print(f"\n{'='*65}")
    print(f" Trajectory Dataset Analysis | File: {dataset_path}")
    print(f"{'='*65}")
    print(f"Total Transitions: {num_transitions:,}")

    if num_transitions == 0:
        return

    episodes = set(t["episode"] for t in dataset)
    print(f"Episodes: {len(episodes)}")

    # 1. Action & Reward statistics
    actions = [t["action"] for t in dataset]
    rewards = [t["reward"] for t in dataset]
    print(f"Mean Reward per Step: {np.mean(rewards):.4f} +/- {np.std(rewards):.4f}")
    print(f"Reward Range: [{np.min(rewards):.2f}, {np.max(rewards):.2f}]")

    # 2. Vector observation sparsity (t vs t+1)
    vector_diffs = [(t["next_obs_vector"] != t["obs_vector"]).float().mean().item() for t in dataset]
    print(f"\n--- 1D Vector Observation Sparsity (t -> t+1) ---")
    print(f"Average % of features changed per step: {np.mean(vector_diffs) * 100:.2f}%")
    print(f"Zero-change step ratio (O_{{t+1}} == O_t): {sum(d == 0 for d in vector_diffs) / len(vector_diffs) * 100:.2f}%")

    # 3. Host Table observation changes (t vs t+1)
    host_changes = 0
    total_hosts = 0
    activity_changes = 0
    compromise_changes = 0

    for t in dataset:
        obs_tbl = {h["hostname"]: h for h in t["obs_table"]}
        nxt_tbl = {h["hostname"]: h for h in t["next_obs_table"]}

        for host, data in obs_tbl.items():
            total_hosts += 1
            nxt_data = nxt_tbl.get(host, {})
            if data["activity"] != nxt_data.get("activity"):
                activity_changes += 1
                host_changes += 1
            elif data["compromised"] != nxt_data.get("compromised"):
                compromise_changes += 1
                host_changes += 1

    print(f"\n--- Host Table Observation Sparsity (t -> t+1) ---")
    print(f"Total Host Evaluations: {total_hosts:,}")
    print(f"Host State Change Ratio: {host_changes / max(total_hosts, 1) * 100:.2f}%")
    print(f"Activity Changes: {activity_changes:,} ({activity_changes / max(total_hosts, 1) * 100:.2f}%)")
    print(f"Compromise Status Changes: {compromise_changes:,} ({compromise_changes / max(total_hosts, 1) * 100:.2f}%)")
    print(f"{'='*65}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze trajectory dataset statistics.")
    parser.add_argument("--dataset", type=str, default="data/test_trajectories.pt")
    args = parser.parse_args()

    analyze_trajectory_dataset(args.dataset)


if __name__ == "__main__":
    main()
