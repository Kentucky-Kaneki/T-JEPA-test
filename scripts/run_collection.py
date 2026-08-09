"""
Full Trajectory Collection & Characterization Pipeline Runner.

Collects the complete 2 Red x 3 Blue x 5 Seeds matrix (30 shards, 150,000 transitions max)
and runs characterization and acceptance gating before model development.
"""

import argparse
import inspect
from pathlib import Path

from cyber_jepa.data.collector import collect_shard
from cyber_jepa.data.characterize import analyze_dataset_shards
import CybORG as cyborg_pkg


RED_POLICIES = ["bline", "meander"]
BLUE_POLICIES = ["sleep", "random", "coverage"]
COLLECTION_SEEDS = [1001, 2003, 3005]


def get_scenario1b_path() -> str:
    cyborg_dir = Path(inspect.getfile(cyborg_pkg)).parent
    path = cyborg_dir / "Simulator" / "Scenarios" / "scenario_files" / "Scenario1b.yaml"
    assert path.exists(), f"Scenario1b.yaml not found at {path}"
    return str(path)


def run_full_collection(
    output_base_dir: Path = Path("data/shards"),
    reports_dir: Path = Path("reports"),
    episodes: int = 100,
    max_steps: int = 50,
) -> None:
    scenario_path = get_scenario1b_path()
    output_base_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f" Starting CybORG Trajectory Collection Pipeline")
    print(f" Matrix: {len(RED_POLICIES)} Red x {len(BLUE_POLICIES)} Blue x {len(COLLECTION_SEEDS)} Seeds = 30 Shards")
    print(f" Episodes per shard: {episodes}, Max steps: {max_steps}")
    print(f"=======================================================\n")

    completed = 0
    total_shards = len(RED_POLICIES) * len(BLUE_POLICIES) * len(COLLECTION_SEEDS)

    for red in RED_POLICIES:
        for blue in BLUE_POLICIES:
            for seed in COLLECTION_SEEDS:
                shard_name = f"shard_{red}_{blue}_{seed}"
                shard_dir = output_base_dir / shard_name

                completed += 1
                print(f"[{completed:2d}/{total_shards}] Collecting {shard_name} ...")

                collect_shard(
                    scenario_path=scenario_path,
                    red_policy=red,
                    blue_policy=blue,
                    episodes=episodes,
                    max_steps=max_steps,
                    seed=seed,
                    dataset_id="cyborg_scenario1b_full",
                    output_dir=shard_dir,
                )

    print(f"\n[+] Trajectory collection complete! Saved {total_shards} shards to {output_base_dir}\n")

    print(f"Running dataset characterization & acceptance audit ...")
    report_dict, md_text = analyze_dataset_shards(output_base_dir, reports_dir)

    print(f"\n=======================================================")
    print(f" Dataset Characterization Summary:")
    print(f" Total Transitions: {report_dict['total_transitions']:,}")
    print(f" Total Episodes:    {report_dict['total_episodes']:,}")
    print(f" Zero-Change Rate:  {report_dict['zero_change_fraction']:.2%}")
    print(f" Acceptance Gates:  {'PASSED' if report_dict['acceptance_gates_passed'] else 'FAILED'}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CybORG Trajectory Collection & Diagnostic Pipeline.")
    parser.add_argument("--output-dir", type=str, default="data/shards")
    parser.add_argument("--reports-dir", type=str, default="reports")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=50)
    args = parser.parse_args()

    run_full_collection(
        output_base_dir=Path(args.output_dir),
        reports_dir=Path(args.reports_dir),
        episodes=args.episodes,
        max_steps=args.max_steps,
    )
