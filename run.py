"""
T-JEPA end-to-end pipeline on NASim.

Step 3 extends the architecture work from Steps 1 and 2 into a proper
self-supervised NASim loop:
  1. Create the NASim environment
  2. Collect rollout-grounded transitions with an exploratory policy
  3. Split train/val by episode and build loaders
  4. Pre-train T-JEPA on masked current-state plus action -> next-state latent

Step 4 attaches a frozen-latent on-policy actor-critic head:
  5. Freeze the JEPA encoder completely
  6. Collect fresh current-policy NASim rollouts each RL epoch
  7. Update policy/value heads only from that rollout
  8. Compare behavior against a random-policy baseline
"""

import argparse

import numpy as np
import torch

from data import (
    build_nasim_loaders,
    collect_rollout_transitions,
    get_env_dims,
    make_nasim_env,
)
from model import TJEPA
from rl import LatentActorCritic, evaluate_control_policy, train_on_policy_a2c
from trainer import pretrain_tjepa


def _print_pretrain_summary(train_stats: list[dict], scenario: str) -> None:
    """Prints a compact training summary using the tracked JEPA statistics."""
    if not train_stats:
        return

    first = train_stats[0]
    last = train_stats[-1]

    min_std = min(stats["repr_std"] for stats in train_stats)
    avg_kl = sum(stats["kl_loss"] for stats in train_stats) / len(train_stats)
    avg_var = sum(stats["var_loss"] for stats in train_stats) / len(train_stats)
    num_collapse_epochs = sum(1 for stats in train_stats if stats["repr_std"] < 0.01)

    temporal_first = first["temporal_loss"]
    temporal_last = last["temporal_loss"]
    temporal_decreased = temporal_last < temporal_first * 0.95
    avg_temporal = sum(stats["temporal_loss"] for stats in train_stats) / len(train_stats)

    print(f"\n  -- Pre-training stats: {scenario} ---------------------")
    print(
        f"  pred_loss:      epoch 1 = {first['pred_loss']:.4f}  ->  final = {last['pred_loss']:.4f}"
    )
    print(
        f"  temporal_loss:  epoch 1 = {temporal_first:.4f}  ->  final = {temporal_last:.4f}"
    )
    print(f"  repr_std:       min={min_std:.4f}  (collapse warnings: {num_collapse_epochs})")
    print(f"  kl_loss:        avg={avg_kl:.4f}  (final={last['kl_loss']:.4f})")
    print(f"  var_loss:       avg={avg_var:.4f}  (final={last['var_loss']:.4f})")

    print("\n  Heuristic checks:")
    loss_slow = (last["pred_loss"] / max(first["pred_loss"], 1e-8)) > 0.1
    print(f"    Pred loss dropped slowly (not instant):   {'Y' if loss_slow else 'N'}")
    print(f"    Temporal loss has signal (avg > 0.001):   {'Y' if avg_temporal > 0.001 else 'N'}")
    print(
        f"    Temporal loss decreased:                  {'Y' if temporal_decreased else '~  may need more epochs'}"
    )
    print(
        f"    No representation collapse (std >= 0.01): {'Y' if num_collapse_epochs == 0 else f'N  ({num_collapse_epochs} epochs)'}"
    )
    print(f"    KL finite and positive:                   {'Y' if 0 < avg_kl < 1e4 else 'N'}")


def _print_rollout_summary(rollout: dict[str, np.ndarray], num_actions: int) -> None:
    """Prints collection statistics relevant to Step 3."""
    actions = rollout["actions"]
    rewards = rollout["rewards"]
    episode_ids = rollout["episode_ids"]
    dones = rollout["dones"]

    unique_episodes, episode_lengths = np.unique(episode_ids, return_counts=True)
    action_counts = np.bincount(actions, minlength=num_actions)

    print(f"        Collected:         {len(actions):,} transitions")
    print(f"        Episodes:          {len(unique_episodes):,}")
    print(f"        Avg episode len:   {episode_lengths.mean():.2f}")
    print(f"        Max episode len:   {episode_lengths.max()}")
    print(f"        Terminal samples:  {int(dones.sum())}")
    print(f"        Reward mean/std:   {rewards.mean():.4f} / {rewards.std():.4f}")
    print(
        f"        Action coverage:   {np.count_nonzero(action_counts)}/{num_actions} "
        f"(min={action_counts.min()}, max={action_counts.max()})"
    )


def main(args):
    device = args.device
    if args.epochs_joint > 0:
        raise ValueError(
            "Joint JEPA + RL training is disabled for this proof-of-concept. "
            "Keep --epochs-joint 0 so the JEPA encoder remains frozen."
        )
    if args.epochs_a2c > 0 and args.max_steps_per_episode is None:
        args.max_steps_per_episode = 100
    if args.epochs_a2c > 0 and not (0.0 <= args.milestone_reward_scale <= 0.25):
        raise ValueError("--milestone-reward-scale must stay in [0.0, 0.25].")

    total_steps = 5 + int(args.epochs_a2c > 0)

    print(f"\n{'=' * 62}")
    print(f"  T-JEPA  |  NASim Self-Supervised Loop  |  Scenario: {args.scenario}")
    print(f"{'=' * 62}\n")

    print(f"[ 1/{total_steps} ] Creating NASim environment ...")
    env = make_nasim_env(args.scenario, fully_obs=args.fully_obs, seed=args.seed)
    num_hosts, host_features, num_actions = get_env_dims(env)
    feature_dims = [host_features] * num_hosts

    print(f"        Scenario:         {args.scenario}")
    print(f"        Obs shape:        ({num_hosts}, {host_features})")
    print(f"        Host tokens:      {num_hosts}")
    print(f"        Num actions:      {num_actions}")
    print(f"        Partial obs:      {not args.fully_obs}")

    print(
        f"\n[ 2/{total_steps} ] Collecting {args.n_transitions:,} transitions "
        f"with {args.exploration_policy!r} exploration ..."
    )
    rollout = collect_rollout_transitions(
        env,
        n_transitions=args.n_transitions,
        exploration_policy=args.exploration_policy,
        seed=args.seed,
        max_steps_per_episode=args.max_steps_per_episode,
    )
    _print_rollout_summary(rollout, num_actions)
    if float(rollout["rewards"].std()) < 1e-6:
        print("        Warning: reward variance is near zero; RL stages will have weak learning signal.")

    print(f"\n[ 3/{total_steps} ] Building rollout-aware data loaders ...")
    train_loader, val_loader, prep = build_nasim_loaders(
        rollout["states"],
        rollout["actions"],
        rollout["next_states"],
        num_hosts=num_hosts,
        host_features=host_features,
        episode_ids=rollout["episode_ids"],
        batch_size=args.batch_size,
        seed=args.seed,
        split_by_episode=not args.allow_transition_leakage,
    )
    print(f"        Train batches:    {len(train_loader)}")
    print(f"        Val batches:      {len(val_loader)}")
    print(f"        Split by episode: {not args.allow_transition_leakage}")
    print(f"        Feature dims:     {prep.feature_dims}")

    print(f"\n[ 4/{total_steps} ] Building T-JEPA model ...")
    tjepa = TJEPA(
        feature_dims=feature_dims,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_dim=args.ffn_dim,
        dropout=0.0,
        pred_dim=args.pred_dim,
        pred_heads=2,
        pred_layers=2,
        ema_decay=args.ema_decay,
        mask_min_ctx=args.mask_min_ctx,
        mask_max_ctx=args.mask_max_ctx,
        mask_min_tgt=args.mask_min_tgt,
        mask_max_tgt=args.mask_max_tgt,
        num_tgt_masks=args.num_tgt_masks,
        normalize=not args.no_normalize,
        var_reg_weight=args.var_reg_weight,
        kl_weight=args.kl_weight,
        num_actions=num_actions,
        temporal_pred_layers=2,
        temporal_weight=args.temporal_weight,
    )
    trainable_params = sum(p.numel() for p in tjepa.parameters() if p.requires_grad)
    print(f"        Trainable params: {trainable_params:,}")
    print(f"        normalize:        {not args.no_normalize}")
    print(f"        mask ctx:         [{args.mask_min_ctx:.2f}, {args.mask_max_ctx:.2f}]")
    print(f"        mask tgt:         [{args.mask_min_tgt:.2f}, {args.mask_max_tgt:.2f}]")
    print(f"        num_tgt_masks:    {args.num_tgt_masks}")
    print(f"        temporal_weight:  {args.temporal_weight}")

    print(f"\n[ 5/{total_steps} ] Pre-training for {args.epochs_pretrain} epochs ...")
    pretrain_losses, train_stats = pretrain_tjepa(
        tjepa,
        train_loader,
        val_loader,
        num_epochs=args.epochs_pretrain,
        lr=args.lr_pretrain,
        device=device,
        verbose=True,
    )
    print(f"        Final pred_loss:     {pretrain_losses[-1]:.4f}")
    print(f"        Final temporal_loss: {train_stats[-1]['temporal_loss']:.4f}")

    _print_pretrain_summary(train_stats, args.scenario)

    if args.epochs_a2c > 0:
        print(f"\n[ 6/{total_steps} ] Frozen-encoder on-policy A2C proof-of-concept ...")
        print("        RL source:          fresh current-policy NASim rollouts only")
        print("        Encoder updates:    disabled")
        print("        Reward base:        raw NASim task reward")
        print(
            f"        Rollout shape:      {args.rl_episodes_per_epoch} episodes/epoch, "
            f"max {args.max_steps_per_episode} steps/episode"
        )
        print(f"        RL seeds:           {args.rl_num_seeds}")
        if args.milestone_reward_scale > 0.0:
            print(
                "        Milestone rewards:  enabled, small, one-time, environment-consistent "
                f"(+{args.milestone_reward_scale:.3f} per progress event)"
            )
        else:
            print("        Milestone rewards:  disabled")
        enough_budget = (
            args.epochs_a2c >= args.min_substantive_rl_epochs
            and args.rl_num_seeds >= 2
            and args.eval_episodes >= 10
            and args.random_baseline_episodes >= 10
        )
        if not enough_budget:
            print(
                "        Budget label:       diagnostic only; too short for a substantive "
                "learning claim"
            )

        seed_results = []
        for seed_idx in range(args.rl_num_seeds):
            rl_seed = args.seed + 100_000 * seed_idx
            torch.manual_seed(rl_seed)
            np.random.seed(rl_seed % (2**32 - 1))
            agent = LatentActorCritic(
                encoder=tjepa,
                num_actions=num_actions,
                hidden_dim=args.hidden_dim,
                actor_hidden_dim=args.a2c_hidden_dim,
                freeze_encoder=True,
            )

            random_env = make_nasim_env(args.scenario, fully_obs=args.fully_obs, seed=rl_seed + 10_000)
            random_stats = evaluate_control_policy(
                None,
                random_env,
                prep,
                num_episodes=args.random_baseline_episodes,
                max_steps_per_episode=args.max_steps_per_episode,
                random_policy=True,
                milestone_reward_scale=args.milestone_reward_scale,
                device=device,
                seed=rl_seed + 20_000,
            )
            print(f"\n        Seed {seed_idx + 1}/{args.rl_num_seeds} random baseline:")
            _print_control_report(random_stats, prefix="          ")

            a2c_history = train_on_policy_a2c(
                agent,
                prep,
                scenario=args.scenario,
                fully_obs=args.fully_obs,
                num_epochs=args.epochs_a2c,
                episodes_per_epoch=args.rl_episodes_per_epoch,
                max_steps_per_episode=args.max_steps_per_episode,
                gamma=args.gamma,
                lr=args.lr_a2c,
                value_coef=args.value_coef,
                entropy_coef=args.entropy_coef,
                milestone_reward_scale=args.milestone_reward_scale,
                device=device,
                seed=rl_seed,
                random_baseline=random_stats,
                fail_fast_epochs=args.fail_fast_epochs,
                min_control_margin=args.min_control_margin,
                eval_episodes=args.eval_episodes,
                verbose=True,
            )

            eval_env = make_nasim_env(args.scenario, fully_obs=args.fully_obs, seed=rl_seed + 30_000)
            eval_stats = evaluate_control_policy(
                agent,
                eval_env,
                prep,
                num_episodes=args.eval_episodes,
                max_steps_per_episode=args.max_steps_per_episode,
                deterministic=True,
                random_policy=False,
                milestone_reward_scale=args.milestone_reward_scale,
                device=device,
                seed=rl_seed + 40_000,
            )
            print(f"\n        Seed {seed_idx + 1}/{args.rl_num_seeds} learned-policy evaluation:")
            _print_control_report(eval_stats, prefix="          ")

            seed_results.append(
                {
                    "seed": rl_seed,
                    "random": random_stats,
                    "learned": eval_stats,
                    "failed_fast": bool(
                        a2c_history and a2c_history[-1].get("failed_poc", 0.0) > 0.0
                    ),
                }
            )

        random_agg = _aggregate_control_reports([result["random"] for result in seed_results])
        learned_agg = _aggregate_control_reports([result["learned"] for result in seed_results])
        print("\n        Aggregate random baseline:")
        _print_control_report(random_agg, prefix="          ")
        print("\n        Aggregate learned policy:")
        _print_control_report(learned_agg, prefix="          ")

        success_gain = (
            learned_agg["goal_success_rate"]
            > random_agg["goal_success_rate"] + args.min_success_rate_margin
        )
        return_gain = learned_agg["mean_return"] > random_agg["mean_return"] + args.min_control_margin
        progress_gain = (
            learned_agg["progress_events_per_episode"]
            > random_agg["progress_events_per_episode"] + args.min_progress_margin
        )
        diverse_enough = learned_agg["action_diversity"] >= args.min_action_diversity
        no_seed_failed_fast = not any(result["failed_fast"] for result in seed_results)
        substantive_success = (
            enough_budget
            and success_gain
            and return_gain
            and progress_gain
            and diverse_enough
            and no_seed_failed_fast
        )

        print("\n        Substantive RL verdict:")
        print(f"          Adequate budget:   {'Y' if enough_budget else 'N'}")
        print(f"          Success-rate gain: {'Y' if success_gain else 'N'}")
        print(f"          Return gain:       {'Y' if return_gain else 'N'}")
        print(f"          Progress gain:     {'Y' if progress_gain else 'N'}")
        print(f"          Diverse policy:    {'Y' if diverse_enough else 'N'}")
        print(f"          No fail-fast seed: {'Y' if no_seed_failed_fast else 'N'}")
        print(f"          Result:            {'SUCCESS' if substantive_success else 'FAILED'}")
        if not substantive_success:
            print(
                "          Interpretation:    no evidence of substantive control learning "
                "under a genuine reward signal"
            )

    print(f"\n{'=' * 62}\n")
    return train_stats


def _print_control_report(stats: dict[str, float], prefix: str = "") -> None:
    print(f"{prefix}goal success rate:  {stats['goal_success_rate']:.3f}")
    print(f"{prefix}avg episode return: {stats['mean_return']:.3f} +/- {stats['std_return']:.3f}")
    print(f"{prefix}avg episode length: {stats['mean_length']:.2f}")
    print(f"{prefix}terminal successes: {int(stats['terminal_successes'])}/{int(stats['episodes'])}")
    print(f"{prefix}action diversity:   {stats['action_diversity']:.3f} ({int(stats['unique_actions'])} actions)")
    print(f"{prefix}progress/episode:   {stats['progress_events_per_episode']:.3f}")


def _aggregate_control_reports(reports: list[dict[str, float]]) -> dict[str, float]:
    if not reports:
        raise ValueError("_aggregate_control_reports requires at least one report.")
    keys = reports[0].keys()
    return {
        key: float(np.mean([report[key] for report in reports]))
        for key in keys
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T-JEPA NASim self-supervised loop")

    parser.add_argument(
        "--scenario",
        type=str,
        default="tiny",
        help="NASim benchmark scenario: tiny, small, small-linear, ...",
    )
    parser.add_argument(
        "--n-transitions",
        type=int,
        default=50_000,
        help="Number of rollout transitions to collect",
    )
    parser.add_argument(
        "--exploration-policy",
        type=str,
        default="random",
        choices=["random", "balanced"],
        help="Exploration policy used during self-supervised data collection",
    )
    parser.add_argument(
        "--fully-obs",
        action="store_true",
        help="Use fully observable NASim instead of partial observability",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=None,
        help="Optional cap for rollout length during collection",
    )
    parser.add_argument(
        "--allow-transition-leakage",
        action="store_true",
        help="Use a plain transition split instead of keeping episodes intact",
    )

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument("--pred-dim", type=int, default=32)

    parser.add_argument("--epochs-pretrain", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr-pretrain", type=float, default=3e-4)
    parser.add_argument("--ema-decay", type=float, default=0.998)

    parser.add_argument("--mask-min-ctx", type=float, default=0.10)
    parser.add_argument("--mask-max-ctx", type=float, default=0.75)
    parser.add_argument("--mask-min-tgt", type=float, default=0.10)
    parser.add_argument("--mask-max-tgt", type=float, default=0.50)
    parser.add_argument("--num-tgt-masks", type=int, default=4)

    parser.add_argument("--var-reg-weight", type=float, default=0.04)
    parser.add_argument("--kl-weight", type=float, default=0.01)
    parser.add_argument("--temporal-weight", type=float, default=0.5)
    parser.add_argument("--no-normalize", action="store_true")

    parser.add_argument("--epochs-a2c", type=int, default=0)
    parser.add_argument("--lr-a2c", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--a2c-hidden-dim", type=int, default=128)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument(
        "--rl-episodes-per-epoch",
        type=int,
        default=8,
        help="Fresh current-policy NASim episodes collected for each on-policy RL epoch",
    )
    parser.add_argument(
        "--rl-num-seeds",
        type=int,
        default=3,
        help="Number of independently initialized policy-head seeds to train and evaluate",
    )
    parser.add_argument(
        "--random-baseline-episodes",
        type=int,
        default=20,
        help="Random-policy evaluation episodes under the same tiny scenario",
    )
    parser.add_argument(
        "--milestone-reward-scale",
        type=float,
        default=0.0,
        help="Optional small one-time bonus for true attack progress events; 0 disables it",
    )
    parser.add_argument(
        "--fail-fast-epochs",
        type=int,
        default=25,
        help="Evaluate and stop early after this many RL epochs if no behavioral gain is visible; 0 disables early stop",
    )
    parser.add_argument(
        "--min-substantive-rl-epochs",
        type=int,
        default=50,
        help="Minimum RL epochs required before the run can be labeled a substantive success",
    )
    parser.add_argument(
        "--min-control-margin",
        type=float,
        default=0.0,
        help="Minimum average-return margin required over random for a behavioral gain",
    )
    parser.add_argument(
        "--min-success-rate-margin",
        type=float,
        default=0.0,
        help="Minimum goal-success-rate margin required over random",
    )
    parser.add_argument(
        "--min-progress-margin",
        type=float,
        default=0.0,
        help="Minimum progress-events-per-episode margin required over random",
    )
    parser.add_argument(
        "--min-action-diversity",
        type=float,
        default=0.20,
        help="Minimum fraction of distinct actions used during learned-policy evaluation",
    )

    parser.add_argument("--epochs-joint", type=int, default=0)

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()
    main(args)
