"""
T-JEPA end-to-end pipeline on NASim.

Step 3 extends the architecture work from Steps 1 and 2 into a proper
self-supervised NASim loop:
  1. Create the NASim environment
  2. Collect rollout-grounded transitions with an exploratory policy
  3. Split train/val by episode and build loaders
  4. Pre-train T-JEPA on masked current-state plus action -> next-state latent

Step 4 optionally attaches a frozen-latent offline actor-critic head:
  5. Freeze the JEPA encoder and train policy/value heads on z_t
  6. Evaluate the learned policy online in NASim

Step 5 optionally unfreezes the encoder in phases and jointly trains JEPA + RL:
  7. Mix stored and fresh policy rollouts
  8. Jointly optimize the representation and policy without letting RL dominate
"""

import argparse

import numpy as np
import torch

from data import (
    build_nasim_loaders,
    build_nasim_rl_loaders,
    collect_rollout_transitions,
    get_env_dims,
    make_nasim_env,
)
from model import TJEPA
from rl import LatentActorCritic, evaluate_policy, train_joint_tjepa_a2c, train_offline_a2c
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
    total_steps = 5 + int(args.epochs_a2c > 0) + int(args.epochs_joint > 0)

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

    agent = None

    if args.epochs_a2c > 0 or args.epochs_joint > 0:
        agent = LatentActorCritic(
            encoder=tjepa,
            num_actions=num_actions,
            hidden_dim=args.hidden_dim,
            actor_hidden_dim=args.a2c_hidden_dim,
            freeze_encoder=True,
        )

    if args.epochs_a2c > 0:
        print(f"\n[ 6/{total_steps} ] Training offline A2C on frozen JEPA latents ...")
        rl_train_loader, rl_val_loader, _, reward_stats = build_nasim_rl_loaders(
            rollout["states"],
            rollout["actions"],
            rollout["rewards"],
            rollout["next_states"],
            rollout["dones"],
            num_hosts=num_hosts,
            host_features=host_features,
            episode_ids=rollout["episode_ids"],
            batch_size=args.batch_size,
            seed=args.seed,
            split_by_episode=not args.allow_transition_leakage,
            reward_clip=args.reward_clip,
            prep=prep,
        )
        print(
            f"        Reward normalization: mean={reward_stats['reward_mean']:.4f}  "
            f"std={reward_stats['reward_std']:.4f}  clip={reward_stats['reward_clip']:.1f}"
        )
        if reward_stats["reward_std"] <= 1e-6:
            print("        Warning: normalized rewards are effectively constant after train split.")
        a2c_history = train_offline_a2c(
            agent,
            rl_train_loader,
            rl_val_loader,
            num_epochs=args.epochs_a2c,
            gamma=args.gamma,
            lr=args.lr_a2c,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            device=device,
            verbose=True,
        )
        final_a2c = a2c_history[-1]
        print(
            f"        Final A2C loss:      {final_a2c['loss']:.4f}  "
            f"(val={final_a2c['val_loss']:.4f})"
        )
        print(
            f"        Final policy/value:  {final_a2c['policy_loss']:.4f} / "
            f"{final_a2c['value_loss']:.4f}"
        )

        eval_env = make_nasim_env(args.scenario, fully_obs=args.fully_obs, seed=args.seed + 10_000)
        eval_stats = evaluate_policy(
            agent,
            eval_env,
            prep,
            num_episodes=args.eval_episodes,
            deterministic=True,
            device=device,
            seed=args.seed + 20_000,
            max_steps_per_episode=args.max_steps_per_episode,
        )
        print(
            f"        Eval return:         {eval_stats['mean_return']:.4f} +/- "
            f"{eval_stats['std_return']:.4f}"
        )
        print(f"        Eval ep length:      {eval_stats['mean_length']:.2f}")

    if args.epochs_joint > 0:
        step_idx = 6 if args.epochs_a2c == 0 else 7
        print(f"\n[ {step_idx}/{total_steps} ] Joint JEPA + RL training with phased unfreezing ...")
        tjepa.ema_decay = args.joint_ema_decay
        encoder_lr = args.lr_joint * args.encoder_lr_scale
        print(
            f"        lambda_rl={args.lambda_rl:.3f}  "
            f"encoder_lr={encoder_lr:.6f}  "
            f"jepa_lr={args.lr_joint:.6f}  "
            f"policy_lr={args.lr_a2c:.6f}"
        )
        print(
            f"        fresh transitions/epoch={args.fresh_transitions_per_epoch:,}  "
            f"EMA={args.joint_ema_decay:.3f}"
        )
        if args.epochs_a2c == 0:
            print("        Warning: joint training is starting without the frozen-encoder RL warmup stage.")

        joint_history = train_joint_tjepa_a2c(
            agent,
            base_rollout=rollout,
            preprocessor=prep,
            scenario=args.scenario,
            fully_obs=args.fully_obs,
            num_hosts=num_hosts,
            host_features=host_features,
            batch_size=args.batch_size,
            num_epochs=args.epochs_joint,
            fresh_transitions_per_epoch=args.fresh_transitions_per_epoch,
            gamma=args.gamma,
            lambda_rl=args.lambda_rl,
            encoder_lr=encoder_lr,
            jepa_lr=args.lr_joint,
            policy_lr=args.lr_a2c,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            reward_clip=args.reward_clip,
            device=device,
            seed=args.seed,
            max_steps_per_episode=args.max_steps_per_episode,
            max_recent_policy_rollouts=args.max_recent_policy_rollouts,
            verbose=True,
        )
        final_joint = joint_history[-1]
        print(
            f"        Final joint loss:    {final_joint['total_loss']:.4f}  "
            f"(val={final_joint['val_total_loss']:.4f})"
        )
        print(
            f"        Final JEPA/RL:      {final_joint['jepa_loss']:.4f} / "
            f"{final_joint['rl_loss']:.4f}"
        )
        print(
            f"        Final stage/std:    {final_joint['stage']} / "
            f"{final_joint['repr_std']:.4f}"
        )

        eval_env = make_nasim_env(args.scenario, fully_obs=args.fully_obs, seed=args.seed + 30_000)
        joint_eval = evaluate_policy(
            agent,
            eval_env,
            prep,
            num_episodes=args.eval_episodes,
            deterministic=True,
            device=device,
            seed=args.seed + 40_000,
            max_steps_per_episode=args.max_steps_per_episode,
        )
        print(
            f"        Joint eval return:   {joint_eval['mean_return']:.4f} +/- "
            f"{joint_eval['std_return']:.4f}"
        )
        print(f"        Joint eval length:   {joint_eval['mean_length']:.2f}")

    print(f"\n{'=' * 62}\n")
    return train_stats


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
    parser.add_argument("--reward-clip", type=float, default=5.0)
    parser.add_argument("--eval-episodes", type=int, default=5)

    parser.add_argument("--epochs-joint", type=int, default=0)
    parser.add_argument("--lr-joint", type=float, default=1e-4)
    parser.add_argument("--lambda-rl", type=float, default=0.1)
    parser.add_argument("--encoder-lr-scale", type=float, default=0.1)
    parser.add_argument("--joint-ema-decay", type=float, default=0.999)
    parser.add_argument("--fresh-transitions-per-epoch", type=int, default=2_000)
    parser.add_argument("--max-recent-policy-rollouts", type=int, default=2)

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()
    main(args)
