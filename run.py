"""
T-JEPA end-to-end pipeline on NASim (Task 2 — Temporal JEPA).

Usage:
    python run.py [--scenario tiny] [--n-transitions 50000] [--device cpu|cuda]
                  [--var-reg-weight F] [--kl-weight F] [--temporal-weight F]
                  [--no-normalize] [--fully-obs]

Steps:
  1. Create NASim environment, read dims
  2. Collect transitions via random policy
  3. Build train/val loaders
  4. Build T-JEPA model (spatial + temporal heads)
  5. Pre-train with combined L_masked + λ*L_future objective
  6. Report pre-training stats summary (Task2.md §10 checks)
"""

import argparse
import torch
import numpy as np

from data    import make_nasim_env, get_env_dims, collect_transitions, build_nasim_loaders
from model   import TJEPA
from trainer import pretrain_tjepa


# ─────────────────────────────────────────────
#  §10 Summary
# ─────────────────────────────────────────────

def _print_pretrain_summary(train_stats: list[dict], scenario: str) -> None:
    """Print Task2.md §10 heuristic checks from the training stats."""
    if not train_stats:
        return
    first = train_stats[0]
    last  = train_stats[-1]

    min_std = min(s["repr_std"]     for s in train_stats)
    avg_kl  = sum(s["kl_loss"]      for s in train_stats) / len(train_stats)
    avg_var = sum(s["var_loss"]      for s in train_stats) / len(train_stats)
    n_collapse = sum(1 for s in train_stats if s["repr_std"] < 0.01)

    # Task2 §10: temporal loss should decrease, not plateau
    temp_first = first["temporal_loss"]
    temp_last  = last["temporal_loss"]
    temp_decreased = temp_last < temp_first * 0.95

    # Check: temporal loss actually has signal (not trivially zero)
    avg_temp = sum(s["temporal_loss"] for s in train_stats) / len(train_stats)

    print(f"\n  -- Pre-training stats: {scenario} ---------------------")
    print(f"  pred_loss:      epoch 1 = {first['pred_loss']:.4f}  ->  final = {last['pred_loss']:.4f}")
    print(f"  temporal_loss:  epoch 1 = {temp_first:.4f}  ->  final = {temp_last:.4f}")
    print(f"  repr_std:       min={min_std:.4f}  (collapse warnings: {n_collapse})")
    print(f"  kl_loss:        avg={avg_kl:.4f}  (final={last['kl_loss']:.4f})")
    print(f"  var_loss:       avg={avg_var:.4f}  (final={last['var_loss']:.4f})")

    print(f"\n  §10 checks (Task2.md):")
    loss_slow = (last["pred_loss"] / max(first["pred_loss"], 1e-8)) > 0.1
    print(f"    Pred loss dropped slowly (not instant):   {'Y' if loss_slow else 'N'}")
    print(f"    Temporal loss has signal (avg > 0.001):   {'Y' if avg_temp > 0.001 else 'N'}")
    print(f"    Temporal loss decreased:                  {'Y' if temp_decreased else '~  may need more epochs'}")
    print(f"    No representation collapse (std >= 0.01): {'Y' if n_collapse == 0 else f'N  ({n_collapse} epochs)'}")
    print(f"    KL finite and positive:                   {'Y' if 0 < avg_kl < 1e4 else 'N'}")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main(args):
    device = args.device
    print(f"\n{'='*62}")
    print(f"  T-JEPA  |  Temporal JEPA  |  NASim: {args.scenario}")
    print(f"{'='*62}\n")

    # -- 1. Environment --------------------------------------------
    print("[ 1/5 ] Creating NASim environment ...")
    env = make_nasim_env(args.scenario, fully_obs=args.fully_obs, seed=args.seed)
    num_hosts, host_features, num_actions = get_env_dims(env)
    feature_dims = [host_features] * num_hosts

    print(f"        Scenario:     {args.scenario}")
    print(f"        Obs shape:    ({num_hosts}, {host_features})  "
          f"-> d={num_hosts} host tokens, each dim={host_features}")
    print(f"        Num actions:  {num_actions}")
    print(f"        Partial obs:  {not args.fully_obs}")

    # -- 2. Collect transitions ------------------------------------
    print(f"\n[ 2/5 ] Collecting {args.n_transitions:,} transitions (random policy) ...")
    states, actions, next_states = collect_transitions(
        env, n_transitions=args.n_transitions, seed=args.seed
    )
    print(f"        Collected:    {len(states):,} transitions")
    print(f"        Action dist:  min={actions.min()}, max={actions.max()}, "
          f"unique={len(np.unique(actions))}/{num_actions}")

    # -- 3. Build loaders -----------------------------------------
    print("\n[ 3/5 ] Building data loaders ...")
    train_loader, val_loader, prep = build_nasim_loaders(
        states, actions, next_states,
        num_hosts=num_hosts,
        host_features=host_features,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(f"        Train batches: {len(train_loader)}  |  "
          f"Val batches: {len(val_loader)}")

    # -- 4. Build T-JEPA model -------------------------------------
    print("\n[ 4/5 ] Building T-JEPA model ...")
    tjepa = TJEPA(
        feature_dims        = feature_dims,
        hidden_dim          = args.hidden_dim,
        num_heads           = args.num_heads,
        num_layers          = args.num_layers,
        ffn_dim             = args.ffn_dim,
        dropout             = 0.0,
        pred_dim            = args.pred_dim,
        pred_heads          = 2,
        pred_layers         = 2,
        ema_decay           = args.ema_decay,
        mask_min_ctx        = 0.10,
        mask_max_ctx        = 0.75,
        mask_min_tgt        = 0.10,
        mask_max_tgt        = 0.50,
        num_tgt_masks       = 4,
        normalize           = not args.no_normalize,
        var_reg_weight      = args.var_reg_weight,
        kl_weight           = args.kl_weight,
        num_actions         = num_actions,
        temporal_pred_layers = 2,
        temporal_weight     = args.temporal_weight,
    )
    trainable = sum(p.numel() for p in tjepa.parameters() if p.requires_grad)
    print(f"        Trainable params:  {trainable:,}")
    print(f"        normalize={not args.no_normalize}  "
          f"var_reg={args.var_reg_weight}  "
          f"kl={args.kl_weight}  "
          f"temporal_w={args.temporal_weight}")

    # -- 5. Pre-train ----------------------------------------------
    print(f"\n[ 5/5 ] Pre-training for {args.epochs_pretrain} epochs ...")
    pretrain_losses, train_stats = pretrain_tjepa(
        tjepa, train_loader, val_loader,
        num_epochs = args.epochs_pretrain,
        lr         = args.lr_pretrain,
        device     = device,
        verbose    = True,
    )
    print(f"        Final pred_loss:     {pretrain_losses[-1]:.4f}")
    print(f"        Final temporal_loss: {train_stats[-1]['temporal_loss']:.4f}")

    _print_pretrain_summary(train_stats, args.scenario)

    print(f"\n{'='*62}\n")
    return train_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T-JEPA Temporal on NASim")

    # Environment
    parser.add_argument("--scenario",      type=str,  default="tiny",
                        help="NASim benchmark scenario: tiny, small, small-linear, ...")
    parser.add_argument("--n-transitions", type=int,  default=50_000,
                        help="Transitions to collect via random policy")
    parser.add_argument("--fully-obs",     action="store_true",
                        help="Use fully observable NASim (default: partial obs / POMDP)")
    parser.add_argument("--seed",          type=int,  default=42)

    # Architecture
    parser.add_argument("--hidden-dim",  type=int,   default=64)
    parser.add_argument("--num-heads",   type=int,   default=4)
    parser.add_argument("--num-layers",  type=int,   default=4)
    parser.add_argument("--ffn-dim",     type=int,   default=256)
    parser.add_argument("--pred-dim",    type=int,   default=32)

    # Training
    parser.add_argument("--epochs-pretrain", type=int,   default=50)
    parser.add_argument("--batch-size",      type=int,   default=256)
    parser.add_argument("--lr-pretrain",     type=float, default=3e-4)
    parser.add_argument("--ema-decay",       type=float, default=0.998)

    # Loss weights
    parser.add_argument("--var-reg-weight",  type=float, default=0.04)
    parser.add_argument("--kl-weight",       type=float, default=0.01)
    parser.add_argument("--temporal-weight", type=float, default=0.5,
                        help="λ weighting temporal loss vs spatial loss")
    parser.add_argument("--no-normalize",    action="store_true")

    # System
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    main(args)
