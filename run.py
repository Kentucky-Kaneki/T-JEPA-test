"""
T-JEPA end-to-end pipeline on the Adult dataset.

Usage:
    python run.py [--epochs-pretrain N] [--epochs-downstream N] [--device cpu|cuda]

Steps:
  1. Download & preprocess data
  2. Pre-train T-JEPA (SSL, no labels)
  3. Extract representations
  4. Train downstream MLP (with T-JEPA representations)
  5. Train baseline MLP (on raw features, for comparison)
  6. Report test accuracy for both
"""

import argparse
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from data       import load_adult, build_loaders
from model      import TJEPA
from downstream import DownstreamMLP
from trainer    import (
    pretrain_tjepa,
    extract_representations,
    train_downstream,
    compute_accuracy,
)


def main(args):
    device = args.device
    print(f"\n{'='*55}")
    print("  T-JEPA  |  Adult Dataset  |  ICLR 2025 Replication")
    print(f"{'='*55}\n")

    # ── 1. Data ──────────────────────────────────────────
    print("[ 1/6 ] Loading & preprocessing data …")
    df, target_col, num_cols, cat_cols = load_adult()
    print(f"        Dataset shape: {df.shape}  "
          f"({len(num_cols)} numerical, {len(cat_cols)} categorical)")

    train_loader, val_loader, test_loader, prep = build_loaders(
        df, target_col, num_cols, cat_cols,
        batch_size=args.batch_size,
    )
    feature_dims = prep.feature_dims
    d = len(feature_dims)
    num_classes  = df[target_col].nunique()
    print(f"        Features (d): {d}  |  Classes: {num_classes}")

    # ── 2. Build T-JEPA model ────────────────────────────
    print("\n[ 2/6 ] Building T-JEPA model …")
    tjepa = TJEPA(
        feature_dims   = feature_dims,
        hidden_dim     = args.hidden_dim,
        num_heads      = args.num_heads,
        num_layers     = args.num_layers,
        ffn_dim        = args.ffn_dim,
        dropout        = 0.0,
        pred_dim       = args.pred_dim,
        pred_heads     = 2,
        pred_layers    = 2,
        ema_decay      = args.ema_decay,
        mask_min_ctx   = 0.10,
        mask_max_ctx   = 0.75,
        mask_min_tgt   = 0.10,
        mask_max_tgt   = 0.50,
        num_tgt_masks  = 4,
        num_reg_tokens = 1,
    )
    n_params = sum(p.numel() for p in tjepa.parameters() if p.requires_grad)
    print(f"        Trainable parameters: {n_params:,}")

    # ── 3. Pre-train T-JEPA ──────────────────────────────
    print(f"\n[ 3/6 ] Pre-training T-JEPA for {args.epochs_pretrain} epochs …")
    pretrain_losses = pretrain_tjepa(
        tjepa, train_loader, val_loader,
        num_epochs  = args.epochs_pretrain,
        lr          = args.lr_pretrain,
        device      = device,
        verbose     = True,
    )
    print(f"        Final pre-train loss: {pretrain_losses[-1]:.4f}")

    # ── 4. Extract representations ───────────────────────
    print("\n[ 4/6 ] Extracting T-JEPA representations …")
    train_h, train_y = extract_representations(tjepa, train_loader, device)
    val_h,   val_y   = extract_representations(tjepa, val_loader,   device)
    test_h,  test_y  = extract_representations(tjepa, test_loader,  device)
    print(f"        Representation shape: {train_h.shape}")

    # ── 5. Train downstream classifier (T-JEPA) ──────────
    print(f"\n[ 5/6 ] Training downstream MLP on T-JEPA representations "
          f"({args.epochs_downstream} epochs) …")
    classifier = DownstreamMLP(
        d           = d,
        hidden_dim  = args.hidden_dim,
        num_classes = num_classes,
        mlp_hidden  = 256,
        num_layers  = 4,
        dropout     = 0.3,
        projection  = "mean",
    )
    result = train_downstream(
        classifier,
        train_h, train_y,
        val_h,   val_y,
        num_epochs  = args.epochs_downstream,
        lr          = args.lr_downstream,
        device      = device,
        verbose     = True,
    )
    tjepa_test_acc = compute_accuracy(classifier, test_h, test_y, device)
    print(f"        Best val acc: {result['best_val_acc']:.4f}  "
          f"(epoch {result['best_epoch']})")
    print(f"        ✓ T-JEPA+MLP test accuracy: {tjepa_test_acc:.4f}")

    # ── 6. Baseline MLP on raw features ─────────────────
    print("\n[ 6/6 ] Training baseline sklearn MLP on raw features …")
    df_train_val = df.sample(frac=0.9, random_state=42)
    df_test_raw  = df.drop(df_train_val.index)

    X_train = prep.transform(
        df_train_val.drop(columns=[target_col])
        if target_col in df_train_val.columns
        else df_train_val
    )
    # Flatten the per-feature list into a single matrix
    import numpy as np
    def flatten(encoded):
        return np.concatenate(encoded, axis=1)

    df_tr  = df.sample(frac=0.8, random_state=42)
    df_te  = df.drop(df_tr.index)
    X_tr   = flatten(prep.transform(df_tr.drop(columns=[target_col])))
    y_tr   = df_tr[target_col].values
    X_te   = flatten(prep.transform(df_te.drop(columns=[target_col])))
    y_te   = df_te[target_col].values

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    baseline_mlp = MLPClassifier(
        hidden_layer_sizes=(256, 256, 256),
        max_iter=200, random_state=42,
        early_stopping=True, validation_fraction=0.1,
        verbose=False,
    )
    baseline_mlp.fit(X_tr_s, y_tr)
    baseline_acc = accuracy_score(y_te, baseline_mlp.predict(X_te_s))
    print(f"        ✓ Baseline MLP test accuracy: {baseline_acc:.4f}")

    # ── Summary ──────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  RESULTS SUMMARY")
    print(f"{'='*55}")
    print(f"  Baseline MLP (raw features):  {baseline_acc:.4f}")
    print(f"  T-JEPA + MLP (pre-trained):   {tjepa_test_acc:.4f}")
    delta = tjepa_test_acc - baseline_acc
    sign  = "+" if delta >= 0 else ""
    print(f"  Improvement:                  {sign}{delta:.4f}")
    print(f"{'='*55}\n")

    return tjepa_test_acc, baseline_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T-JEPA on Adult dataset")

    # Architecture
    parser.add_argument("--hidden-dim",   type=int,   default=64)
    parser.add_argument("--num-heads",    type=int,   default=4)
    parser.add_argument("--num-layers",   type=int,   default=4)
    parser.add_argument("--ffn-dim",      type=int,   default=256)
    parser.add_argument("--pred-dim",     type=int,   default=32)

    # Training
    parser.add_argument("--epochs-pretrain",    type=int,   default=50)
    parser.add_argument("--epochs-downstream",  type=int,   default=50)
    parser.add_argument("--batch-size",         type=int,   default=256)
    parser.add_argument("--lr-pretrain",        type=float, default=3e-4)
    parser.add_argument("--lr-downstream",      type=float, default=1e-3)
    parser.add_argument("--ema-decay",          type=float, default=0.998)

    # System
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    main(args)
