"""
Frozen Linear Probing Diagnostic Pipeline for Cyber-JEPA.

Extracts frozen latents z_t without updating encoder parameters and fits regularized
logistic and ridge regression probes for host compromise, attacker presence, future compromise at t+4,
and critical operational-server risk. Enforces strict transition_id sidecar joins.
"""

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def compute_variance_normalized_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute variance-normalized R^2 = 1 - sum((y - y_hat)^2) / sum((y - y_bar)^2)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 1e-12:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Any,
    n_bootstraps: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute point estimate and percentile bootstrap confidence interval (lower, upper)."""
    rng = np.random.RandomState(seed)
    N = len(y_true)
    if N == 0:
        return 0.0, 0.0, 0.0

    point_est = float(metric_fn(y_true, y_pred))
    boot_scores: list[float] = []

    for _ in range(n_bootstraps):
        idxs = rng.randint(0, N, size=N)
        try:
            score = float(metric_fn(y_true[idxs], y_pred[idxs]))
            boot_scores.append(score)
        except Exception:
            continue

    if not boot_scores:
        return point_est, point_est, point_est

    alpha = (1.0 - ci_level) / 2.0
    lower = float(np.percentile(boot_scores, alpha * 100))
    upper = float(np.percentile(boot_scores, (1.0 - alpha) * 100))
    return point_est, lower, upper


class LinearProbeEvaluator:
    """Frozen linear probing evaluator for latent representation quality."""

    @classmethod
    def join_sidecar_labels(
        cls,
        transitions_df: pd.DataFrame,
        oracle_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Strict sidecar join by transition_id raising ValueError on unaligned or missing IDs."""
        if "transition_id" not in transitions_df.columns:
            raise KeyError("transitions_df missing required column 'transition_id'")
        if "transition_id" not in oracle_df.columns:
            raise KeyError("oracle_df sidecar missing required column 'transition_id'")

        trans_ids = set(transitions_df["transition_id"].dropna())
        oracle_ids = set(oracle_df["transition_id"].dropna())

        missing = trans_ids - oracle_ids
        if missing:
            raise ValueError(f"Strict sidecar join failed: {len(missing)} transition_ids missing in oracle sidecar (e.g. {list(missing)[:3]})")

        merged = pd.merge(transitions_df, oracle_df, on="transition_id", how="inner", suffixes=("", "_oracle"))
        if len(merged) != len(transitions_df):
            raise ValueError(f"Strict sidecar join mismatch: expected {len(transitions_df)} rows, got {len(merged)}")

        return merged

    @torch.no_grad()
    def extract_latents_and_labels(
        self,
        encoder: nn.Module,
        data_loader: Any,
        device: torch.device,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Extract frozen latents z_t, target labels, and transition IDs."""
        encoder.eval()
        encoder.to(device)

        latents_list: list[np.ndarray] = []
        labels_list: list[Any] = []
        trans_ids_list: list[str] = []

        for batch in data_loader:
            hist = batch["history_flat"].to(device)
            z = encoder(hist)
            if hasattr(z, "global_token"):
                z = z.global_token
            elif z.dim() == 4:
                z = z[:, -1, :, :].mean(dim=1)
            elif z.dim() == 3:
                z = z[:, -1, :]

            latents_list.append(z.cpu().numpy())

            if "labels" in batch:
                labels_list.extend(batch["labels"])
            else:
                labels_list.extend([0] * len(hist))

            if "transition_id" in batch:
                trans_ids_list.extend(batch["transition_id"])
            else:
                trans_ids_list.extend([f"t_{i}" for i in range(len(hist))])

        latents = np.concatenate(latents_list, axis=0)
        labels = np.array(labels_list)
        return latents, labels, trans_ids_list

    def train_and_evaluate_probe(
        self,
        train_latents: np.ndarray,
        train_labels: np.ndarray,
        test_latents: np.ndarray,
        test_labels: np.ndarray,
        is_classification: bool = True,
        c_val: float = 1.0,
    ) -> dict[str, Any]:
        """Fit probe (LogisticRegression or Ridge) and evaluate with 95% CIs and variance-normalized R^2."""

        if is_classification:
            if len(np.unique(train_labels)) < 2:
                return {
                    "macro_f1": 0.0,
                    "balanced_accuracy": 0.0,
                    "auroc": 0.0,
                    "ci_f1": (0.0, 0.0, 0.0),
                    "note": "Insufficient classes in training split (< 2)",
                }

            clf = LogisticRegression(C=c_val, max_iter=500, random_state=42)
            clf.fit(train_latents, train_labels)

            preds = clf.predict(test_latents)
            macro_f1, f1_low, f1_high = compute_bootstrap_ci(
                test_labels, preds, lambda y, p: f1_score(y, p, average="macro")
            )
            bal_acc = float(balanced_accuracy_score(test_labels, preds))

            auroc = 0.0
            try:
                if len(np.unique(test_labels)) > 1:
                    probs = clf.predict_proba(test_latents)
                    if probs.shape[1] == 2:
                        auroc = float(roc_auc_score(test_labels, probs[:, 1]))
                    else:
                        auroc = float(roc_auc_score(test_labels, probs, multi_class="ovr", average="macro"))
            except Exception:
                auroc = 0.0

            cm = confusion_matrix(test_labels, preds).tolist()

            return {
                "macro_f1": macro_f1,
                "macro_f1_ci_95": [f1_low, f1_high],
                "balanced_accuracy": bal_acc,
                "auroc": auroc,
                "confusion_matrix": cm,
                "c_val": c_val,
            }
        else:
            reg = Ridge(alpha=1.0 / max(1e-5, c_val), random_state=42)
            reg.fit(train_latents, train_labels)

            preds = reg.predict(test_latents)
            mse = float(np.mean((test_labels - preds) ** 2))
            r2, r2_low, r2_high = compute_bootstrap_ci(
                test_labels, preds, compute_variance_normalized_r2
            )

            return {
                "mse": mse,
                "r2_variance_normalized": r2,
                "r2_ci_95": [r2_low, r2_high],
                "c_val": c_val,
            }
