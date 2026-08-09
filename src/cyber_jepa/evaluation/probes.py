"""
Frozen Linear Probing Diagnostic Pipeline for Cyber-JEPA.

Extracts frozen latents z_t without updating encoder parameters and fits regularized
logistic regression probes for host compromise, attacker presence, future compromise at t+4,
and critical operational-server risk.
"""

from typing import Any
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix


class LinearProbeEvaluator:
    """Frozen linear probing evaluator for latent representation quality."""

    @torch.no_grad()
    def extract_latents_and_labels(
        self,
        encoder: nn.Module,
        data_loader: Any,
        device: torch.device,
        oracle_key: str = "host_compromise",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract frozen latents z_t and corresponding target labels."""
        encoder.eval()
        encoder.to(device)

        latents_list: list[np.ndarray] = []
        labels_list: list[Any] = []

        for batch in data_loader:
            hist = batch["history_flat"].to(device)
            z = encoder(hist)
            if z.dim() == 4:
                z = z[:, -1, :, :].mean(dim=1)
            elif z.dim() == 3:
                z = z[:, -1, :]

            latents_list.append(z.cpu().numpy())

            # Labels from batch or synthetic default for test harness
            if "labels" in batch:
                labels_list.extend(batch["labels"])
            else:
                labels_list.extend([0] * len(hist))

        latents = np.concatenate(latents_list, axis=0)
        labels = np.array(labels_list)
        return latents, labels

    def train_and_evaluate_probe(
        self,
        train_latents: np.ndarray,
        train_labels: np.ndarray,
        test_latents: np.ndarray,
        test_labels: np.ndarray,
        c_val: float = 1.0,
    ) -> dict[str, Any]:
        """Fit regularized logistic regression probe on train split and evaluate on test split."""
        if len(np.unique(train_labels)) < 2:
            return {
                "macro_f1": 0.0,
                "balanced_accuracy": 0.0,
                "auroc": 0.0,
                "note": "Insufficient classes in training split (< 2)",
            }

        clf = LogisticRegression(C=c_val, max_iter=500, random_state=42)
        clf.fit(train_latents, train_labels)

        preds = clf.predict(test_latents)
        macro_f1 = float(f1_score(test_labels, preds, average="macro"))
        bal_acc = float(balanced_accuracy_score(test_labels, preds))

        probs = None
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
            "balanced_accuracy": bal_acc,
            "auroc": auroc,
            "confusion_matrix": cm,
            "c_val": c_val,
        }
