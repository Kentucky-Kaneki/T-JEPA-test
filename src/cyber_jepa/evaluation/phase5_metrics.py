"""Phase 5 representation, persistence, dynamics, and action diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from cyber_jepa.evaluation.metrics import compute_action_degradation, compute_predictive_metrics


def geometry_with_principal_components(latents: torch.Tensor) -> dict[str, Any]:
    values = latents.detach().float().cpu().numpy()
    centered = values - values.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, len(values) - 1)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    total = float(eigenvalues.sum())
    fractions = eigenvalues / total if total > 1e-12 else np.zeros_like(eigenvalues)
    nonzero = fractions[fractions > 1e-12]
    effective_rank = float(np.exp(-(nonzero * np.log(nonzero)).sum())) if len(nonzero) else 1.0
    std = values.std(axis=0)
    return {
        "effective_rank": effective_rank,
        "effective_rank_fraction": effective_rank / values.shape[1],
        "pc1_variance_explained": float(fractions[0]) if len(fractions) else 0.0,
        "pc5_cumulative_variance_explained": float(fractions[:5].sum()),
        "median_dim_std": float(np.median(std)),
        "mean_dim_std": float(np.mean(std)),
        "near_constant_fraction": float(np.mean(std < 0.01)),
        "off_diagonal_covariance_mse": float(((covariance - np.diag(np.diag(covariance))) ** 2).mean()),
        "covariance_spectrum_top5": eigenvalues[:5].tolist(),
        "is_collapsed": bool(np.median(std) < 0.01 or effective_rank / values.shape[1] < 0.10),
    }


@torch.no_grad()
def evaluate_phase5_model(model: Any, loader: Any, device: torch.device, threshold: float) -> dict[str, Any]:
    model.eval()
    predictions, targets, contexts, dynamics = [], [], [], []
    batch_for_action = None
    for batch in loader:
        history = batch["history_flat"].to(device)
        actions = batch["action_seq"].to(device)
        target_obs = batch["target_flat"].to(device)
        _, prediction, target = model(history, actions, target_obs)
        context = model.online_encoder(history)
        context = context.global_token if hasattr(context, "global_token") else context
        predictions.append(prediction.cpu())
        targets.append(target.cpu())
        contexts.append(context.cpu())
        dynamics.append(torch.sqrt(torch.mean((history[:, -1] - target_obs).square(), dim=1)).cpu() > threshold)
        if batch_for_action is None:
            batch_for_action = (history, actions, target_obs)
    pred = torch.cat(predictions)
    target = torch.cat(targets)
    context = torch.cat(contexts)
    dynamic = torch.cat(dynamics)
    predictive = compute_predictive_metrics(pred, target, persistence_latents=context, changed_mask=dynamic)
    predictive["static_smooth_l1"] = float(F.smooth_l1_loss(pred[~dynamic], target[~dynamic]).item()) if (~dynamic).any() else predictive["smooth_l1"]
    predictive["dynamic_fraction"] = float(dynamic.float().mean())
    action = compute_action_degradation(model, *batch_for_action) if batch_for_action is not None else {}
    return {"prediction": predictive, "geometry": geometry_with_principal_components(context), "action": action}
