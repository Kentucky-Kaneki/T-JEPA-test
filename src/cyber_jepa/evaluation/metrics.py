"""
Predictive Evaluation Metrics for Cyber-JEPA.

Calculates Smooth L1, Cosine Similarity, Latent R^2, Changed vs Unchanged Target Metrics,
Action-Zeroed Degradation, and Action-Shuffled Degradation.
"""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def compute_predictive_metrics(
    pred_latents: torch.Tensor,         # [N, D]
    target_latents: torch.Tensor,       # [N, D]
    persistence_latents: torch.Tensor | None = None, # [N, D] z_t
    changed_mask: torch.Tensor | None = None, # [N] boolean
) -> dict[str, float]:
    """Compute comprehensive predictive metrics over latent predictions."""
    pred_np = pred_latents.detach().cpu().numpy()
    target_np = target_latents.detach().cpu().numpy()

    # 1. Smooth L1 Loss
    smooth_l1 = float(F.smooth_l1_loss(pred_latents, target_latents).item())

    # 2. Cosine Similarity
    cos_sim = float(F.cosine_similarity(pred_latents, target_latents, dim=-1).mean().item())

    # 3. Latent R^2
    target_var = float(np.var(target_np, axis=0).sum())
    mse = float(np.mean((pred_np - target_np) ** 2))
    r2 = float(1.0 - (mse / max(1e-6, target_var)))

    metrics = {
        "smooth_l1": smooth_l1,
        "cosine_similarity": cos_sim,
        "latent_r2": r2,
    }

    # 4. Changed vs Unchanged Targets
    if changed_mask is not None and len(changed_mask) == len(pred_latents):
        changed_mask_np = changed_mask.detach().cpu().numpy().astype(bool)
        if changed_mask_np.any():
            metrics["changed_smooth_l1"] = float(F.smooth_l1_loss(pred_latents[changed_mask_np], target_latents[changed_mask_np]).item())
        else:
            metrics["changed_smooth_l1"] = smooth_l1

        unchanged_mask_np = ~changed_mask_np
        if unchanged_mask_np.any():
            metrics["unchanged_smooth_l1"] = float(F.smooth_l1_loss(pred_latents[unchanged_mask_np], target_latents[unchanged_mask_np]).item())
        else:
            metrics["unchanged_smooth_l1"] = smooth_l1

    # 5. Improvement over Latent Persistence
    if persistence_latents is not None:
        pers_mse = float(np.mean((persistence_latents.detach().cpu().numpy() - target_np) ** 2))
        metrics["persistence_mse"] = pers_mse
        metrics["improvement_over_persistence"] = float(1.0 - (mse / max(1e-6, pers_mse)))

    return metrics


def compute_action_degradation(
    model: Any,
    context: torch.Tensor,
    action_seq: torch.Tensor,
    target_latents: torch.Tensor,
) -> dict[str, float]:
    """Calculate performance degradation under action-zeroed and action-shuffled inference."""
    with torch.no_grad():
        # Baseline loss
        loss_base, _, _ = model(context, action_seq, target_latents)

        # Action zeroed
        zero_actions = torch.zeros_like(action_seq)
        loss_zeroed, _, _ = model(context, zero_actions, target_latents)

        # Action shuffled within batch
        shuffled_indices = torch.randperm(action_seq.shape[0])
        shuffled_actions = action_seq[shuffled_indices]
        loss_shuffled, _, _ = model(context, shuffled_actions, target_latents)

    base_val = loss_base.item()
    zero_val = loss_zeroed.item()
    shuff_val = loss_shuffled.item()

    return {
        "baseline_loss": base_val,
        "action_zeroed_loss": zero_val,
        "action_shuffled_loss": shuff_val,
        "action_zeroed_degradation": float(zero_val - base_val),
        "action_shuffled_degradation": float(shuff_val - base_val),
        "action_sensitive": bool(shuff_val > base_val + 1e-4),
    }
