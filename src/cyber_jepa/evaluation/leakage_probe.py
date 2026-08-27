"""
Action-Leakage Probe (Phase 4 addition #1).

`fusion_masking.py`'s strict mode exists to preserve a "neutral C_t" - the state
tokens' / global token's output should not be able to tell what action is about to
happen. Nothing in the delivered pipeline actually tests that claim:
`probes.LinearProbeEvaluator.extract_latents_and_labels` always calls the fused
encoder with `actions=None` (see `probes.py`'s `z = encoder(hist)`), so it never
builds action tokens and never exercises `build_fusion_attention_mask` at all -
strict and permissive are indistinguishable to it by construction.

This module extracts the encoder's context representation from an
actions-INCLUDED pass and asks a linear probe whether it can recover the action
that was taken. Expected result if the architecture behaves as designed: strict
mode -> near-chance accuracy, permissive mode -> well above chance. Reuses
`probes.LinearProbeEvaluator.train_and_evaluate_probe` unmodified; only the
extraction step differs from `probes.py`.

Probes at the discrete action-index granularity (0..65) since that's the only
action-identity field `data/dataset.py` currently carries per window
(`action_seq`). If you want host-target-specific leakage (e.g. "can permissive
mode tell WHICH host is about to be scanned, even if it can't tell the action
type"), extend `CyberJEPADataset.__getitem__` to also carry
`action_type_id` / `host_target_id` per sample and swap the label source below.
"""

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from cyber_jepa.evaluation.probes import LinearProbeEvaluator


@torch.no_grad()
def extract_context_and_action_labels(
    fused_encoder: nn.Module,     # FlatFusedRepresentation / FeatureFusedRepresentation
    data_loader: Any,
    device: torch.device,
    action_position: int = 0,     # which step of the K-length action_seq to probe for
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the fused encoder's global/context token WITH actions attached, plus
    the discrete action index at `action_position` as the leakage label. Distinct
    from `probes.LinearProbeEvaluator.extract_latents_and_labels`, which always
    passes `actions=None` and therefore cannot see any mask effect."""
    fused_encoder.eval()
    fused_encoder.to(device)

    context_list: list[np.ndarray] = []
    label_list: list[int] = []

    for batch in data_loader:
        hist = batch["history_flat"].to(device)
        actions = batch["action_seq"].to(device)

        ctx = fused_encoder.encode_context(hist, actions=actions, return_context_tokens=True)
        global_token = ctx.global_token   # [B, D] - the state-side "neutral" token

        context_list.append(global_token.cpu().numpy())
        label_list.extend(actions[:, action_position].cpu().numpy().tolist())

    return np.concatenate(context_list, axis=0), np.array(label_list)


def run_leakage_probe(
    fused_encoder: nn.Module,
    train_loader: Any,
    test_loader: Any,
    device: torch.device,
    action_position: int = 0,
) -> dict[str, Any]:
    """Fit + evaluate the leakage probe, reusing `LinearProbeEvaluator`'s classifier
    machinery. Returns the same dict shape as `train_and_evaluate_probe`
    (classification branch) plus a `chance_accuracy` reference so the result is
    interpretable without a second lookup."""
    train_ctx, train_labels = extract_context_and_action_labels(fused_encoder, train_loader, device, action_position)
    test_ctx, test_labels = extract_context_and_action_labels(fused_encoder, test_loader, device, action_position)

    evaluator = LinearProbeEvaluator()
    result = evaluator.train_and_evaluate_probe(
        train_ctx, train_labels, test_ctx, test_labels, is_classification=True,
    )

    _, counts = np.unique(test_labels, return_counts=True)
    result["chance_accuracy"] = float(counts.max() / counts.sum()) if len(counts) else 0.0
    result["action_position_probed"] = action_position
    return result
