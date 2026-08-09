"""
Unit tests for predictive metrics, frozen linear probes, latent geometry diagnostics, and selection decision tree.
"""

from pathlib import Path
import numpy as np
import torch
import pytest

from cyber_jepa.evaluation.metrics import compute_predictive_metrics, compute_action_degradation
from cyber_jepa.evaluation.diagnostics import compute_latent_geometry_diagnostics
from cyber_jepa.evaluation.selection import apply_preregistered_selection_rule


def test_predictive_metrics():
    """Verify compute_predictive_metrics calculations."""
    pred = torch.randn(10, 64)
    target = torch.randn(10, 64)
    pers = torch.randn(10, 64)
    changed_mask = torch.tensor([True, False] * 5, dtype=torch.bool)

    metrics = compute_predictive_metrics(pred, target, pers, changed_mask)

    assert "smooth_l1" in metrics
    assert "cosine_similarity" in metrics
    assert "latent_r2" in metrics
    assert "changed_smooth_l1" in metrics
    assert "unchanged_smooth_l1" in metrics
    assert "improvement_over_persistence" in metrics


def test_latent_geometry_diagnostics():
    """Verify compute_latent_geometry_diagnostics and collapse flag."""
    # Normal latents
    latents = torch.randn(100, 64)
    diag = compute_latent_geometry_diagnostics(latents)

    assert diag["median_std"] > 0.1
    assert diag["effective_rank"] > 5.0
    assert not diag["is_collapsed"]

    # Collapsed latents (near zero std)
    latents_collapsed = torch.zeros(100, 64) + 0.001 * torch.randn(100, 64)
    diag_col = compute_latent_geometry_diagnostics(latents_collapsed)
    assert diag_col["is_collapsed"]


def test_preregistered_selection_rule():
    """Verify Section 13 decision tree selection logic."""
    runs = [
        # Collapsed run
        {
            "run_id": "r1",
            "representation": "flat",
            "is_collapsed": True,
            "beats_persistence": True,
            "action_sensitive": True,
        },
        # Non-action-sensitive run
        {
            "run_id": "r2",
            "representation": "feature",
            "is_collapsed": False,
            "beats_persistence": True,
            "action_sensitive": False,
        },
        # Valid host run
        {
            "run_id": "r3",
            "representation": "host",
            "is_collapsed": False,
            "beats_persistence": True,
            "action_sensitive": True,
            "ood_future_compromise_macro_f1": 0.82,
        },
        # Valid hierarchical run
        {
            "run_id": "r4",
            "representation": "hierarchical",
            "is_collapsed": False,
            "beats_persistence": True,
            "action_sensitive": True,
            "ood_future_compromise_macro_f1": 0.79,
        },
    ]

    res = apply_preregistered_selection_rule(runs)

    assert res["winner_selected"] is True
    assert res["selected_representation"] == "host"
    assert res["num_survivors"] == 2
    assert len(res["rejections"]) == 2


def test_linear_probe_evaluator():
    """Verify LinearProbeEvaluator training and evaluation."""
    from cyber_jepa.evaluation.probes import LinearProbeEvaluator

    evaluator = LinearProbeEvaluator()

    train_z = np.random.randn(20, 64)
    train_y = np.array([0, 1] * 10)
    test_z = np.random.randn(10, 64)
    test_y = np.array([0, 1] * 5)

    metrics = evaluator.train_and_evaluate_probe(train_z, train_y, test_z, test_y, c_val=1.0)

    assert "macro_f1" in metrics
    assert "balanced_accuracy" in metrics
    assert "auroc" in metrics
    assert "confusion_matrix" in metrics


def test_world_model_interface_dataclasses():
    """Verify TargetSpec, LatentContext, LatentPrediction dataclasses."""
    from cyber_jepa.models.interface import TargetSpec, LatentContext, LatentPrediction

    target = TargetSpec(granularity="host", target_id="Defender", query_indices=[0])
    ctx = LatentContext(context_tokens=torch.randn(2, 4, 64), history_len=4, metadata={})
    pred = LatentPrediction(predicted_latent=torch.randn(2, 64), validity_mask=torch.ones(2, 4), target=target, horizon=4)

    assert target.granularity == "host"
    assert ctx.history_len == 4
    assert pred.horizon == 4
