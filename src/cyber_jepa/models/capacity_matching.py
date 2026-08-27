"""
Capacity Matching for Test 1 (separate ActionEncoder) vs Test 2 (fused) - Phase 4.

Per the confirmed research plan: Test 1's own trained runs (`CyberJEPA` with
`aggregator_mode in {"legacy_last_step_mean","token_preserving_predictor"}`) are NOT
retrained or resized for this study. Test 2's fused encoder is instead grown so its
total trainable parameter count matches Test 1's *combined* action-handling budget:
online encoder + ActionEncoder + predictor body (everything Test 1 spends to go from
observation+action to a prediction). This isolates "where is the action fused" as the
single experimental variable, rather than confounding it with a capacity difference.

`hidden_dim` (D) is deliberately NOT part of the search space: D is the shared
comparison axis diagnostics/probes/selection all key off, and changing it would
confound representation geometry with capacity. Only `num_layers` (preferred - a
depth increase is a more neutral capacity change than widening every layer) and, as a
fallback, `ffn_dim` are searched.
"""

from dataclasses import dataclass
from typing import Callable
import torch.nn as nn

from cyber_jepa.models.jepa import count_subsystem_parameters


@dataclass
class MatchedFusedConfig:
    num_layers: int
    ffn_dim: int
    achieved_params: int
    target_params: int
    search_trace: list[tuple[int, int, int]]   # (num_layers, ffn_dim, params) tried


def compute_test1_action_pathway_budget(test1_model: nn.Module) -> int:
    """
    Total trainable parameters Test 1 spends handling actions + context jointly:
    online encoder + ActionEncoder + predictor body (excludes the aggregator, which
    Test 2 has no equivalent of, and the frozen EMA target encoder, which is not
    trainable in either condition).
    """
    counts = count_subsystem_parameters(test1_model)
    return counts["tokenizer_and_encoder"] + counts["action_encoder"] + counts["predictor_body"]


def search_matched_fused_config(
    build_and_count_fn: Callable[[int, int], int],
    target_params: int,
    layer_grid: list[int] | None = None,
    ffn_grid: list[int] | None = None,
    tolerance: float = 0.01,
) -> MatchedFusedConfig:
    """
    Greedy search for (num_layers, ffn_dim) whose fused-encoder parameter count is the
    closest match to `target_params`, growing depth first (cheaper/more neutral
    capacity increase than widening every layer's FFN) and falling back to width if no
    depth in `layer_grid` gets within `tolerance` of the target.

    `build_and_count_fn(num_layers, ffn_dim) -> int` should construct a throwaway
    instance of the fused encoder wrapped in `CyberJEPAFused` (matching the real one's
    other hyperparameters) and return
    `sum(p.numel() for p in model.parameters() if p.requires_grad)`.
    """
    layer_grid = layer_grid or list(range(1, 13))
    ffn_grid = ffn_grid or [256]

    trace: list[tuple[int, int, int]] = []
    best: tuple[int, int, int] | None = None   # (num_layers, ffn_dim, params)
    best_gap = float("inf")

    for ffn_dim in ffn_grid:
        for num_layers in layer_grid:
            params = build_and_count_fn(num_layers, ffn_dim)
            trace.append((num_layers, ffn_dim, params))
            gap = abs(params - target_params) / max(1, target_params)
            if gap < best_gap:
                best_gap = gap
                best = (num_layers, ffn_dim, params)
            if gap <= tolerance:
                return MatchedFusedConfig(num_layers, ffn_dim, params, target_params, trace)

    assert best is not None, "search grid was empty"
    num_layers, ffn_dim, params = best
    return MatchedFusedConfig(num_layers, ffn_dim, params, target_params, trace)
