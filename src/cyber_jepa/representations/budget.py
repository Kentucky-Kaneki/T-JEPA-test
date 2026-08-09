"""
Parameter budget utility for Cyber-JEPA candidate representations.

Selects feed-forward network (FFN) widths from candidate widths [128, 192, 256, 320, 384]
to keep total trainable parameter counts within 10% across all four representation models.
"""

from typing import Any
import torch
import torch.nn as nn


CANDIDATE_FFN_WIDTHS = [128, 192, 256, 320, 384]


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Count trainable and total parameters in a PyTorch module."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def select_ffn_width_for_budget(
    model_factory: Any,
    target_param_count: int,
    candidate_widths: list[int] = CANDIDATE_FFN_WIDTHS,
) -> int:
    """Select the candidate FFN width that brings model parameter count closest to target."""
    best_width = candidate_widths[0]
    best_diff = float("inf")

    for width in candidate_widths:
        model = model_factory(ffn_dim=width)
        trainable, _ = count_parameters(model)
        diff = abs(trainable - target_param_count)
        if diff < best_diff:
            best_diff = diff
            best_width = width

    return best_width


def check_parameter_budget_alignment(models: dict[str, nn.Module], max_tolerance: float = 0.10) -> bool:
    """Assert that all model parameter counts fall within max_tolerance (10%) of mean."""
    counts = {name: count_parameters(m)[0] for name, m in models.items()}
    if not counts:
        return True

    mean_count = sum(counts.values()) / len(counts)
    for name, cnt in counts.items():
        rel_diff = abs(cnt - mean_count) / mean_count
        if rel_diff > max_tolerance:
            raise ValueError(
                f"Model '{name}' parameter count {cnt:,} exceeds {max_tolerance:.1%} tolerance "
                f"relative to mean {mean_count:,.0f} (relative diff = {rel_diff:.2%})"
            )
    return True
