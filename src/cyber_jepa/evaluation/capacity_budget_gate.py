"""
Capacity-Budget Regression Gate (Phase 4 addition #3).

`capacity_matching.search_matched_fused_config` returns whichever (num_layers,
ffn_dim) got CLOSEST to the target budget even if it never landed within
`tolerance` - see its `assert best is not None` fallback path, which is reached
whenever nothing in `layer_grid`/`ffn_grid` hits the gap threshold. That means a
silent capacity drift (e.g. from a future hyperparameter edit, or from
`action_semantics.py`/`predictor.py` changing) is possible without anything
raising. This mirrors the intent of Gate 4 (`test_parameter_budget_alignment`,
referenced in `orchestrator.verify_all_nine_gates`) but scoped specifically to
the 4 Phase 4 variants, and checked against the exact numbers
`PHASE4_README.md` already documents as verified:

    flat_fused_{strict,permissive}:    549,056 / 549,440 target -> 0.07% gap
    feature_fused_{strict,permissive}: 522,240 / 523,520 target -> 0.24% gap

Call `check_all_variants_capacity_budget(variants)` right after
`build_all_phase4_variants(...)` and before spending any training compute.
"""

from typing import Any

from cyber_jepa.evaluation.phase4_variants import Phase4Variant

DEFAULT_TOLERANCE = 0.01   # 1%, matches capacity_matching.py's own default tolerance


def check_variant_capacity_budget(variant: Phase4Variant, tolerance: float = DEFAULT_TOLERANCE) -> dict[str, Any]:
    """Raises AssertionError if `variant`'s achieved param count exceeds `tolerance`
    of its target; otherwise returns a small report dict for logging."""
    mc = variant.matched_config
    gap = abs(mc.achieved_params - mc.target_params) / max(1, mc.target_params)
    passed = gap <= tolerance

    report = {
        "variant": variant.name,
        "target_params": mc.target_params,
        "achieved_params": mc.achieved_params,
        "gap_fraction": gap,
        "tolerance": tolerance,
        "num_layers": mc.num_layers,
        "ffn_dim": mc.ffn_dim,
        "passed": passed,
    }
    assert passed, (
        f"Capacity budget gate FAILED for '{variant.name}': achieved {mc.achieved_params} "
        f"params vs target {mc.target_params} params ({gap:.2%} gap > {tolerance:.2%} tolerance). "
        f"Search trace tail: {mc.search_trace[-3:]}"
    )
    return report


def check_all_variants_capacity_budget(
    variants: list[Phase4Variant], tolerance: float = DEFAULT_TOLERANCE,
) -> list[dict[str, Any]]:
    """Runs `check_variant_capacity_budget` over all 4 variants; fails fast on the
    first violation (raises), printing every prior variant's report along the way."""
    reports = []
    for v in variants:
        report = check_variant_capacity_budget(v, tolerance)
        print(f"[capacity gate] {v.name}: {report['achieved_params']:,} / {report['target_params']:,} "
              f"params ({report['gap_fraction']:.2%} gap) - {'PASS' if report['passed'] else 'FAIL'}")
        reports.append(report)
    return reports
