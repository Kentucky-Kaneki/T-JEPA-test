"""
Preregistered Representation Selection Rule Implementation.

Applies the step-by-step decision rules defined in Section 13 to select the winning
observation representation—or explicitly conclude that no candidate beats controls.
"""

import json
from pathlib import Path
from typing import Any


PREFERENCE_ORDER = ["host", "feature", "hierarchical", "flat"]


def apply_preregistered_selection_rule(
    run_results: list[dict[str, Any]],
    output_report_path: Path | None = None,
) -> dict[str, Any]:
    """
    Applies Section 13 decision rules over completed evaluation runs.

    Input: list of run result dictionaries containing:
        - run_id, representation, horizon, seed
        - is_collapsed: bool
        - beats_persistence: bool
        - action_sensitive: bool (action shuffling degrades performance)
        - ood_future_compromise_macro_f1: float
        - ood_changed_latent_improvement: float
        - current_compromise_macro_f1: float
        - effective_rank: float
        - inference_latency_ms: float
        - num_trainable_params: int
    """
    survivors: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []

    for run in run_results:
        rep = run.get("representation", "unknown")
        run_id = run.get("run_id", "unknown")

        # Gate 1: Reject collapsed runs
        if run.get("is_collapsed", False):
            rejections.append({"run_id": run_id, "rep": rep, "reason": "Collapsed latent space (std < 0.01 or eff_rank < 10%)"})
            continue

        # Gate 2: Reject models that do not beat latent persistence on changed targets
        if not run.get("beats_persistence", False):
            rejections.append({"run_id": run_id, "rep": rep, "reason": "Failed to beat latent persistence baseline on changed targets"})
            continue

        # Gate 3: Reject action-conditioned models unaffected/improved by action shuffling
        if not run.get("action_sensitive", False):
            rejections.append({"run_id": run_id, "rep": rep, "reason": "Failed action-sensitivity test (unaffected by action shuffling)"})
            continue

        survivors.append(run)

    if not survivors:
        selection_result = {
            "winner_selected": False,
            "selected_representation": None,
            "conclusion": "No candidate representation satisfied all preregistered constraints. Cyber-JEPA is not validated beyond persistence controls in this experiment.",
            "num_survivors": 0,
            "rejections": rejections,
        }
    else:
        # Step 4: Group survivors by representation and rank by mean OOD future-compromise macro-F1 at k=4
        rep_scores: dict[str, list[float]] = {}
        for s in survivors:
            rep = s["representation"]
            f1 = float(s.get("ood_future_compromise_macro_f1", 0.0))
            rep_scores.setdefault(rep, []).append(f1)

        mean_scores = {rep: sum(scores) / len(scores) for rep, scores in rep_scores.items()}

        # Sort candidates by mean score descending, breaking ties by PREFERENCE_ORDER
        sorted_reps = sorted(
            mean_scores.keys(),
            key=lambda r: (mean_scores[r], -PREFERENCE_ORDER.index(r) if r in PREFERENCE_ORDER else -99),
            reverse=True,
        )

        winning_rep = sorted_reps[0]
        selection_result = {
            "winner_selected": True,
            "selected_representation": winning_rep,
            "winning_mean_ood_f1": mean_scores[winning_rep],
            "candidate_rankings": {r: mean_scores[r] for r in sorted_reps},
            "num_survivors": len(survivors),
            "rejections": rejections,
            "conclusion": f"Validated representation '{winning_rep}' selected according to preregistered Section 13 rules.",
        }

    if output_report_path is not None:
        output_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_report_path, "w") as f:
            json.dump(selection_result, f, indent=2)

    return selection_result
