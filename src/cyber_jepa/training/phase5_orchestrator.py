"""Pure planning and validation-only selection helpers for Phase 5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Phase5Candidate:
    name: str
    static_to_dynamic: tuple[int, int] | None
    action_weight: float = 0.0


class Phase5Orchestrator:
    """Keeps stage gates explicit and prevents test/OOD-driven model selection."""

    @staticmethod
    def stage1_candidates(ratios: dict[str, list[int] | None]) -> list[Phase5Candidate]:
        return [Phase5Candidate(name, None if ratio is None else (int(ratio[0]), int(ratio[1]))) for name, ratio in ratios.items()]

    @staticmethod
    def stage2_candidates(ratio_name: str, ratio: list[int], action_weights: list[float]) -> list[Phase5Candidate]:
        return [Phase5Candidate(f"{ratio_name}_action{weight:g}", (int(ratio[0]), int(ratio[1])), float(weight)) for weight in action_weights]

    @staticmethod
    def rank_by_validation(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Rank only validation outputs; test and policy-transfer fields are ignored."""
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in run_rows:
            groups.setdefault(row["run_id"].rsplit("_seed", 1)[0], []).append(row)
        rankings = []
        for name, rows in groups.items():
            metrics = [row["validation"] for row in rows]
            score = sum(
                metric["prediction"].get("improvement_over_persistence", -1.0)
                + metric["geometry"]["effective_rank_fraction"]
                - metric["prediction"].get("changed_smooth_l1", 0.0)
                + row["validation_probe"]["macro_f1"]
                for row, metric in zip(rows, metrics)
            ) / len(rows)
            rankings.append({
                "candidate": name,
                "validation_score": score,
                "mean_validation_f1": sum(row["validation_probe"]["macro_f1"] for row in rows) / len(rows),
                "mean_rank_fraction": sum(metric["geometry"]["effective_rank_fraction"] for metric in metrics) / len(metrics),
                "mean_persistence_improvement": sum(metric["prediction"].get("improvement_over_persistence", 0.0) for metric in metrics) / len(metrics),
            })
        return sorted(rankings, key=lambda row: row["validation_score"], reverse=True)
