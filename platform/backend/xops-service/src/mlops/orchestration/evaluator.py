"""Evaluator — 승급 판정. primary 지표 우선순위 f1 > accuracy > mae > mse.

f1·accuracy는 높을수록, mae·mse는 낮을수록 우수. 두 모델에 공통으로 존재하는
최우선 지표로 후보(candidate)가 현행(current)보다 나으면 승급.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

# 우선순위와 방향(True=높을수록 우수)
_PRIORITY: tuple[tuple[str, bool], ...] = (
    ("f1", True),
    ("accuracy", True),
    ("mae", False),
    ("mse", False),
)


@dataclass(frozen=True)
class EvaluationResult:
    """승급 판정 결과."""

    promote: bool
    primary_metric: str | None
    current_value: float | None
    candidate_value: float | None
    reason: str


class Evaluator:
    """primary 지표 기준으로 후보 승급 여부를 판정."""

    def evaluate(self, current: Mapping[str, float], candidate: Mapping[str, float]) -> EvaluationResult:
        for metric, higher_is_better in _PRIORITY:
            if metric in current and metric in candidate:
                cur, cand = current[metric], candidate[metric]
                better = cand > cur if higher_is_better else cand < cur
                arrow = "↑" if higher_is_better else "↓"
                return EvaluationResult(
                    promote=better,
                    primary_metric=metric,
                    current_value=cur,
                    candidate_value=cand,
                    reason=f"primary={metric}({arrow}) current={cur} candidate={cand}",
                )
        return EvaluationResult(
            promote=False,
            primary_metric=None,
            current_value=None,
            candidate_value=None,
            reason="공통 primary 지표가 없어 승급을 보류합니다.",
        )
