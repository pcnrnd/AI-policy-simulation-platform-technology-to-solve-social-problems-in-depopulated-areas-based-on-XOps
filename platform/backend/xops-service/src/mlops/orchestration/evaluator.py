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


def ranking_key(metrics: Mapping[str, float]) -> tuple[float, ...]:
    """여러 후보를 한 줄로 세우기 위한 사전식 정렬 키(클수록 우수).

    승급 판정(`Evaluator.evaluate`)은 두 모델을 **최우선 공통 지표 하나로만** 비교한다.
    반면 그리드 학습은 후보 N개의 순위를 매겨야 하므로, 같은 우선순위(f1>accuracy>mae>mse)를
    동점 시 다음 지표로 넘어가는 사전식 키로 확장해 쓴다. 오차 지표는 부호를 뒤집어
    "클수록 우수"로 통일한다. 지표가 없으면 최하위로 둔다.
    """
    def signed(metric: str, higher_is_better: bool) -> float:
        if metric not in metrics:
            return float("-inf")
        return metrics[metric] if higher_is_better else -metrics[metric]

    return tuple(signed(metric, higher) for metric, higher in _PRIORITY)


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
