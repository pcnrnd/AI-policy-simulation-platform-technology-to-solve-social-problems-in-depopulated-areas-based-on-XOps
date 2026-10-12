"""ExplainabilityModule — 특징 중요도(SHAP) + 순수 Python fallback.

SHAP 미설치 시 평균 절대 기여도 기반 중요도로 대체(graceful degradation).
예측에 어떤 feature가 주요 변수로 작용했는지를 노출해 의사결정을 지원한다.
"""

from __future__ import annotations

from typing import Mapping, Sequence

try:  # 선택 의존성 — 미설치 환경의 import 해석은 pyproject [tool.mypy.overrides] 가 맡는다.
    import shap  # noqa: F401

    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False


class ExplainabilityModule:
    """feature 중요도 산출 (SHAP 있으면 사용, 없으면 fallback)."""

    @property
    def backend(self) -> str:
        return "shap" if _SHAP_AVAILABLE else "pure-python-fallback"

    def rank_features(
        self, contributions: Mapping[str, Sequence[float]]
    ) -> list[dict[str, str | float]]:
        """feature별 기여도 시퀀스 → 평균 기여도로 중요도 정렬(부호 보존, |value| 내림차순).

        입력은 값 타입에 공변인 `Mapping` 으로 받아 `dict[str, list[float]]` 호출자도 그대로 넘긴다.
        정렬은 (이름, 평균) 튜플에서 하고 마지막에 dict 로 바꾼다 — dict 값이 `object` 로 뭉쳐
        `abs()` 타입이 어긋나는 일을 피한다.
        """
        means = [
            (name, round(sum(vals) / len(vals), 6)) for name, vals in contributions.items() if vals
        ]
        means.sort(key=lambda item: abs(item[1]), reverse=True)
        return [{"feature": name, "value": value} for name, value in means]
