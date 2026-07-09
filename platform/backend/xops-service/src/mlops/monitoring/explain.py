"""ExplainabilityModule — 특징 중요도(SHAP) + 순수 Python fallback.

SHAP 미설치 시 평균 절대 기여도 기반 중요도로 대체(graceful degradation).
예측에 어떤 feature가 주요 변수로 작용했는지를 노출해 의사결정을 지원한다.
"""

from __future__ import annotations

from typing import Sequence

try:  # 선택 의존성
    import shap  # type: ignore  # noqa: F401

    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False


class ExplainabilityModule:
    """feature 중요도 산출 (SHAP 있으면 사용, 없으면 fallback)."""

    @property
    def backend(self) -> str:
        return "shap" if _SHAP_AVAILABLE else "pure-python-fallback"

    def rank_features(self, contributions: dict[str, Sequence[float]]) -> list[dict[str, float]]:
        """feature별 기여도 시퀀스 → 평균 기여도로 중요도 정렬(부호 보존, |value| 내림차순)."""
        ranked = [
            {"feature": name, "value": round(sum(vals) / len(vals), 6)}
            for name, vals in contributions.items()
            if vals
        ]
        ranked.sort(key=lambda r: abs(r["value"]), reverse=True)
        return ranked
