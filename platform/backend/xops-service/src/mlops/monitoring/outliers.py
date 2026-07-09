"""OutlierDetector — Z-score / IQR 이상치 탐지 (순수 Python)."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Sequence

from src.core.settings import get_settings


@dataclass(frozen=True)
class OutlierResult:
    """이상치 탐지 결과 — 인덱스·값·경계."""

    method: str
    outliers: list[dict[str, float]] = field(default_factory=list)
    lower: float | None = None
    upper: float | None = None


class OutlierDetector:
    """Z-score(기본) 및 IQR 방식 이상치 탐지."""

    def zscore(self, values: Sequence[float]) -> OutlierResult:
        if len(values) < 2:
            raise ValueError("z-score 계산에는 2개 이상의 값이 필요합니다.")
        threshold = get_settings().zscore_threshold
        mean = statistics.fmean(values)
        stdev = statistics.pstdev(values)
        if stdev == 0:
            return OutlierResult(method="zscore", lower=mean, upper=mean)
        found = [
            {"index": i, "value": v, "z": round((v - mean) / stdev, 4)}
            for i, v in enumerate(values)
            if abs((v - mean) / stdev) > threshold
        ]
        return OutlierResult(
            method="zscore", outliers=found, lower=mean - threshold * stdev, upper=mean + threshold * stdev
        )

    def iqr(self, values: Sequence[float]) -> OutlierResult:
        if len(values) < 4:
            raise ValueError("IQR 계산에는 4개 이상의 값이 필요합니다.")
        mult = get_settings().iqr_multiplier
        q1, _, q3 = statistics.quantiles(values, n=4)
        iqr = q3 - q1
        lower, upper = q1 - mult * iqr, q3 + mult * iqr
        found = [{"index": i, "value": v} for i, v in enumerate(values) if v < lower or v > upper]
        return OutlierResult(method="iqr", outliers=found, lower=lower, upper=upper)
