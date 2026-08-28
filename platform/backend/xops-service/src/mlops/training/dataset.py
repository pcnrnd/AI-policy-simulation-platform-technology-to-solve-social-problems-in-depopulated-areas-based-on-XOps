"""학습 데이터셋 구성 — 시드의 지역 인구 시계열에서 지도학습 표본을 만든다.

시드(mock_data.json)에서 실제 시계열인 것은 `regions[].history`(지역별 인구 추이)뿐이므로
이것을 원천으로 쓴다. 값은 지역 규모(85,200 vs 38,400)에 좌우되지 않도록 전부
**상대 변화율**로 환산한다.

- 피처: 관측 창(lag) 안의 기간별 변화율 + 지역 정적 지표(위험지수·출산율·고령화지수)
- 회귀 타깃: h기간 뒤 상대 변화율
- 분류 라벨: 감소폭이 임계(기간당 threshold × h)를 초과했는지

모델별 구분은 **예측 지평 h**로 둔다. h가 커지면 문제가 실제로 어려워지므로 지표가
자연히 낮아진다(시드 상수의 난이도 순서와 같은 방향).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

_STATIC_FEATURES = ("riskIndex", "birthRate", "agingIndex")
_PERCENT = 100.0


@dataclass(frozen=True)
class TrainingDataset:
    """학습 표본 — 모든 하이퍼파라미터 후보가 공유한다.

    lag 후보마다 표본 수가 달라지면 지표를 비교할 수 없으므로, 항상 그리드의 최대 lag로
    창을 잡아 표본 집합을 고정하고 작은 lag는 뒤쪽 변화율 피처만 사용한다(`select`).
    """

    feature_names: list[str]
    rows: list[list[float]]
    targets: list[float]
    labels: list[int]
    horizon: int
    max_lag: int
    static_count: int

    def __len__(self) -> int:
        return len(self.rows)

    def select(self, lag: int) -> tuple[list[str], list[list[float]]]:
        """lag개 관측 창에 해당하는 피처만 남긴 (이름, 행) 반환."""
        if not 2 <= lag <= self.max_lag:
            raise ValueError(f"lag는 2 이상 {self.max_lag} 이하여야 합니다: {lag}")
        lag_columns = len(self.feature_names) - self.static_count
        keep = lag - 1  # 창 안의 기간별 변화율 개수
        start = lag_columns - keep
        names = self.feature_names[start:lag_columns] + self.feature_names[lag_columns:]
        rows = [row[start:lag_columns] + row[lag_columns:] for row in self.rows]
        return names, rows

    def summary(self) -> dict[str, Any]:
        """아티팩트·실행 기록에 남길 데이터셋 요약."""
        return {
            "rows": len(self.rows),
            "features": len(self.feature_names),
            "horizon": self.horizon,
            "max_lag": self.max_lag,
            "positive_labels": sum(self.labels),
            "protocol": "leave-one-out",
        }


def _rates(series: Sequence[float]) -> list[float]:
    """연속 관측치의 기간별 상대 변화율(%)."""
    return [(series[i] / series[i - 1] - 1.0) * _PERCENT for i in range(1, len(series))]


def _numeric(value: Any, *, field: str, where: str, positive: bool = False) -> float:
    """시드 값 하나를 검증해 float으로 변환. 실패 시 무엇이 왜 잘못됐는지 담아 ValueError."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}의 {field}가 숫자가 아닙니다: {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{where}의 {field}가 유한한 수가 아닙니다: {value!r}")
    if positive and number <= 0.0:
        raise ValueError(f"{where}의 {field}는 0보다 커야 합니다(변화율 분모로 쓰임): {value!r}")
    return number


def _validated_region(region: Any, index: int) -> tuple[list[float], list[float]]:
    """지역 하나에서 (인구 시계열, 정적 지표)를 검증해 꺼낸다.

    시드(mock_data.json)는 컨테이너에 읽기 전용으로 마운트되는 **외부 입력**이므로
    여기가 신뢰 경계다. 0·null·누락·비수치·NaN을 이 지점에서 ValueError로 명시 실패시켜
    ZeroDivisionError/TypeError가 상위로 새지 않게 한다(호출자는 ValueError를 잡아
    경고 후 fallback으로 흐른다).
    """
    if not isinstance(region, dict):
        raise ValueError(f"regions[{index}]가 객체가 아닙니다: {type(region).__name__}")
    where = f"regions[{index}]({region.get('id', '?')})"

    raw_history = region.get("history")
    if not isinstance(raw_history, (list, tuple)) or not raw_history:
        raise ValueError(f"{where}의 history가 배열이 아니거나 비어 있습니다: {raw_history!r}")
    history = [
        _numeric(value, field=f"history[{i}]", where=where, positive=True)
        for i, value in enumerate(raw_history)
    ]

    statics: list[float] = []
    for name in _STATIC_FEATURES:
        if name not in region:
            raise ValueError(f"{where}에 필수 정적 지표 {name}이 없습니다.")
        statics.append(_numeric(region[name], field=name, where=where))
    return history, statics


def build_dataset(
    regions: Sequence[dict[str, Any]],
    *,
    horizon: int,
    max_lag: int,
    decline_threshold_pct: float,
) -> TrainingDataset:
    """지역 시계열 → (피처, 회귀 타깃, 분류 라벨). 표본 순서는 결정적이다."""
    if horizon < 1 or max_lag < 2:
        raise ValueError("horizon은 1 이상, max_lag는 2 이상이어야 합니다.")
    if not isinstance(regions, (list, tuple)):
        raise ValueError(f"regions가 배열이 아닙니다: {type(regions).__name__}")

    lag_names = [f"rate_t-{max_lag - 1 - i}" for i in range(max_lag - 1)]
    feature_names = lag_names + list(_STATIC_FEATURES)
    rows: list[list[float]] = []
    targets: list[float] = []
    labels: list[int] = []
    # 지평이 길어질수록 누적 감소폭이 커지므로 임계도 지평에 비례시킨다.
    threshold = decline_threshold_pct * horizon

    for index, region in enumerate(regions):
        history, statics = _validated_region(region, index)
        for anchor in range(max_lag - 1, len(history) - horizon):
            window = history[anchor - max_lag + 1 : anchor + 1]
            change = (history[anchor + horizon] / history[anchor] - 1.0) * _PERCENT
            rows.append(_rates(window) + statics)
            targets.append(change)
            labels.append(1 if change < -threshold else 0)

    if not rows:
        raise ValueError("학습 표본을 만들 수 없습니다 — 시드 시계열이 창 길이보다 짧습니다.")
    return TrainingDataset(
        feature_names=feature_names,
        rows=rows,
        targets=targets,
        labels=labels,
        horizon=horizon,
        max_lag=max_lag,
        static_count=len(_STATIC_FEATURES),
    )
