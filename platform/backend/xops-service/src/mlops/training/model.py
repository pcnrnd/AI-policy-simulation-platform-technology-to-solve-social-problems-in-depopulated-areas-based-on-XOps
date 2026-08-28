"""RidgeRegressor — 순수 Python 릿지 회귀(정규방정식 + 부분피벗 가우스 소거).

numpy/scikit-learn을 쓰지 않는다: xops-service 런타임 의존성은
fastapi·uvicorn·pydantic·pydantic-settings뿐이라 Docker 이미지에 없기 때문이다.
metrics.py(6지표 순수 계산)와 같은 방침이며, 난수를 쓰지 않으므로 학습 결과는 결정적이다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

# 표준편차가 0인 열(상수 피처)은 1로 대체해 0으로 나누는 것을 막는다.
_ZERO_STD_REPLACEMENT = 1.0
# 특이행렬 판정 임계 — 피벗 절대값이 이보다 작으면 해가 유일하지 않다고 본다.
_SINGULAR_EPS = 1e-12


def _column_stats(rows: Sequence[Sequence[float]]) -> tuple[list[float], list[float]]:
    """열별 평균과 표준편차(모표준편차). 상수 열의 표준편차는 1로 대체."""
    n_rows = len(rows)
    n_cols = len(rows[0])
    means = [sum(row[j] for row in rows) / n_rows for j in range(n_cols)]
    stds: list[float] = []
    for j in range(n_cols):
        variance = sum((row[j] - means[j]) ** 2 for row in rows) / n_rows
        deviation = variance**0.5
        stds.append(deviation if deviation > _SINGULAR_EPS else _ZERO_STD_REPLACEMENT)
    return means, stds


def _scale(row: Sequence[float], means: Sequence[float], stds: Sequence[float]) -> list[float]:
    """단일 표본을 학습 시 통계로 표준화."""
    return [(value - mean) / std for value, mean, std in zip(row, means, stds)]


def solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """부분피벗 가우스 소거로 Ax=b를 푼다. 특이행렬이면 ValueError."""
    size = len(rhs)
    # 확대행렬을 새로 만들어 입력을 변형하지 않는다.
    table = [list(matrix[i]) + [rhs[i]] for i in range(size)]

    for col in range(size):
        pivot_row = max(range(col, size), key=lambda r: abs(table[r][col]))
        if abs(table[pivot_row][col]) < _SINGULAR_EPS:
            raise ValueError("정규방정식이 특이행렬입니다 — 정규화 계수(lambda)를 높이세요.")
        table[col], table[pivot_row] = table[pivot_row], table[col]
        pivot = table[col][col]
        for row in range(col + 1, size):
            factor = table[row][col] / pivot
            if factor == 0.0:
                continue
            for k in range(col, size + 1):
                table[row][k] -= factor * table[col][k]

    solution = [0.0] * size
    for row in reversed(range(size)):
        total = table[row][size] - sum(table[row][k] * solution[k] for k in range(row + 1, size))
        solution[row] = total / table[row][row]
    return solution


@dataclass
class RidgeRegressor:
    """릿지 회귀 — 표준화된 피처에 대한 계수와 절편을 보유."""

    coefficients: list[float]
    intercept: float
    means: list[float]
    stds: list[float]
    feature_names: list[str]
    ridge_lambda: float

    @classmethod
    def fit(
        cls,
        rows: Sequence[Sequence[float]],
        targets: Sequence[float],
        *,
        ridge_lambda: float,
        feature_names: Sequence[str],
    ) -> RidgeRegressor:
        """(XᵀX + λI)w = Xᵗy 를 풀어 학습. 피처는 표준화, 절편은 타깃 평균."""
        if len(rows) != len(targets) or not rows:
            raise ValueError("rows와 targets는 길이가 같고 비어 있지 않아야 합니다.")
        means, stds = _column_stats(rows)
        scaled = [_scale(row, means, stds) for row in rows]
        target_mean = sum(targets) / len(targets)
        centered = [value - target_mean for value in targets]

        n_cols = len(scaled[0])
        gram = [
            [
                sum(r[i] * r[j] for r in scaled) + (ridge_lambda if i == j else 0.0)
                for j in range(n_cols)
            ]
            for i in range(n_cols)
        ]
        moment = [sum(r[i] * y for r, y in zip(scaled, centered)) for i in range(n_cols)]
        weights = solve_linear_system(gram, moment)
        return cls(
            coefficients=weights,
            intercept=target_mean,
            means=means,
            stds=stds,
            feature_names=list(feature_names),
            ridge_lambda=ridge_lambda,
        )

    def predict_one(self, row: Sequence[float]) -> float:
        """단일 표본 예측."""
        scaled = _scale(row, self.means, self.stds)
        return self.intercept + sum(w * x for w, x in zip(self.coefficients, scaled))

    def predict(self, rows: Sequence[Sequence[float]]) -> list[float]:
        """배치 예측."""
        return [self.predict_one(row) for row in rows]

    def to_dict(self) -> dict[str, Any]:
        """아티팩트 직렬화 (JSON 저장용)."""
        return {
            "kind": "ridge-regression",
            "coefficients": [round(c, 8) for c in self.coefficients],
            "intercept": round(self.intercept, 8),
            "means": [round(m, 8) for m in self.means],
            "stds": [round(s, 8) for s in self.stds],
            "feature_names": list(self.feature_names),
            "ridge_lambda": self.ridge_lambda,
        }


class MeanRegressor:
    """절편-only 기준선 모델 — 학습 타깃 평균을 그대로 예측한다.

    실측 학습 후보를 비교할 대상이 필요하다. 시드 상수(0.892 등)는 같은 평가 프로토콜로
    측정한 값이 아니라 비교 근거가 되지 못하므로, 동일 데이터·동일 LOO 프로토콜로
    측정한 이 기준선을 최초 승급 판정의 현행(current)으로 쓴다.
    """

    def __init__(self, target_mean: float) -> None:
        self._mean = target_mean

    @classmethod
    def fit(cls, targets: Sequence[float]) -> MeanRegressor:
        if not targets:
            raise ValueError("targets가 비어 있습니다.")
        return cls(sum(targets) / len(targets))

    def predict_one(self, row: Sequence[float]) -> float:
        """입력과 무관하게 학습 평균을 반환 — RidgeRegressor와 같은 인터페이스."""
        del row  # 기준선 모델은 피처를 쓰지 않는다.
        return self._mean
