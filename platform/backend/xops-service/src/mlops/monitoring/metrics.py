"""MetricCollector — 6대 지표(MSE·MAE·F1·Accuracy·Recall·Precision) 실제 계산.

회귀(MSE·MAE)는 연속값 y_true/y_pred, 분류(Accuracy·Precision·Recall·F1)는
라벨 y_true/y_pred로부터 계산. 순수 Python(외부 의존 없음).
"""

from __future__ import annotations

from typing import Sequence

REGRESSION_METRICS = ("mse", "mae")
CLASSIFICATION_METRICS = ("accuracy", "precision", "recall", "f1")


def _require_same_length(a: Sequence[object], b: Sequence[object]) -> None:
    if len(a) != len(b) or not a:
        raise ValueError("y_true와 y_pred는 길이가 같고 비어 있지 않아야 합니다.")


def mse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    _require_same_length(y_true, y_pred)
    return sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    _require_same_length(y_true, y_pred)
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)


def _binary_counts(y_true: Sequence[int], y_pred: Sequence[int], positive: int = 1) -> tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for t, p in zip(y_true, y_pred):
        if p == positive and t == positive:
            tp += 1
        elif p == positive and t != positive:
            fp += 1
        elif p != positive and t == positive:
            fn += 1
        else:
            tn += 1
    return tp, fp, tn, fn


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


class MetricCollector:
    """예측 결과로부터 6대 지표를 계산."""

    def regression(self, y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
        return {"mse": round(mse(y_true, y_pred), 6), "mae": round(mae(y_true, y_pred), 6)}

    def classification(self, y_true: Sequence[int], y_pred: Sequence[int], positive: int = 1) -> dict[str, float]:
        _require_same_length(y_true, y_pred)
        tp, fp, tn, fn = _binary_counts(y_true, y_pred, positive)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        accuracy = _safe_div(tp + tn, tp + fp + tn + fn)
        return {
            "accuracy": round(accuracy, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
