"""Trainer — 하이퍼파라미터 그리드를 실제 학습해 SOTA 후보를 도출한다.

흐름: 데이터셋 구성 → 후보별 LOO 교차검증 예측 → `MetricCollector`로 6지표 실측 →
승급 우선순위(f1>accuracy>mae>mse)로 최고 후보 선택 → 추론 지연 측정 → 아티팩트 기록.

표본이 적어(시드 기준 10~14) 단일 홀드아웃은 지표가 거칠게 양자화되므로 LOO를 쓴다.
후보 수만큼 (5×5) 선형계를 푸는 비용이라 인프로세스에서 즉시 끝난다.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from src.core.logger import get_logger
from src.core.seed import get_seed
from src.core.settings import get_settings
from src.mlops.monitoring.metrics import MetricCollector
from src.mlops.orchestration.evaluator import ranking_key
from src.mlops.training.dataset import TrainingDataset, build_dataset
from src.mlops.training.model import MeanRegressor, RidgeRegressor

_logger = get_logger("xops.training")
_metrics = MetricCollector()
# 지연 측정 반복 횟수 — 단발 측정은 타이머 분해능에 묻힌다.
_LATENCY_REPEATS = 20


class _Predictor(Protocol):
    """LOO 평가에 필요한 최소 인터페이스."""

    def predict_one(self, row: Sequence[float]) -> float: ...


# 학습 폴드 → 예측기. 후보(릿지)와 기준선(절편-only)이 같은 LOO 루프를 공유한다.
_FitFn = Callable[[Sequence[Sequence[float]], Sequence[float]], _Predictor]


@dataclass(frozen=True)
class CandidateResult:
    """후보 하나의 하이퍼파라미터와 실측 지표."""

    lag: int
    ridge_lambda: float
    metrics: dict[str, float]

    def hyperparams(self) -> dict[str, float]:
        return {"lag": float(self.lag), "ridge_lambda": self.ridge_lambda}

    def to_dict(self) -> dict[str, Any]:
        return {"lag": self.lag, "ridge_lambda": self.ridge_lambda, "metrics": self.metrics}


@dataclass(frozen=True)
class TrainingResult:
    """학습 1회의 산출물 — 오케스트레이터가 후보 지표·기준선·아티팩트를 여기서 읽는다."""

    model_id: str
    horizon: int
    baseline_metrics: dict[str, float]
    candidate_metrics: dict[str, float]
    best: dict[str, float]
    candidates: list[dict[str, Any]] = field(default_factory=list)
    dataset: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    model: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """PipelineRun에 실을 학습 요약(모델 가중치는 제외)."""
        return {
            "horizon": self.horizon,
            "best": self.best,
            "candidates_evaluated": len(self.candidates),
            "dataset": self.dataset,
            "baseline_metrics": self.baseline_metrics,
            "latency_ms": self.latency_ms,
        }


def _label_of(prediction: float, threshold: float) -> int:
    """예측 변화율을 분류 라벨로 — 감소폭이 임계를 넘으면 1."""
    return 1 if prediction < -threshold else 0


def _score(
    predictions: Sequence[float], dataset: TrainingDataset, threshold: float
) -> dict[str, float]:
    """실측 예측값에서 6지표 전부를 계산(기존 MetricCollector 재사용)."""
    predicted_labels = [_label_of(value, threshold) for value in predictions]
    return {
        **_metrics.regression(dataset.targets, predictions),
        **_metrics.classification(dataset.labels, predicted_labels),
    }


def _loo_predictions(
    rows: Sequence[Sequence[float]],
    targets: Sequence[float],
    *,
    fit: _FitFn,
) -> list[float]:
    """Leave-one-out 예측 — 각 표본을 나머지로 학습한 모델로 예측한다."""
    predictions: list[float] = []
    for held_out in range(len(rows)):
        train_rows = [row for i, row in enumerate(rows) if i != held_out]
        train_targets = [value for i, value in enumerate(targets) if i != held_out]
        model = fit(train_rows, train_targets)
        predictions.append(model.predict_one(rows[held_out]))
    return predictions


class Trainer:
    """인프로세스 재학습 — 시드 시계열로 후보 모델을 실제 학습한다."""

    def train(self, model_id: str, *, horizon: int, version: str) -> TrainingResult:
        """그리드 전체를 학습해 최고 후보와 기준선을 함께 반환하고 아티팩트를 남긴다."""
        settings = get_settings()
        lags = sorted(settings.train_lag_windows)
        threshold = settings.train_decline_threshold_pct * horizon
        dataset = build_dataset(
            get_seed()["regions"],
            horizon=horizon,
            max_lag=lags[-1],
            decline_threshold_pct=settings.train_decline_threshold_pct,
        )

        baseline = _score(
            _loo_predictions(dataset.rows, dataset.targets, fit=lambda _r, t: MeanRegressor.fit(t)),
            dataset,
            threshold,
        )
        candidates = self._evaluate_grid(dataset, lags, settings.train_ridge_lambdas, threshold)
        best = max(candidates, key=lambda c: ranking_key(c.metrics))

        names, rows = dataset.select(best.lag)
        final = RidgeRegressor.fit(
            rows, dataset.targets, ridge_lambda=best.ridge_lambda, feature_names=names
        )
        result = TrainingResult(
            model_id=model_id,
            horizon=horizon,
            baseline_metrics=baseline,
            candidate_metrics=best.metrics,
            best=best.hyperparams(),
            candidates=[c.to_dict() for c in candidates],
            dataset=dataset.summary(),
            latency_ms=_measure_latency(final, rows),
            model=final.to_dict(),
        )
        _write_artifact(result, version)
        _logger.info(
            f"train model={model_id} h={horizon} rows={len(dataset)} candidates={len(candidates)} "
            f"best_lag={best.lag} best_lambda={best.ridge_lambda} f1={best.metrics.get('f1')}"
        )
        return result

    def _evaluate_grid(
        self,
        dataset: TrainingDataset,
        lags: Sequence[int],
        lambdas: Sequence[float],
        threshold: float,
    ) -> list[CandidateResult]:
        """lag × lambda 조합을 모두 학습. 표본 집합은 고정이라 지표가 서로 비교 가능하다."""
        return [
            self._evaluate_candidate(dataset, lag=lag, ridge_lambda=lam, threshold=threshold)
            for lag in lags
            for lam in lambdas
        ]

    @staticmethod
    def _evaluate_candidate(
        dataset: TrainingDataset,
        *,
        lag: int,
        ridge_lambda: float,
        threshold: float,
    ) -> CandidateResult:
        """후보 하나를 LOO로 평가. 하이퍼파라미터는 함수 인자로 받아 루프 변수 캡처를 피한다."""
        names, rows = dataset.select(lag)

        def fit(
            train_rows: Sequence[Sequence[float]], train_targets: Sequence[float]
        ) -> RidgeRegressor:
            return RidgeRegressor.fit(
                train_rows, train_targets, ridge_lambda=ridge_lambda, feature_names=names
            )

        predictions = _loo_predictions(rows, dataset.targets, fit=fit)
        return CandidateResult(
            lag=lag,
            ridge_lambda=ridge_lambda,
            metrics=_score(predictions, dataset, threshold),
        )


def _measure_latency(model: RidgeRegressor, rows: Sequence[Sequence[float]]) -> float:
    """1건 추론의 평균 지연(ms) — 카나리 헬스체크에 쓰는 실측치."""
    start = time.perf_counter()
    for _ in range(_LATENCY_REPEATS):
        model.predict(rows)
    elapsed = time.perf_counter() - start
    per_prediction = elapsed / (_LATENCY_REPEATS * len(rows))
    return round(per_prediction * 1000.0, 6)


def artifact_path(model_id: str, version: str) -> Path:
    """아티팩트 파일 경로 — data/models/<model_id>/<version>.json (gitignore 대상)."""
    return get_settings().model_artifact_dir / model_id / f"{version}.json"


def _write_artifact(result: TrainingResult, version: str) -> Path:
    """학습 산출물을 로컬 JSON으로 저장. 승급 여부와 무관하게 기록한다(실험 산출물)."""
    path = artifact_path(result.model_id, version)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": result.model_id,
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "hyperparams": result.best,
        "metrics": result.candidate_metrics,
        "baseline_metrics": result.baseline_metrics,
        "dataset": result.dataset,
        "latency_ms": result.latency_ms,
        "candidates": result.candidates,
        "model": result.model,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
