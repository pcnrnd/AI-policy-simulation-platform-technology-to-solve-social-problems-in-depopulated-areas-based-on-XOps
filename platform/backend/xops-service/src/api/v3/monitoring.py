"""MLOps 모니터링 엔드포인트 — 6대 지표·드리프트(PSI/KL)·이상치·설명가능성.

GET 계열은 mock_data.json 시드(대시보드 시계열)를 서빙하고, POST 계열은 입력으로 실제 계산.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Query

from src.core.seed import get_seed
from src.mlops.monitoring.drift import DriftDetector, DriftResult
from src.mlops.monitoring.explain import ExplainabilityModule
from src.mlops.monitoring.metrics import MetricCollector
from src.mlops.monitoring.outliers import OutlierDetector
from src.mlops.orchestration.registry import get_registry
from src.schemas.monitoring import ClassificationInput, DriftInput, ExplainInput, OutlierInput, RegressionInput

router = APIRouter(prefix="/monitoring", tags=["monitoring"])
_metrics = MetricCollector()
_drift = DriftDetector()
_outliers = OutlierDetector()
_explain = ExplainabilityModule()

_SERIES = ("accuracy", "f1", "precision", "recall", "mse", "mae")


def _maybe_retrain(result: DriftResult, model_id: str | None, auto_retrain: bool) -> dict[str, Any] | None:
    """드리프트가 임계를 넘고 auto_retrain이면 해당 모델의 재학습을 자동 발화."""
    if not (result.drifted and auto_retrain and model_id):
        return None
    from dataclasses import asdict as _asdict

    return _asdict(get_registry().trigger(model_id=model_id, trigger="drift"))


@router.get("/metrics")
def metrics_history() -> dict[str, Any]:
    """6대 지표 시계열 + 최신 스냅샷 (대시보드용 시드)."""
    hist = get_seed()["metrics_history"]
    latest = {k: hist[k][-1] for k in _SERIES if k in hist}
    return {"history": hist, "latest": latest}


@router.post("/metrics/regression")
def compute_regression(body: RegressionInput) -> dict[str, float]:
    """MSE·MAE 실제 계산."""
    return _metrics.regression(body.y_true, body.y_pred)


@router.post("/metrics/classification")
def compute_classification(body: ClassificationInput) -> dict[str, float]:
    """Accuracy·Precision·Recall·F1 실제 계산."""
    return _metrics.classification(body.y_true, body.y_pred, body.positive)


@router.get("/drift")
def drift_from_seed(
    drifted: bool = Query(False, description="true면 드리프트 주입 분포 사용"),
    model_id: str | None = Query(None, description="드리프트 감지 시 재학습 대상 모델"),
    auto_retrain: bool = Query(False, description="드리프트 임계 초과 시 재학습 자동 발화"),
) -> dict[str, Any]:
    """시드 분포(reference vs current_normal|current_drifted)로 PSI/KL 판정."""
    dist = get_seed()["drift_distribution"]
    current = dist["current_drifted"] if drifted else dist["current_normal"]
    result = _drift.detect(dist["reference"], current)
    return {**asdict(result), "buckets": dist["buckets"], "retrain": _maybe_retrain(result, model_id, auto_retrain)}


@router.post("/drift")
def compute_drift(
    body: DriftInput,
    model_id: str | None = Query(None, description="드리프트 감지 시 재학습 대상 모델"),
    auto_retrain: bool = Query(False, description="드리프트 임계 초과 시 재학습 자동 발화"),
) -> dict[str, Any]:
    """임의 분포로 PSI/KL 판정."""
    result = _drift.detect(body.reference, body.current)
    return {**asdict(result), "retrain": _maybe_retrain(result, model_id, auto_retrain)}


@router.post("/outliers")
def detect_outliers(body: OutlierInput, method: str = Query("zscore", pattern="^(zscore|iqr)$")) -> dict[str, Any]:
    """Z-score(기본) 또는 IQR 이상치 탐지."""
    result = _outliers.iqr(body.values) if method == "iqr" else _outliers.zscore(body.values)
    return asdict(result)


@router.get("/explain")
def explain_from_seed() -> dict[str, Any]:
    """SHAP 특징 중요도 (시드) + 사용 backend 표기."""
    return {"backend": _explain.backend, "features": get_seed()["shap_features"]}


@router.post("/explain")
def rank_features(body: ExplainInput) -> dict[str, Any]:
    """feature 기여도 → 중요도 정렬."""
    return {"backend": _explain.backend, "features": _explain.rank_features(body.contributions)}
