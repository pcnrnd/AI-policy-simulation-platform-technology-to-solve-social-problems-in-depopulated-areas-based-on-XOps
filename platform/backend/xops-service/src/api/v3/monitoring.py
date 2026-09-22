"""MLOps 모니터링 엔드포인트 — 6대 지표·드리프트(PSI/KL)·이상치·설명가능성.

GET 계열은 `model_id` 가 주어지면 실측(SQLite 실행 이력·학습 아티팩트)을 먼저 찾고, 없을 때만
mock_data.json 시드로 내려간다. 어느 쪽을 내려줬는지는 응답의 `source`("measured"|"seed")에
항상 표기한다 — 화면이 데모 표시 OFF에서 시드를 값 자리에 넣지 않으려면 이 구분이 필요하다.
POST 계열은 입력으로 실제 계산한다(시드와 무관).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Depends, Query

from src.api.dependencies import optional_auth
from src.core.seed import get_seed
from src.mlops.monitoring.drift import DriftDetector, DriftResult
from src.mlops.monitoring.explain import ExplainabilityModule
from src.mlops.monitoring.metrics import MetricCollector
from src.mlops.monitoring.outliers import OutlierDetector
from src.mlops.monitoring import realdata_bridge, sources
from src.mlops.orchestration.registry import get_registry
from src.schemas.monitoring import ClassificationInput, DriftInput, ExplainInput, OutlierInput, RegressionInput

router = APIRouter(prefix="/monitoring", tags=["monitoring"])
_metrics = MetricCollector()
_drift = DriftDetector()
_outliers = OutlierDetector()
_explain = ExplainabilityModule()

_SERIES = ("accuracy", "f1", "precision", "recall", "mse", "mae")

# GET 3종의 시드 폴백 게이트. 기본 false = 데모 표시 OFF — 실측이 없으면 시드 값을 내려보내는
# 대신 빈 응답(`source: null`)으로 답한다. 화면이 `source` 만 보고 거르던 규칙을 응답 자체로
# 옮긴 것이라, 데모 OFF에서는 시드 수치가 네트워크 응답에도 남지 않는다.
_INCLUDE_SEED = Query(False, description="실측이 없을 때 시드 값으로 폴백 — 데모 표시 ON 전용")

# 데모 표시 OFF(`include_seed=false`)에서 실데이터 모델을 물어보면 rd_* 저장소를 같은 스키마로
# 돌려준다. 실데이터는 `data:read` 토큰이 있는 호출에만 싣는다 — 토큰이 없으면 기존 공개 GET
# 계약 그대로 빈 응답이다(401을 새로 만들지 않고, 무인증으로 실데이터를 흘리지도 않는다).
_OPTIONAL_AUTH = Depends(optional_auth("data:read"))


def _realdata_allowed(model_id: str | None, include_seed: bool, auth: dict[str, Any] | None) -> bool:
    return not include_seed and auth is not None and realdata_bridge.is_realdata_model(model_id)


def _maybe_retrain(result: DriftResult, model_id: str | None, auto_retrain: bool) -> dict[str, Any] | None:
    """드리프트가 임계를 넘고 auto_retrain이면 해당 모델의 재학습을 자동 발화."""
    if not (result.drifted and auto_retrain and model_id):
        return None
    from dataclasses import asdict as _asdict

    return _asdict(get_registry().trigger(model_id=model_id, trigger="drift"))


@router.get("/metrics")
def metrics_history(
    model_id: str | None = Query(None, description="실측 지표를 읽을 모델. 생략하면 시드"),
    include_seed: bool = _INCLUDE_SEED,
    auth: dict[str, Any] | None = _OPTIONAL_AUTH,
) -> dict[str, Any]:
    """6대 지표 시계열 + 최신 스냅샷.

    `model_id` 의 실측 학습 실행이 있으면 그 지표 추이(`source="measured"`)를, 없으면
    시드 시계열(`source="seed"`)을 돌려준다. `include_seed=false` 면 시드 대신 빈 시계열이다.
    """
    if _realdata_allowed(model_id, include_seed, auth):
        realdata = realdata_bridge.metrics(model_id)  # type: ignore[arg-type]
        if realdata is not None:
            return realdata
    measured = sources.measured_metrics(model_id) if model_id else None
    if measured is not None:
        return {**measured, "source": "measured", "model_id": model_id}
    if not include_seed:
        return {"history": {}, "latest": {}, "labels": None, "latency_ms": None, "source": None}
    hist = get_seed()["metrics_history"]
    latest = {k: hist[k][-1] for k in _SERIES if k in hist}
    return {"history": hist, "latest": latest, "labels": None, "latency_ms": None, "source": "seed"}


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
    include_seed: bool = _INCLUDE_SEED,
    auth: dict[str, Any] | None = _OPTIONAL_AUTH,
) -> dict[str, Any]:
    """시드 분포(reference vs current_normal|current_drifted)로 PSI/KL 판정.

    PSI/KL 계산은 실계산이지만 **입력 분포가 시드**다. 실 추론 입력 수집 경로가 아직 없으므로
    `include_seed=false`(데모 표시 OFF)에서는 판정 자체를 내지 않고 빈 분포를 돌려준다.
    실데이터 드리프트는 `GET /realdata/models/{model_id}/drift` 가 따로 담당한다.
    """
    if _realdata_allowed(model_id, include_seed, auth):
        realdata = realdata_bridge.drift(model_id)  # type: ignore[arg-type]
        if realdata is not None:
            return realdata
    if not include_seed:
        return {"buckets": [], "reference": [], "current": [], "source": None, "retrain": None}
    dist = get_seed()["drift_distribution"]
    current = dist["current_drifted"] if drifted else dist["current_normal"]
    result = _drift.detect(dist["reference"], current)
    return {
        **asdict(result),
        "buckets": dist["buckets"],
        "reference": dist["reference"],
        "current": current,
        # PSI/KL 계산 자체는 실계산이지만 **입력 분포가 시드**다. 실 추론 입력 수집 경로가
        # 없는 동안은 실측이라고 표기하지 않는다(화면은 이 표기로 빈 상태를 고른다).
        "source": "seed",
        "retrain": _maybe_retrain(result, model_id, auto_retrain),
    }


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
def explain_from_seed(
    model_id: str | None = Query(None, description="실측 기여도를 읽을 모델. 생략하면 시드"),
    include_seed: bool = _INCLUDE_SEED,
    auth: dict[str, Any] | None = _OPTIONAL_AUTH,
) -> dict[str, Any]:
    """특징 중요도 + 사용 backend 표기.

    `model_id` 의 현행 운영 버전에 학습 아티팩트가 있으면 그 표준화 계수를 기여도로 쓴다
    (`source="measured"`). 없으면 시드 `shap_features`(`source="seed"`).
    """
    if _realdata_allowed(model_id, include_seed, auth):
        realdata = realdata_bridge.explain(model_id)  # type: ignore[arg-type]
        if realdata is not None:
            return {"backend": "linear-shap", **realdata}
    if model_id:
        version = next(
            (m["version"] for m in get_registry().models() if m["model_id"] == model_id), None
        )
        measured = sources.measured_features(model_id, version) if version else None
        if measured:
            return {
                "backend": _explain.backend,
                "features": measured,
                "source": "measured",
                "model_id": model_id,
                "version": version,
                "basis": "표준화 릿지 계수 (1 표준편차당 타깃 변화율)",
            }
    if not include_seed:
        return {"backend": _explain.backend, "features": [], "source": None}
    return {"backend": _explain.backend, "features": get_seed()["shap_features"], "source": "seed"}


@router.post("/explain")
def rank_features(body: ExplainInput) -> dict[str, Any]:
    """feature 기여도 → 중요도 정렬."""
    return {"backend": _explain.backend, "features": _explain.rank_features(body.contributions)}
