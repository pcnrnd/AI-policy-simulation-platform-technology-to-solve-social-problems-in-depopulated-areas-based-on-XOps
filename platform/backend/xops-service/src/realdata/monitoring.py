"""R4 모니터링 — evaluation(validation/operational)·explain(선형 SHAP)·drift(PSI).

기존 `src.mlops.monitoring.drift.population_stability_index`(PSI 계산 본체)를 재사용한다.
시드 폴백은 하지 않는다 — 실데이터 조회·아티팩트 로딩이 실패하면 status="error"로 노출한다.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, Sequence

from src.mlops.monitoring.drift import population_stability_index
from src.realdata import candidates as candidates_mod

_DRIFT_BINS = 10
_RECONSTRUCTION_TOLERANCE = 1e-6
_EXPLAIN_NOTE = "독립 특성 가정·선형 기여도이며 인과 효과가 아닙니다(선형 SHAP)."


def _import_models() -> ModuleType:
    from src.realdata import models

    return models


def _import_snapshot() -> ModuleType:
    from src.realdata import snapshot

    return snapshot


def _import_errors() -> ModuleType:
    from src.realdata import errors

    return errors


def _rows_of(dataset: Any) -> list[dict[str, Any]]:
    if isinstance(dataset, dict):
        return list(dataset.get("rows", []))
    return list(getattr(dataset, "rows", []) or [])


# ── evaluation ───────────────────────────────────────────────
def evaluation(model_id: str, version: str) -> dict[str, Any]:
    """`GET /realdata/models/{model_id}/evaluation?version=`(R4).

    validation = 학습 시 3개월 순차 평가(후보에 저장된 metrics/baseline/eval_period).
    operational = 반영 후 도착한 관측으로 계산 — 아직 도착하지 않았으면 kind="pending"."""
    candidate = candidates_mod.get_candidate(model_id, version)
    if candidate is None:
        return {"status": "empty", "message": f"후보를 찾을 수 없습니다: {model_id}/{version}", "data": None}

    metrics = candidate["metrics"]
    validation = {
        "kind": "validation",
        "metrics": {k: metrics.get(k) for k in ("mae", "rmse", "wape")},
        "baseline": candidate["baseline"],
        "eval_period": metrics.get("eval_period"),
    }
    operational = _operational_evaluation(model_id, candidate)
    return {"status": "ok", "message": None, "data": {"validation": validation, "operational": operational}}


def _operational_evaluation(model_id: str, candidate: dict[str, Any]) -> dict[str, Any]:
    models_mod = _import_models()
    snapshot_mod = _import_snapshot()
    errors_mod = _import_errors()

    try:
        artifact = models_mod.load_artifact(model_id, candidate["version"])
    except errors_mod.RealdataError as exc:
        return {"kind": "error", "message": str(exc)}

    observed_end = artifact.get("observed_end_month")
    forecast_month = artifact.get("forecast_month")
    if observed_end is None or forecast_month is None:
        return {"kind": "error", "message": "아티팩트에 observed_end_month/forecast_month가 없습니다."}

    try:
        dataset = snapshot_mod.load_dataset(candidate["dataset_id"])
    except errors_mod.RealdataError as exc:
        return {"kind": "error", "message": str(exc)}

    rows_for_month = [row for row in _rows_of(dataset) if row.get("base_ym") == forecast_month and "y" in row]
    if not rows_for_month:
        return {"kind": "pending", "pending_months": [forecast_month]}

    features = artifact.get("features", [])
    pairs = []
    for row in rows_for_month:
        feature_row = {feature: row.get(feature) for feature in features}
        predicted = models_mod.predict(artifact, feature_row)
        pairs.append((row["y"], predicted))

    abs_errors = [abs(actual - predicted) for actual, predicted in pairs]
    mae = sum(abs_errors) / len(abs_errors)
    rmse = (sum(err**2 for err in abs_errors) / len(abs_errors)) ** 0.5
    denom = sum(abs(actual) for actual, _ in pairs) or 1.0
    wape = sum(abs_errors) / denom

    return {
        "kind": "operational",
        "base_ym": forecast_month,
        "n": len(pairs),
        "metrics": {"mae": mae, "rmse": rmse, "wape": wape},
    }


# ── explain (선형 SHAP) ──────────────────────────────────────
def explain(model_id: str, version: str, base_ym: int, dong_code: str) -> dict[str, Any]:
    """`GET /realdata/models/{model_id}/explain?version=&base_ym=&dong_code=`(R4).

    phi_j = coef_j × (x_j − mean_j) / std_j (표준화 공간 계수 기준),
    base_value = intercept, reconstruction_check = |base_value + Σphi − prediction| (허용 1e-6)."""
    models_mod = _import_models()
    snapshot_mod = _import_snapshot()
    errors_mod = _import_errors()

    try:
        artifact = models_mod.load_artifact(model_id, version)
    except errors_mod.RealdataError as exc:
        return {"status": "error", "message": str(exc), "data": None}

    try:
        dataset = snapshot_mod.load_dataset(artifact.get("dataset_id"))
    except errors_mod.RealdataError as exc:
        return {"status": "error", "message": str(exc), "data": None}

    row = next(
        (r for r in _rows_of(dataset) if r.get("base_ym") == base_ym and r.get("dong_code") == dong_code),
        None,
    )
    if row is None:
        return {"status": "empty", "message": f"{base_ym}/{dong_code} 관측행이 없습니다.", "data": None}

    features: list[str] = artifact["features"]
    coef: list[float] = artifact["coef"]
    means: dict[str, float] = artifact["train_means"]
    stds: dict[str, float] = artifact["train_stds"]
    intercept: float = artifact["intercept"]

    contributions = []
    total = 0.0
    for i, feature in enumerate(features):
        if feature not in row:
            return {"status": "error", "message": f"학습행에 피처 {feature}가 없습니다.", "data": None}
        x = row[feature]
        std = stds.get(feature) or 1e-9
        phi = coef[i] * (x - means.get(feature, 0.0)) / std
        contributions.append({"feature": feature, "value": x, "phi": phi})
        total += phi

    prediction = models_mod.predict(artifact, {feature: row.get(feature) for feature in features})
    base_value = intercept
    reconstruction_check = abs(base_value + total - prediction)

    return {
        "status": "ok",
        "message": None,
        "data": {
            "base_value": base_value,
            "contributions": contributions,
            "prediction": prediction,
            "reconstruction_check": reconstruction_check,
            "within_tolerance": reconstruction_check <= _RECONSTRUCTION_TOLERANCE,
            "note": _EXPLAIN_NOTE,
        },
    }


# ── drift (PSI, 고정 구간·bins 10) ───────────────────────────
def _histogram(values: Sequence[float], bins: int, edges: Sequence[float] | None = None) -> dict[str, Any]:
    lo, hi = min(values), max(values)
    if edges is None:
        if hi <= lo:
            hi = lo + 1.0
        step = (hi - lo) / bins
        edges = [lo + step * i for i in range(bins + 1)]
    counts = [0] * bins
    for value in values:
        idx = bins - 1
        for i in range(bins):
            if edges[i] <= value < edges[i + 1]:
                idx = i
                break
        counts[idx] += 1
    return {"edges": list(edges), "counts": counts}


def _psi_for_field(reference_rows: list[dict[str, Any]], current_rows: list[dict[str, Any]], field: str) -> dict[str, Any] | None:
    ref_values = [row[field] for row in reference_rows if row.get(field) is not None]
    cur_values = [row[field] for row in current_rows if row.get(field) is not None]
    if not ref_values or not cur_values:
        return None
    ref_hist = _histogram(ref_values, _DRIFT_BINS)
    cur_hist = _histogram(cur_values, _DRIFT_BINS, edges=ref_hist["edges"])
    psi = population_stability_index(ref_hist["counts"], cur_hist["counts"])
    return {"feature": field, "psi": round(psi, 6), "n_reference": len(ref_values), "n_current": len(cur_values)}


def drift(model_id: str, version: str) -> dict[str, Any]:
    """`GET /realdata/models/{model_id}/drift?version=`(R4).

    학습 종료 이후 실관측이 있으면 kind="operational"(학습 구간 vs 이후 관측),
    없으면 학습 내 검증 구간(eval_period)을 대신 써서 kind="historical",
    그마저 없으면 data.status="none"."""
    candidate = candidates_mod.get_candidate(model_id, version)
    if candidate is None:
        return {"status": "empty", "message": f"후보를 찾을 수 없습니다: {model_id}/{version}", "data": None}

    models_mod = _import_models()
    snapshot_mod = _import_snapshot()
    errors_mod = _import_errors()

    try:
        artifact = models_mod.load_artifact(model_id, version)
    except errors_mod.RealdataError as exc:
        return {"status": "error", "message": str(exc), "data": None}

    try:
        dataset = snapshot_mod.load_dataset(candidate["dataset_id"])
    except errors_mod.RealdataError as exc:
        return {"status": "error", "message": str(exc), "data": None}

    rows = _rows_of(dataset)
    observed_end = artifact.get("observed_end_month")
    eval_period = candidate["metrics"].get("eval_period") or {}

    post_rows = [row for row in rows if observed_end is not None and (row.get("base_ym") or 0) > observed_end]
    if post_rows:
        eval_from = eval_period.get("from")
        reference_rows = [row for row in rows if eval_from is None or (row.get("base_ym") or 0) < eval_from]
        current_rows = post_rows
        kind: str | None = "operational"
    else:
        eval_from, eval_to = eval_period.get("from"), eval_period.get("to")
        if eval_from is None or eval_to is None:
            return {"status": "ok", "message": None, "data": {"status": "none", "kind": None, "features": [], "target": None}}
        reference_rows = [row for row in rows if (row.get("base_ym") or 0) < eval_from]
        current_rows = [row for row in rows if eval_from <= (row.get("base_ym") or 0) <= eval_to]
        kind = "historical"

    if not reference_rows or not current_rows:
        return {"status": "ok", "message": None, "data": {"status": "none", "kind": None, "features": [], "target": None}}

    features = artifact.get("features", [])
    feature_results = [
        result
        for result in (_psi_for_field(reference_rows, current_rows, feature) for feature in features)
        if result is not None
    ]
    target_result = _psi_for_field(reference_rows, current_rows, "y")

    return {
        "status": "ok",
        "message": None,
        "data": {"status": "ok", "kind": kind, "features": feature_results, "target": target_result},
    }
