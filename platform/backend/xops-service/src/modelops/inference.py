"""Active artifact inference and observed-input sensitivity; no persistent writes."""
from __future__ import annotations

import json
import math
from typing import Any
from uuid import uuid4

from fastapi import HTTPException

from src.core.settings import get_settings
from src.mlops.training.dataset import build_dataset
from src.modelops import datasets, service, store
from src.realdata.features import add_month, build_feature_rows
from src.realdata.snapshot import DatasetRecord, _content_hash

_REAL_ADJUSTABLE = {"y_lag1", "y_lag2", "y_lag3", "y_yoy"}


def validate_version(mode: str, instance_id: str, version_id: str, *, require_active: bool = True):
    """추론 전 연결·계산 정보 검증 — 모델·버전·학습데이터·아티팩트가 서로 맞는지. 운영 이력 완료 판정도 같은 검사를 쓴다."""
    instance = store.get(mode, "instance", instance_id)
    if require_active and (not instance["active_version"] or instance["active_version"] != version_id):
        raise HTTPException(409, "활성 버전을 확인한 뒤 다시 실행하세요.")
    artifact = service.load_artifact(mode, instance_id, version_id)
    version = store.get(mode, "version", version_id)
    dataset, document = datasets.load(mode, version["dataset_id"], version["dataset_hash"])
    expected = {"model_id": instance_id, "version": version_id, "mode": mode, "source": mode, "region_id": instance["region_id"],
        "target": instance["target"], "model_type": instance["model_type"], "dataset_id": dataset["id"],
        "content_hash": dataset["content_hash"], "job_id": version["job_id"]}
    if (any(artifact.get(key) != value for key, value in expected.items())
        or any(dataset[key] != instance[key] for key in ("region_id", "target"))
        or any(version[key] != instance[key] for key in ("model_type", "region_id", "target"))):
        raise HTTPException(409, "모델과 학습데이터의 연결 정보를 확인하세요.")
    names = artifact["features"]
    arrays = [artifact.get(key, []) for key in ("coef", "train_means", "train_stds")]
    if (not names or len(set(names)) != len(names) or any(len(values) != len(names) for values in arrays)
        or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for values in arrays for v in values)
        or not isinstance(artifact.get("intercept"), (int, float)) or not math.isfinite(artifact["intercept"])
        or any(std <= 0 for std in artifact["train_stds"]) or names != version["features"]):
        raise HTTPException(409, "모델의 입력과 계산 정보를 확인하세요.")
    return instance, artifact, version, dataset, document


def _context(mode: str, instance_id: str, version_id: str, dong_code: str | None):
    instance, artifact, version, dataset, document = validate_version(mode, instance_id, version_id)
    names = artifact["features"]
    if mode == "real":
        record = DatasetRecord(**document)
        if record.target != dataset["target"] or record.content_hash != dataset["content_hash"] or _content_hash(record.rows) != dataset["content_hash"]:
            raise HTTPException(409, "등록한 관측 데이터의 내용을 확인하세요.")
        period = add_month(record.observed_to, 1)
        rows = build_feature_rows(record, for_month=period)
        history = build_feature_rows(record)
        observed = [row for row in history if row["base_ym"] <= add_month(record.observed_to, -3)]
        dong_map = json.loads(get_settings().realdata_dong_map_path.read_text(encoding="utf-8"))
        labels = {entry["dong_code"]: entry["dong_name"] for entry in dong_map["entries"]}
        options = [{"code": row["dong_code"], "name": labels.get(row["dong_code"], row["dong_code"])} for row in rows]
        selected = next((row for row in rows if row["dong_code"] == (dong_code or (rows[0]["dong_code"] if rows else ""))), None)
        observed = [row for row in observed if selected and row["dong_code"] == selected["dong_code"]]
        range_provenance = {"scope": "selected_dong_training_rows", "from": min((row["base_ym"] for row in observed), default=None),
            "to": max((row["base_ym"] for row in observed), default=None), "row_count": len(observed)}
        adjustable = _REAL_ADJUSTABLE - ({"y_yoy"} if not selected or not selected["has_yoy"] else set())
        months = {row["base_ym"]: row for row in history if selected and row["dong_code"] == selected["dong_code"]}
        if selected:
            months[period] = selected
    elif "panel" in document:  # 데모 요인 패널 — 최근 관측월 행을 입력으로 쓴다
        observed = document["panel"]["rows"]
        selected = observed[-1] if observed and dong_code in (None, "all") else None
        options = [{"code": "all", "name": "전체"}]
        period = dataset["observed_to"]
        adjustable = set(names)
        range_provenance = {"scope": "mock_training_rows", "from": dataset["observed_from"], "to": dataset["observed_to"], "row_count": len(observed)}
        months = {row["base_ym"]: row for row in observed}
    else:
        training = build_dataset([document["region"]], horizon=1, max_lag=2, decline_threshold_pct=1.0)
        training_names, matrix = training.select(2)
        if training_names != names:
            raise HTTPException(409, "모델의 입력 항목과 학습데이터를 확인하세요.")
        observed = [dict(zip(names, row)) for row in matrix]
        selected = observed[-1] if observed and dong_code in (None, "all") else None
        options = [{"code": "all", "name": "전체"}]
        period = dataset["observed_to"]
        adjustable = set(names)
        range_provenance = {"scope": "mock_training_rows", "from": dataset["observed_from"], "to": dataset["observed_to"], "row_count": len(observed)}
        months = None  # 옛 지역 인구 추이 원천 — 행에 기준월이 없어 기간 분석을 하지 않는다
    if selected is None or not observed:
        raise HTTPException(422, "선택한 지역의 예측 입력을 만들 수 없습니다. 관측 기간과 행정동을 확인하세요.")
    inputs = {name: selected[name] for name in names}
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in inputs.values()):
        raise HTTPException(409, "관측 입력 수치를 확인하세요.")
    controls = []
    for name in names:
        if name not in adjustable:
            continue
        low, high = min(row[name] for row in observed), max(row[name] for row in observed)
        if high > low:
            controls.append({"feature": name, "min": low, "max": high, "step": (high-low)/100, "default": inputs[name]})
    provenance = {"instance_id": instance_id, "model_type": instance["model_type"], "version_id": version_id,
        "job_id": version["job_id"], "dataset_id": dataset["id"], "dataset_hash": dataset["content_hash"],
        "source_dataset_id": dataset["source_dataset_id"], "artifact_hash": version["artifact_hash"],
        "region_id": instance["region_id"], "target": instance["target"], "features": names,
        "observed_from": dataset["observed_from"], "observed_to": dataset["observed_to"],
        "evaluation_period": artifact.get("eval", {}).get("period"), "metrics": version["metrics"], "baseline": version["baseline"],
        "prediction_period": period, "input_kind": "next_observed_month" if mode == "real" else "last_observed_period",
        "range_provenance": range_provenance, "dong_code": dong_code or options[0]["code"], "mode": mode, "source": mode}
    context = {"mode": mode, "source": mode, "provenance": provenance, "inputs": inputs, "controls": controls, "dong_options": options}
    return artifact, context, months


def inputs(mode: str, instance_id: str, version_id: str, dong_code: str | None = None) -> dict[str, Any]:
    _, context, _ = _context(mode, instance_id, version_id, dong_code)
    return context


def _calculate(artifact: dict[str, Any], values: dict[str, float], reference: dict[str, float] | None = None):
    """예측 = intercept + Σ coef × (x − 학습 평균)/std. 기여도는 reference(비교 달 입력)가 있으면 coef × (x − 기준)/std."""
    terms = list(zip(artifact["features"], artifact["coef"], artifact["train_means"], artifact["train_stds"]))
    from_mean = [coef * ((values[name]-mean)/std) for name, coef, mean, std in terms]
    contributions = [{"feature": name, "input": values[name], "coefficient": coef, "mean": mean, "std": std,
        "value": base if reference is None else coef * ((values[name]-reference[name])/std)}
        for (name, coef, mean, std), base in zip(terms, from_mean)]
    prediction = artifact["intercept"] + sum(from_mean)
    if not math.isfinite(prediction) or any(not math.isfinite(row["value"]) for row in contributions):
        raise HTTPException(422, "입력값이 너무 큽니다. 관측 범위를 참고해 다시 입력하세요.")
    return prediction, contributions


def _adjust(values: dict[str, float], changes: dict[str, float]):
    adjusted = {**values, **changes}
    # 전년 관측이 없을 때의 대체 입력은 기존 feature 계약대로 전월 값에 따른다.
    if adjusted.get("has_yoy") == 0 and "y_lag1" in changes:
        adjusted["y_yoy"] = adjusted["y_lag1"]
    return adjusted


def _difference(prediction: float, baseline: float):
    delta = prediction-baseline
    if not math.isfinite(delta):
        raise HTTPException(422, "예측 변화량을 계산할 수 없습니다. 입력 범위를 확인하세요.")
    return delta


def _check_basis(body: dict[str, Any], provenance: dict[str, Any]) -> None:
    if any(body[key] != provenance[key] for key in ("region_id", "target", "dataset_id", "dataset_hash")):
        raise HTTPException(409, "선택한 모델과 학습데이터를 다시 확인하세요.")


def _check_still_active(mode: str, instance_id: str, version_id: str) -> None:
    if store.get(mode, "instance", instance_id)["active_version"] != version_id:
        raise HTTPException(409, "활성 버전이 변경됐습니다. 모델을 다시 확인하세요.")


def predict(mode: str, instance_id: str, version_id: str, body: dict[str, Any]) -> dict[str, Any]:
    artifact, context, _ = _context(mode, instance_id, version_id, body.get("dong_code"))
    provenance = context["provenance"]
    _check_basis(body, provenance)
    controls = {row["feature"]: row for row in context["controls"]}
    changes = body.get("adjustments", {})
    if any(name not in controls for name in changes):
        raise HTTPException(422, "조정할 수 있는 관측 입력을 선택하세요.")
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in changes.values()):
        raise HTTPException(422, "유효한 입력 수치를 확인하세요.")
    values = _adjust(context["inputs"], changes)
    base, _ = _calculate(artifact, context["inputs"])
    prediction, contributions = _calculate(artifact, values)
    outside = [name for name, control in controls.items() if not control["min"] <= values[name] <= control["max"]]
    recommendations = []
    if provenance["model_type"] == "simulation":
        for name, control in controls.items():
            current = values[name]
            if not control["min"] <= current <= control["max"]:
                continue
            for direction in (-1, 1):
                adjusted = max(control["min"], min(control["max"], current + direction*control["step"]))
                if adjusted == current:
                    continue
                candidate, _ = _calculate(artifact, _adjust(values, {name: adjusted}))
                recommendations.append({"feature": name, "from": current, "to": adjusted, "input_delta": adjusted-current,
                    "prediction": candidate, "delta": _difference(candidate,prediction), "range_min": control["min"], "range_max": control["max"]})
        recommendations.sort(key=lambda row: (-row["delta"], row["feature"], row["to"]))
    _check_still_active(mode, instance_id, version_id)
    return {**context, "inference_id": "inference_" + uuid4().hex, "intercept": artifact["intercept"],
        "baseline_inputs": context["inputs"], "inputs": values, "baseline_prediction": base, "prediction": prediction,
        "delta": _difference(prediction,base), "contributions": contributions, "outside_observed_range": outside, "recommendations": recommendations}


# ── ① 분석 기간·비교 기준(추론 전용, 재학습·저장 없음) ──
PERIOD_MAX_MONTHS = 36


def _month_index(base_ym: int) -> int:
    return (base_ym // 100) * 12 + base_ym % 100 - 1


def _period_bounds(mode: str, observed_from: int, observed_to: int) -> tuple[int, int]:
    """기간으로 고를 수 있는 달 — 실데이터는 랙 3개월이 생기는 달부터 예측월(관측 다음 달)까지, 데모 패널은 관측 기간."""
    if mode == "real":
        return add_month(observed_from, 3), add_month(observed_to, 1)
    return observed_from, observed_to


def _period_months(body: dict[str, Any], bounds: tuple[int, int]) -> list[int]:
    p_from, p_to = body["period_from"], body["period_to"]
    if any(not 1 <= value % 100 <= 12 for value in (p_from, p_to)):
        raise HTTPException(422, "분석 기간은 연월(YYYYMM)로 입력하세요.")
    if p_from > p_to:
        raise HTTPException(422, "분석 기간의 시작 달이 끝 달보다 늦습니다.")
    length = _month_index(p_to) - _month_index(p_from) + 1
    if length > PERIOD_MAX_MONTHS:
        raise HTTPException(422, f"분석 기간은 {PERIOD_MAX_MONTHS}개월 이내로 고르세요.")
    if p_from < bounds[0] or p_to > bounds[1]:
        raise HTTPException(422, "분석 기간을 학습데이터의 관측 범위 안에서 고르세요.")
    return [add_month(p_from, k) for k in range(length)]


def _reference_ym(t: int, baseline: str, length: int) -> int | None:
    return {"prev_year": add_month(t, -12), "prev_period": add_month(t, -length)}.get(baseline)


def period_analysis(mode: str, instance_id: str, version_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """기간 안 달마다 활성 모델의 기여도를 비교 기준 대비로 계산한다. 입력·기준 달 행이 없는 달은 보간 없이 뺀다.
    항등식: 달마다 prediction − reference_prediction = Σ contributions.value (학습 평균 기준의 reference_prediction = intercept)."""
    artifact, context, history = _context(mode, instance_id, version_id, body.get("dong_code"))
    provenance = context["provenance"]
    _check_basis(body, provenance)
    if history is None:
        raise HTTPException(422, "이 모델은 기간 분석을 지원하지 않습니다. 관측 월이 있는 학습데이터로 학습한 모델을 고르세요.")
    months = _period_months(body, _period_bounds(mode, provenance["observed_from"], provenance["observed_to"]))
    baseline, names = body["baseline"], artifact["features"]
    means = dict(zip(names, artifact["train_means"]))
    rows, skipped = [], []
    for t in months:
        row, ref_ym = history.get(t), _reference_ym(t, baseline, len(months))
        ref_row = history.get(ref_ym) if ref_ym is not None else None
        status = "no_input" if row is None else "no_reference" if ref_ym is not None and ref_row is None else "used"
        entry: dict[str, Any] = {"base_ym": t, "status": status, "reference_ym": ref_ym, "prediction": None,
            "reference_prediction": None, "actual": row.get("y") if row else None, "contributions": []}
        if status == "used":
            values = {name: row[name] for name in names}
            reference = {name: ref_row[name] for name in names} if ref_row is not None else None
            prediction, contributions = _calculate(artifact, values, reference)
            entry["prediction"] = prediction
            entry["reference_prediction"] = _calculate(artifact, reference)[0] if reference else artifact["intercept"]
            entry["contributions"] = [{"feature": c["feature"], "input": c["input"],
                "reference_input": reference[c["feature"]] if reference else means[c["feature"]], "value": c["value"]} for c in contributions]
        else:
            skipped.append({"base_ym": t, "reason": status})
        rows.append(entry)
    used = [row for row in rows if row["status"] == "used"]
    if not used:
        raise HTTPException(422, "분석 기간에 계산할 수 있는 달이 없습니다. 기간이나 비교 기준을 바꾸세요.")
    total = dict.fromkeys(names, 0.0)
    for row in used:
        for c in row["contributions"]:
            total[c["feature"]] += c["value"]
    summary = {"used_months": len(used), "skipped": skipped,
        "total": [{"feature": name, "value": value} for name, value in total.items()],
        "mean": [{"feature": name, "value": value / len(used)} for name, value in total.items()],
        "total_delta": _difference(sum(row["prediction"] for row in used), sum(row["reference_prediction"] for row in used))}
    _check_still_active(mode, instance_id, version_id)
    return {"mode": mode, "source": mode, "intercept": artifact["intercept"],
        "provenance": {**provenance, "period_from": months[0], "period_to": months[-1], "period_baseline": baseline},
        "months": rows, "summary": summary}
