"""R5 분석·진단·예측·대응 후보 엔진.

A·B 모듈을 그대로 쓴다: `snapshot`(스냅샷 조회) · `features.add_month`(월 연산) · `models`(예측) ·
`candidates.get_active/get_candidate`(활성 모델). 규칙 임계치·대응 문구는 `rules.py`(데이터)에서
가져온다. 정책 효과 수치·RICE·`policyStrategies`는 쓰지 않는다(계약 R5 금지) — 대응은 항상
"후보"(카테고리·근거·한계)까지만 담는다.

두 원천의 `y` 필드가 이미 타깃 값이다(방문 스냅샷의 `y`=nonlocal_visitors, 소비 스냅샷의
`y`=observed_sales_krw) — 규칙 엔진은 타깃 구분 없이 `y`만 본다. `현재` 구획만 방문의
local/foreign_visitors 부가 필드를 추가로 노출한다.
"""

from __future__ import annotations

from typing import Any

from src.realdata import candidates as candidates_mod
from src.realdata import models as models_mod
from src.realdata import rules
from src.realdata import snapshot as snapshot_mod
from src.realdata.errors import RealdataError
from src.realdata.features import add_month, build_feature_rows

DONG_ALL = "all"
VISITOR_TARGET = "nonlocal_visitors"
SALES_TARGET = "observed_sales_krw"

# 현재(current) 구획에 노출할 필드 — (원본 row 키, 표시 필드명).
_CURRENT_FIELDS: dict[str, list[tuple[str, str]]] = {
    VISITOR_TARGET: [("y", "nonlocal_visitors"), ("local_visitors", "local_visitors"), ("foreign_visitors", "foreign_visitors")],
    SALES_TARGET: [("y", "observed_sales_krw")],
}


# ── 데이터셋 로딩 ──────────────────────────────────────────────
def _latest_dataset_summary(target: str) -> dict[str, Any] | None:
    matches = [d for d in snapshot_mod.list_datasets() if d["spec"].get("target") == target]
    if not matches:
        return None
    return max(matches, key=lambda d: d["created_at"])


def _rows_by_month(rows: list[dict[str, Any]], dong_code: str) -> dict[int, dict[str, float]]:
    """base_ym -> 필드별 합계. `dong_code == "all"`이면 23동 합산, 아니면 해당 동만."""
    out: dict[int, dict[str, float]] = {}
    for row in rows:
        if dong_code != DONG_ALL and row["dong_code"] != dong_code:
            continue
        bucket = out.setdefault(row["base_ym"], {})
        for key, value in row.items():
            if key in ("base_ym", "dong_code") or value is None:
                continue
            bucket[key] = bucket.get(key, 0) + value
    return out


# ── current ────────────────────────────────────────────────────
def _compare(current: float, previous: float | None) -> dict[str, Any] | None:
    if previous is None:
        return None
    diff = current - previous
    return {"previous": previous, "diff": diff, "pct": (diff / previous) if previous else None}


def _current_field(months: dict[int, dict[str, float]], base_ym: int, row_key: str, display_name: str) -> dict[str, Any]:
    current = months.get(base_ym, {}).get(row_key)
    if current is None:
        observed = sorted(months.keys())
        return {
            "status": "empty",
            "field": display_name,
            "message": f"{base_ym} 관측이 없습니다.",
            "observed_range": {"from": observed[0], "to": observed[-1]} if observed else None,
        }
    return {
        "status": "ok",
        "field": display_name,
        "value": current,
        "yoy": _compare(current, months.get(add_month(base_ym, -12), {}).get(row_key)),
        "mom": _compare(current, months.get(add_month(base_ym, -1), {}).get(row_key)),
    }


def _build_current(visitor_months: dict[int, dict[str, float]], sales_months: dict[int, dict[str, float]], base_ym: int) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for months, target in ((visitor_months, VISITOR_TARGET), (sales_months, SALES_TARGET)):
        for row_key, display in _CURRENT_FIELDS[target]:
            metrics[display] = _current_field(months, base_ym, row_key, display)
    status = "ok" if any(m["status"] == "ok" for m in metrics.values()) else "empty"
    return {"status": status, "metrics": metrics}


# ── diagnosis (rules_v1) ───────────────────────────────────────
def _rule_yoy_decline_streak(months: dict[int, dict[str, float]], target: str, base_ym: int, dong_code: str) -> dict[str, Any] | None:
    window = [add_month(base_ym, -k) for k in range(rules.YOY_DECLINE_STREAK_MONTHS - 1, -1, -1)]
    values: list[float] = []
    baselines: list[float] = []
    for m in window:
        value = months.get(m, {}).get("y")
        baseline = months.get(add_month(m, -12), {}).get("y")
        if value is None or baseline is None:
            return None
        values.append(value)
        baselines.append(baseline)
    if not all(v < b for v, b in zip(values, baselines)):
        return None
    return {
        "rule_id": f"yoy_decline_streak_{target}",
        "rules_version": rules.RULES_VERSION,
        "summary": f"{rules.TARGET_LABELS[target]} 최근 {rules.YOY_DECLINE_STREAK_MONTHS}개월 연속 전년동월 대비 감소",
        "evidence": {"dong_code": dong_code, "months": window, "values": values, "baseline_values": baselines},
        "limitations": rules.limitations_for(target),
    }


def _rule_seasonal_deviation(months: dict[int, dict[str, float]], target: str, base_ym: int, dong_code: str) -> dict[str, Any] | None:
    current = months.get(base_ym, {}).get("y")
    if current is None:
        return None
    month_num = base_ym % 100
    history = sorted((m, v["y"]) for m, v in months.items() if m != base_ym and m % 100 == month_num and "y" in v)
    if len(history) < 2:
        return None
    baseline = sum(v for _, v in history) / len(history)
    if baseline == 0:
        return None
    deviation = (current - baseline) / baseline
    if abs(deviation) < rules.SEASONAL_DEVIATION_THRESHOLD:
        return None
    return {
        "rule_id": f"seasonal_deviation_{target}",
        "rules_version": rules.RULES_VERSION,
        "summary": f"{rules.TARGET_LABELS[target]} 계절 대비 이탈(같은 달 과거 평균 대비 {deviation:+.1%})",
        "evidence": {
            "dong_code": dong_code,
            "months": [base_ym],
            "values": [current],
            "baseline_values": [baseline],
            "history_months": [m for m, _ in history],
        },
        "limitations": rules.limitations_for(target),
    }


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    denom = (var_x * var_y) ** 0.5
    return cov / denom if denom else 0.0


def _rule_divergence(
    visitor_months: dict[int, dict[str, float]], sales_months: dict[int, dict[str, float]], base_ym: int, dong_code: str
) -> dict[str, Any] | None:
    common = sorted(m for m in (set(visitor_months) & set(sales_months)) if m <= base_ym)
    if len(common) < rules.MIN_COMMON_MONTHS_FOR_DIVERGENCE:
        return None
    visits = [visitor_months[m]["y"] for m in common]
    sales = [sales_months[m]["y"] for m in common]
    if any(v == 0 for v in visits):
        return None
    ratios = [s / v for s, v in zip(sales, visits)]

    recent_n = min(rules.MIN_COMMON_MONTHS_FOR_DIVERGENCE, len(common))
    recent_months, recent_ratios = common[-recent_n:], ratios[-recent_n:]
    baseline_ratios = ratios[:-recent_n]
    if not baseline_ratios:
        return None
    baseline_ratio = sum(baseline_ratios) / len(baseline_ratios)
    recent_ratio = sum(recent_ratios) / len(recent_ratios)
    if baseline_ratio == 0:
        return None
    deviation = (recent_ratio - baseline_ratio) / baseline_ratio
    if abs(deviation) < rules.DIVERGENCE_RATIO_DEVIATION_THRESHOLD:
        return None

    return {
        "rule_id": "visit_sales_divergence",
        "rules_version": rules.RULES_VERSION,
        "summary": f"방문 대비 소비 비율 괴리(최근 대비 편차 {deviation:+.1%}, 상관 {_pearson(visits, sales):.2f})",
        "evidence": {
            "dong_code": dong_code,
            "months": recent_months,
            "values": recent_ratios,
            "baseline_values": [baseline_ratio] * len(recent_ratios),
            "common_months": common,
            "correlation": _pearson(visits, sales),
        },
        "limitations": rules.limitations_for("both"),
    }


def _build_diagnosis(
    visitor_months: dict[int, dict[str, float]], sales_months: dict[int, dict[str, float]], base_ym: int, dong_code: str
) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    def _has_enough(months: dict[int, dict[str, float]], target: str) -> bool:
        if len(months) < rules.MIN_OBSERVED_MONTHS:
            skipped.append(
                {
                    "target": target,
                    "reason": f"관측 {len(months)}개월 — 최소 {rules.MIN_OBSERVED_MONTHS}개월 미만이라 규칙을 적용하지 않았습니다.",
                }
            )
            return False
        return True

    visitor_ok = _has_enough(visitor_months, VISITOR_TARGET)
    sales_ok = _has_enough(sales_months, SALES_TARGET)

    for months, target, ok in ((visitor_months, VISITOR_TARGET, visitor_ok), (sales_months, SALES_TARGET, sales_ok)):
        if not ok:
            continue
        for finding in (
            _rule_yoy_decline_streak(months, target, base_ym, dong_code),
            _rule_seasonal_deviation(months, target, base_ym, dong_code),
        ):
            if finding:
                findings.append(finding)

    if visitor_ok and sales_ok:
        divergence = _rule_divergence(visitor_months, sales_months, base_ym, dong_code)
        if divergence:
            findings.append(divergence)
    elif not (visitor_ok or sales_ok):
        pass  # 위에서 이미 둘 다 skipped에 사유가 남는다
    else:
        skipped.append({"target": "visit_sales_divergence", "reason": "방문·소비 양쪽 모두 관측 12개월 이상이 필요합니다."})

    return {"status": "ok", "rules_version": rules.RULES_VERSION, "findings": findings, "skipped": skipped}


# ── responses ────────────────────────────────────────────────
def _build_responses(findings: list[dict[str, Any]]) -> dict[str, Any]:
    if not findings:
        return {"status": "insufficient_evidence", "candidates": []}
    candidates: list[dict[str, Any]] = []
    for finding in findings:
        template = rules.RESPONSE_CANDIDATES.get(finding["rule_id"])
        if template is None:
            continue
        candidates.append(
            {
                "rule_id": finding["rule_id"],
                "category": template["category"],
                "text": template["text"],
                "based_on": {
                    "dong_code": finding["evidence"]["dong_code"],
                    "months": finding["evidence"]["months"],
                    "values": finding["evidence"]["values"],
                },
                "limitations": finding["limitations"],
            }
        )
    return {"status": "ok" if candidates else "insufficient_evidence", "candidates": candidates}


# ── forecast ───────────────────────────────────────────────────
def _forecast_for_model(model_id: str, dong_code: str) -> dict[str, Any]:
    active = candidates_mod.get_active(model_id)
    if active is None:
        return {"status": "model_required", "model_id": model_id, "message": "활성 모델이 없습니다."}

    candidate = candidates_mod.get_candidate(model_id, active["version"])
    if candidate is None:
        return {"status": "model_required", "model_id": model_id, "message": "활성 모델 후보 정보를 찾을 수 없습니다."}

    try:
        artifact = models_mod.load_artifact(model_id, active["version"])
        dataset = snapshot_mod.load_dataset(candidate["dataset_id"])
    except RealdataError as exc:
        return {"status": "error", "model_id": model_id, "version": active["version"], "message": str(exc)}

    forecast_month = artifact["forecast_month"]
    observed_end_month = artifact["observed_end_month"]

    feature_rows = build_feature_rows(dataset, for_month=forecast_month)
    if dong_code != DONG_ALL:
        feature_rows = [row for row in feature_rows if row["dong_code"] == dong_code]
    if not feature_rows:
        return {
            "status": "insufficient_data",
            "model_id": model_id,
            "version": active["version"],
            "message": f"{forecast_month} 예측에 필요한 랙 관측이 없습니다.",
        }

    predictions = models_mod.predict(artifact, feature_rows)
    prediction = sum(predictions) if dong_code == DONG_ALL else predictions[0]

    dataset_months = _rows_by_month(dataset.rows, dong_code)
    baseline_row = dataset_months.get(add_month(forecast_month, -12)) or dataset_months.get(add_month(forecast_month, -1))

    return {
        "status": "ok",
        "model_id": model_id,
        "version": active["version"],
        "observed_end_month": observed_end_month,
        "forecast_month": forecast_month,
        "prediction": prediction,
        "baseline": {"name": "yoy_or_prev_month", "value": baseline_row.get("y") if baseline_row else None},
        "validation": artifact.get("eval"),
    }


# ── 진입점 ───────────────────────────────────────────────────
def analyze(*, region: str, dong_code: str, base_ym: int, model_id: str | None) -> dict[str, Any]:
    visitor_summary = _latest_dataset_summary(VISITOR_TARGET)
    sales_summary = _latest_dataset_summary(SALES_TARGET)

    if visitor_summary is None and sales_summary is None:
        return {
            "status": "empty",
            "message": "생성된 스냅샷이 없습니다. POST /realdata/datasets로 먼저 스냅샷을 생성하세요.",
            "data": None,
        }

    visitor_dataset = snapshot_mod.load_dataset(visitor_summary["dataset_id"]) if visitor_summary else None
    sales_dataset = snapshot_mod.load_dataset(sales_summary["dataset_id"]) if sales_summary else None

    visitor_months = _rows_by_month(visitor_dataset.rows, dong_code) if visitor_dataset else {}
    sales_months = _rows_by_month(sales_dataset.rows, dong_code) if sales_dataset else {}

    current = _build_current(visitor_months, sales_months, base_ym)
    diagnosis = _build_diagnosis(visitor_months, sales_months, base_ym, dong_code)
    responses = _build_responses(diagnosis["findings"])

    model_ids = [model_id] if model_id else list(snapshot_mod.TARGETS.values())
    forecast_models = {mid: _forecast_for_model(mid, dong_code) for mid in model_ids}
    forecast_status = "ok" if any(m["status"] == "ok" for m in forecast_models.values()) else next(iter(forecast_models.values()))["status"]

    provenance = {
        "dataset_ids": {
            VISITOR_TARGET: visitor_summary["dataset_id"] if visitor_summary else None,
            SALES_TARGET: sales_summary["dataset_id"] if sales_summary else None,
        },
        "models": {mid: {"version": entry.get("version")} for mid, entry in forecast_models.items()},
        "observed_end_month": {
            VISITOR_TARGET: visitor_summary["observed_to"] if visitor_summary else None,
            SALES_TARGET: sales_summary["observed_to"] if sales_summary else None,
        },
        "rules_version": rules.RULES_VERSION,
    }

    return {
        "status": "ok",
        "message": None,
        "data": {
            "region": region,
            "dong_code": dong_code,
            "base_ym": base_ym,
            "current": current,
            "diagnosis": diagnosis,
            "forecast": {"status": forecast_status, "models": forecast_models},
            "responses": responses,
            "provenance": provenance,
        },
    }
