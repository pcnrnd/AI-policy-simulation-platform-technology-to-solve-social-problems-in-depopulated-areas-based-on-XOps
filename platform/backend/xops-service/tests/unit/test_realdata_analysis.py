"""R5 분석 엔진 단위 테스트 — 규칙 3종(연속 감소·괴리·계절 이탈), 근거 부족, 효과 수치 부재,
model_required/insufficient_data 분기를 검증한다."""

from __future__ import annotations

from typing import Any

import pytest

from src.realdata import analysis
from src.realdata.features import add_month

_DONG = "45190250"
_VISITOR_DATASET_ID = "ds-visitors"
_SALES_DATASET_ID = "ds-sales"
_MODEL_VISITORS = "namwon-nonlocal-visitors-next-month"
_MODEL_SALES = "namwon-observed-sales-next-month"


class _FakeDataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows


def _months(start: int, n: int) -> list[int]:
    out = [start]
    for _ in range(n - 1):
        out.append(add_month(out[-1], 1))
    return out


_ALL_MONTHS = _months(202101, 36)  # 202101~202312

_VISITOR_OVERRIDES = {202110: 1050, 202208: 1000, 202209: 1000, 202210: 1000, 202308: 900, 202309: 850, 202310: 800}
_SALES_OVERRIDES = {202110: 480_000, 202208: 500_000, 202209: 510_000, 202210: 520_000, 202308: 520_000, 202309: 530_000, 202310: 540_000}


def _visitor_rows(dong: str = _DONG) -> list[dict[str, Any]]:
    return [
        {
            "base_ym": ym,
            "dong_code": dong,
            "y": _VISITOR_OVERRIDES.get(ym, 1000),
            "local_visitors": 2000,
            "foreign_visitors": 5,
        }
        for ym in _ALL_MONTHS
    ]


def _sales_rows(dong: str = _DONG) -> list[dict[str, Any]]:
    return [
        {"base_ym": ym, "dong_code": dong, "y": _SALES_OVERRIDES.get(ym, 500_000), "observed_sales_krw": _SALES_OVERRIDES.get(ym, 500_000)}
        for ym in _ALL_MONTHS
    ]


def _dataset_summaries() -> list[dict[str, Any]]:
    return [
        {"dataset_id": _VISITOR_DATASET_ID, "spec": {"target": "nonlocal_visitors"}, "created_at": "2026-01-01T00:00:00Z", "observed_to": 202310},
        {"dataset_id": _SALES_DATASET_ID, "spec": {"target": "observed_sales_krw"}, "created_at": "2026-01-01T00:00:00Z", "observed_to": 202312},
    ]


def _patch_snapshots(monkeypatch: pytest.MonkeyPatch, *, visitor_rows: list[dict[str, Any]], sales_rows: list[dict[str, Any]], summaries: list[dict[str, Any]] | None = None) -> None:
    monkeypatch.setattr(analysis.snapshot_mod, "list_datasets", lambda: summaries if summaries is not None else _dataset_summaries())
    datasets = {_VISITOR_DATASET_ID: _FakeDataset(visitor_rows), _SALES_DATASET_ID: _FakeDataset(sales_rows)}
    monkeypatch.setattr(analysis.snapshot_mod, "load_dataset", lambda dataset_id: datasets[dataset_id])
    # forecast 기본은 model_required(활성 모델 없음) — forecast 전용 테스트에서 개별 오버라이드.
    monkeypatch.setattr(analysis.candidates_mod, "get_active", lambda model_id: None)


# ── empty / no snapshot ────────────────────────────────────────
def test_analyze_returns_empty_when_no_snapshots_exist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analysis.snapshot_mod, "list_datasets", lambda: [])
    result = analysis.analyze(region="namwon", dong_code=_DONG, base_ym=202310, model_id=None)
    assert result["status"] == "empty"
    assert "POST /realdata/datasets" in result["message"]


# ── current ─────────────────────────────────────────────────────
def test_current_reports_yoy_mom_and_empty_status(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_snapshots(monkeypatch, visitor_rows=_visitor_rows(), sales_rows=_sales_rows())
    result = analysis.analyze(region="namwon", dong_code=_DONG, base_ym=202310, model_id=None)

    current = result["data"]["current"]
    assert current["status"] == "ok"
    visitors = current["metrics"]["nonlocal_visitors"]
    assert visitors["status"] == "ok"
    assert visitors["value"] == 800
    assert visitors["yoy"]["previous"] == 1000  # 202209? no: yoy = base_ym-12 = 202210
    assert visitors["mom"]["previous"] == 850  # 202309


def test_current_is_empty_with_observed_range_when_base_ym_not_observed(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_snapshots(monkeypatch, visitor_rows=_visitor_rows(), sales_rows=_sales_rows())
    result = analysis.analyze(region="namwon", dong_code=_DONG, base_ym=202401, model_id=None)

    visitors = result["data"]["current"]["metrics"]["nonlocal_visitors"]
    assert visitors["status"] == "empty"
    assert visitors["observed_range"] == {"from": 202101, "to": 202312}


# ── diagnosis: rule 1 (yoy decline streak) + rule 3 (seasonal) ──
def test_diagnosis_finds_yoy_decline_streak_for_visitors_but_not_sales(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_snapshots(monkeypatch, visitor_rows=_visitor_rows(), sales_rows=_sales_rows())
    result = analysis.analyze(region="namwon", dong_code=_DONG, base_ym=202310, model_id=None)

    rule_ids = {f["rule_id"] for f in result["data"]["diagnosis"]["findings"]}
    assert "yoy_decline_streak_nonlocal_visitors" in rule_ids
    assert "yoy_decline_streak_observed_sales_krw" not in rule_ids  # 소비는 증가 추세로 설계됨

    finding = next(f for f in result["data"]["diagnosis"]["findings"] if f["rule_id"] == "yoy_decline_streak_nonlocal_visitors")
    assert finding["evidence"]["dong_code"] == _DONG
    assert finding["evidence"]["months"] == [202308, 202309, 202310]
    assert finding["evidence"]["values"] == [900, 850, 800]
    assert finding["evidence"]["baseline_values"] == [1000, 1000, 1000]
    assert finding["limitations"]


def test_diagnosis_finds_seasonal_deviation_for_visitors_but_not_sales(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_snapshots(monkeypatch, visitor_rows=_visitor_rows(), sales_rows=_sales_rows())
    result = analysis.analyze(region="namwon", dong_code=_DONG, base_ym=202310, model_id=None)

    rule_ids = {f["rule_id"] for f in result["data"]["diagnosis"]["findings"]}
    assert "seasonal_deviation_nonlocal_visitors" in rule_ids
    assert "seasonal_deviation_observed_sales_krw" not in rule_ids  # +8%는 20% 임계 미만


def test_diagnosis_finds_visit_sales_divergence(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_snapshots(monkeypatch, visitor_rows=_visitor_rows(), sales_rows=_sales_rows())
    result = analysis.analyze(region="namwon", dong_code=_DONG, base_ym=202310, model_id=None)

    finding = next((f for f in result["data"]["diagnosis"]["findings"] if f["rule_id"] == "visit_sales_divergence"), None)
    assert finding is not None
    assert finding["evidence"]["months"] == [202308, 202309, 202310]
    assert "correlation" in finding["evidence"]


def test_diagnosis_skips_rules_with_reason_when_observed_months_below_minimum(monkeypatch: pytest.MonkeyPatch) -> None:
    short_visitor_rows = [r for r in _visitor_rows() if r["base_ym"] >= 202401]  # 0개월(고의로 base_ym 밖) → <12개월
    _patch_snapshots(monkeypatch, visitor_rows=short_visitor_rows, sales_rows=_sales_rows())
    result = analysis.analyze(region="namwon", dong_code=_DONG, base_ym=202310, model_id=None)

    diagnosis = result["data"]["diagnosis"]
    visitor_findings = [f for f in diagnosis["findings"] if "nonlocal_visitors" in f["rule_id"]]
    assert visitor_findings == []
    reasons = {s["target"] for s in diagnosis["skipped"]}
    assert "nonlocal_visitors" in reasons
    assert "visit_sales_divergence" in reasons  # 방문 관측 부족 → 괴리 규칙도 건너뜀


# ── responses: 후보만, 효과 수치 없음 ───────────────────────────
def test_responses_empty_and_insufficient_evidence_when_no_findings(monkeypatch: pytest.MonkeyPatch) -> None:
    flat_visitor_rows = [{"base_ym": ym, "dong_code": _DONG, "y": 1000, "local_visitors": 2000, "foreign_visitors": 5} for ym in _ALL_MONTHS]
    flat_sales_rows = [{"base_ym": ym, "dong_code": _DONG, "y": 500_000, "observed_sales_krw": 500_000} for ym in _ALL_MONTHS]
    _patch_snapshots(monkeypatch, visitor_rows=flat_visitor_rows, sales_rows=flat_sales_rows)

    result = analysis.analyze(region="namwon", dong_code=_DONG, base_ym=202310, model_id=None)

    assert result["data"]["diagnosis"]["findings"] == []
    assert result["data"]["responses"] == {"status": "insufficient_evidence", "candidates": []}


def test_responses_built_from_findings_have_no_effect_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_snapshots(monkeypatch, visitor_rows=_visitor_rows(), sales_rows=_sales_rows())
    result = analysis.analyze(region="namwon", dong_code=_DONG, base_ym=202310, model_id=None)

    responses = result["data"]["responses"]
    assert responses["status"] == "ok"
    assert responses["candidates"]
    for candidate in responses["candidates"]:
        assert set(candidate) == {"rule_id", "category", "text", "based_on", "limitations"}


# ── forecast ─────────────────────────────────────────────────
def test_forecast_model_required_when_no_active_model(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_snapshots(monkeypatch, visitor_rows=_visitor_rows(), sales_rows=_sales_rows())
    result = analysis.analyze(region="namwon", dong_code=_DONG, base_ym=202310, model_id=_MODEL_VISITORS)

    forecast = result["data"]["forecast"]
    assert forecast["status"] == "model_required"
    assert forecast["models"][_MODEL_VISITORS]["status"] == "model_required"


def test_forecast_insufficient_data_when_no_feature_rows_for_dong(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_snapshots(monkeypatch, visitor_rows=_visitor_rows(), sales_rows=_sales_rows())
    monkeypatch.setattr(analysis.candidates_mod, "get_active", lambda model_id: {"version": "v1"})
    monkeypatch.setattr(analysis.candidates_mod, "get_candidate", lambda model_id, version: {"dataset_id": _VISITOR_DATASET_ID})
    monkeypatch.setattr(
        analysis.models_mod,
        "load_artifact",
        lambda model_id, version: {"forecast_month": 202311, "observed_end_month": 202310, "eval": {"metrics": {"mae": 1.0}}},
    )
    monkeypatch.setattr(analysis, "build_feature_rows", lambda dataset, for_month=None: [])

    result = analysis.analyze(region="namwon", dong_code=_DONG, base_ym=202310, model_id=_MODEL_VISITORS)

    forecast = result["data"]["forecast"]["models"][_MODEL_VISITORS]
    assert forecast["status"] == "insufficient_data"


def test_forecast_ok_returns_prediction_baseline_and_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_snapshots(monkeypatch, visitor_rows=_visitor_rows(), sales_rows=_sales_rows())
    monkeypatch.setattr(analysis.candidates_mod, "get_active", lambda model_id: {"version": "v1"})
    monkeypatch.setattr(analysis.candidates_mod, "get_candidate", lambda model_id, version: {"dataset_id": _VISITOR_DATASET_ID})
    monkeypatch.setattr(
        analysis.models_mod,
        "load_artifact",
        lambda model_id, version: {"forecast_month": 202311, "observed_end_month": 202310, "eval": {"metrics": {"mae": 12.5}, "baseline": {"mae": 20.0}}},
    )
    monkeypatch.setattr(
        analysis,
        "build_feature_rows",
        lambda dataset, for_month=None: [{"dong_code": _DONG, "y_lag1": 800.0}],
    )
    monkeypatch.setattr(analysis.models_mod, "predict", lambda artifact, rows: [777.0])

    result = analysis.analyze(region="namwon", dong_code=_DONG, base_ym=202310, model_id=_MODEL_VISITORS)

    forecast = result["data"]["forecast"]["models"][_MODEL_VISITORS]
    assert forecast["status"] == "ok"
    assert forecast["forecast_month"] == 202311
    assert forecast["prediction"] == 777.0
    assert forecast["baseline"]["value"] == 1000  # forecast_month(202311)-12개월 = 202211(오버라이드 없음, 기본값 1000)
    assert forecast["validation"]["metrics"]["mae"] == 12.5
