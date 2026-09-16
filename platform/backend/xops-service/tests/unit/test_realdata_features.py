"""피처 엔지니어링 단위 테스트 — 누출 차단·랙 결손 제외·forecast 표본 완전성(R2-1, R2-1a, R2-2)."""

from __future__ import annotations

from typing import Any

from src.realdata import features
from src.realdata.snapshot import DatasetRecord

_DONG_A = "45190250"
_DONG_B = "45190310"


def _mk_dataset(target: str, rows: list[dict[str, Any]]) -> DatasetRecord:
    return DatasetRecord(
        dataset_id="ds-test000000",
        target=target,
        model_id="test-model",
        spec={},
        quality={},
        observed_from=min((r["base_ym"] for r in rows), default=0),
        observed_to=max((r["base_ym"] for r in rows), default=0),
        row_count=len(rows),
        excluded_rows={},
        content_hash="deadbeef",
        file_path="/tmp/x.json",
        created_at="2026-01-01T00:00:00+00:00",
        rows=rows,
    )


def _visitor_rows(dong: str, months: list[int], base_value: int = 100) -> list[dict[str, Any]]:
    return [
        {"base_ym": m, "dong_code": dong, "y": base_value + i, "local_visitors": 1, "foreign_visitors": 1}
        for i, m in enumerate(months)
    ]


def test_add_month_wraps_year_boundary() -> None:
    assert features.add_month(202312, 1) == 202401
    assert features.add_month(202301, -1) == 202212


def test_build_feature_rows_requires_three_continuous_lags() -> None:
    # 202201~202206 연속 6개월 — 랙 3개가 모두 있는 표본은 202204~202206만 만들어진다.
    rows = _visitor_rows(_DONG_A, [202201, 202202, 202203, 202204, 202205, 202206])
    dataset = _mk_dataset("nonlocal_visitors", rows)

    feature_rows = features.build_feature_rows(dataset)

    assert {r["base_ym"] for r in feature_rows} == {202204, 202205, 202206}
    assert features.excluded_rows_summary()["lag_missing"] == 3


def test_build_feature_rows_excludes_samples_around_gap() -> None:
    # 202205가 원천에 없음 — 랙이 그 결손을 물고 있는 202206~202208은 제외되고,
    # 결손 이전(202204)과 3개월치 랙이 다시 온전해지는 202209는 살아남는다.
    rows = _visitor_rows(_DONG_A, [202201, 202202, 202203, 202204, 202206, 202207, 202208, 202209])
    dataset = _mk_dataset("nonlocal_visitors", rows)

    feature_rows = features.build_feature_rows(dataset)

    produced_months = {r["base_ym"] for r in feature_rows}
    assert produced_months == {202204, 202209}


def test_future_values_do_not_change_past_feature_rows() -> None:
    """R2-2 누출 테스트: 미래 값을 바꿔도 과거 표본의 피처는 불변."""
    months = [202201, 202202, 202203, 202204, 202205]
    rows = _visitor_rows(_DONG_A, months)
    dataset = _mk_dataset("nonlocal_visitors", rows)
    baseline_rows = features.build_feature_rows(dataset)
    baseline_204 = next(r for r in baseline_rows if r["base_ym"] == 202204)

    mutated_rows = [dict(r) for r in rows]
    for r in mutated_rows:
        if r["base_ym"] == 202205:
            r["y"] = 999999
    mutated_dataset = _mk_dataset("nonlocal_visitors", mutated_rows)
    mutated_feature_rows = features.build_feature_rows(mutated_dataset)
    mutated_204 = next(r for r in mutated_feature_rows if r["base_ym"] == 202204)

    assert baseline_204 == mutated_204


def test_yoy_missing_sets_has_yoy_zero_not_excluded() -> None:
    months = [202201, 202202, 202203, 202204]
    rows = _visitor_rows(_DONG_A, months)
    dataset = _mk_dataset("nonlocal_visitors", rows)

    feature_rows = features.build_feature_rows(dataset)
    row_204 = next(r for r in feature_rows if r["base_ym"] == 202204)

    assert row_204["has_yoy"] == 0
    # v0.3: 0 센티널 대신 같은 행의 y_lag1로 채운다. 결측 사실은 has_yoy가 계속 구분한다.
    assert row_204["y_yoy"] == row_204["y_lag1"]
    assert row_204["y_yoy"] != 0


def test_yoy_present_is_not_replaced_by_lag1() -> None:
    months = [202201, 202202, 202203, 202204, 202301, 202302, 202303, 202304]
    rows = _visitor_rows(_DONG_A, months)
    dataset = _mk_dataset("nonlocal_visitors", rows)

    feature_rows = features.build_feature_rows(dataset)
    row_304 = next(r for r in feature_rows if r["base_ym"] == 202304)
    expected_yoy = next(r["y"] for r in rows if r["base_ym"] == 202204)

    assert row_304["has_yoy"] == 1
    assert row_304["y_yoy"] == expected_yoy


def test_dong_one_hot_covers_all_23_canonical_codes() -> None:
    rows = _visitor_rows(_DONG_A, [202201, 202202, 202203, 202204])
    dataset = _mk_dataset("nonlocal_visitors", rows)

    names = features.feature_names(dataset)
    dong_columns = [n for n in names if n.startswith("dong_")]
    assert len(dong_columns) == 23

    feature_rows = features.build_feature_rows(dataset)
    row = feature_rows[0]
    assert sum(row[c] for c in dong_columns) == 1
    assert row[f"dong_{_DONG_A}"] == 1


def test_for_month_builds_forecast_rows_without_leakage() -> None:
    months = [202201, 202202, 202203, 202204]
    rows = _visitor_rows(_DONG_A, months) + _visitor_rows(_DONG_B, months, base_value=500)
    dataset = _mk_dataset("nonlocal_visitors", rows)

    forecast_rows = features.build_feature_rows(dataset, for_month=202205)

    assert len(forecast_rows) == 2  # 두 동만 랙 3개월(202202~202204)이 온전함
    for row in forecast_rows:
        assert row["base_ym"] == 202205
        assert row["y"] is None


def test_sales_target_has_no_cross_feature() -> None:
    """R2-1(v0.2): 소비 모델도 방문 교차 피처(visitors_lag1) 없이 자기 타깃 랙만 쓴다."""
    rows = [
        {"base_ym": m, "dong_code": _DONG_A, "y": 100 + i, "observed_sales_krw": 100 + i}
        for i, m in enumerate([202201, 202202, 202203, 202204])
    ]
    dataset = _mk_dataset("observed_sales_krw", rows)

    names = features.feature_names(dataset)
    assert "visitors_lag1" not in names

    feature_rows = features.build_feature_rows(dataset)
    assert feature_rows
    assert all("visitors_lag1" not in r for r in feature_rows)


def test_feature_names_identical_for_both_targets() -> None:
    """R2-1(v0.2): 두 모델은 랙·전년동월·계절·동 one-hot만 쓰는 동일 피처 구성이다."""
    rows = [{"base_ym": 202201, "dong_code": _DONG_A, "y": 1, "observed_sales_krw": 1}]
    visitors_names = features.feature_names(_mk_dataset("nonlocal_visitors", rows))
    sales_names = features.feature_names(_mk_dataset("observed_sales_krw", rows))
    assert visitors_names == sales_names


def test_for_month_forecast_covers_all_observed_dongs_r2_1a() -> None:
    """R2-1a: forecast_month에 대해 관측이 있는 동 전부(23)의 피처 행이 나와야 한다."""
    canonical_dongs = features._canonical_dongs()
    assert len(canonical_dongs) == 23

    months = [202201, 202202, 202203, 202204]
    rows = [row for dong in canonical_dongs for row in _visitor_rows(dong, months)]

    for target in ("nonlocal_visitors", "observed_sales_krw"):
        dataset = _mk_dataset(target, rows)
        forecast_rows = features.build_feature_rows(dataset, for_month=202205)
        assert {r["dong_code"] for r in forecast_rows} == set(canonical_dongs)
        assert len(forecast_rows) == 23
