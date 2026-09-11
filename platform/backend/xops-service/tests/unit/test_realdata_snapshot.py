"""스냅샷 생성 단위 테스트 — 재현성·매핑 실패·0 채움 없음·업서트(R1-6, R1-7)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.core.settings import Settings
from src.realdata import pg_reader, snapshot
from src.realdata.errors import DatasetNotFound, MappingError

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "realdata"


def _load_fixture(name: str) -> list[dict[str, Any]]:
    return json.loads((_FIXTURE_DIR / name).read_text(encoding="utf-8"))


@pytest.fixture()
def isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    settings = Settings(realdata_dataset_dir=tmp_path / "datasets")
    monkeypatch.setattr(snapshot, "get_settings", lambda: settings)
    return settings


def test_create_dataset_visitors_is_reproducible(
    monkeypatch: pytest.MonkeyPatch, isolated_settings: Settings
) -> None:
    fixture = _load_fixture("kt_monthly_visitors.json")
    monkeypatch.setattr(pg_reader, "fetch_all", lambda *a, **kw: fixture)

    first = snapshot.create_dataset("nonlocal_visitors")
    second = snapshot.create_dataset("nonlocal_visitors")

    assert first.dataset_id == second.dataset_id
    assert first.content_hash == second.content_hash
    assert first.dataset_id.startswith("ds-")
    assert first.row_count == len(fixture) == 48
    assert first.model_id == "namwon-nonlocal-visitors-next-month"
    assert first.observed_from == 202201
    assert first.observed_to == 202304


def test_create_dataset_visitors_quality_and_no_zero_fill(
    monkeypatch: pytest.MonkeyPatch, isolated_settings: Settings
) -> None:
    fixture = _load_fixture("kt_monthly_visitors.json")
    monkeypatch.setattr(pg_reader, "fetch_all", lambda *a, **kw: fixture)

    record = snapshot.create_dataset("nonlocal_visitors")

    assert record.quality["month_count"] == 16
    assert record.quality["dong_count"] == 3
    assert record.quality["missing_month_dong_count"] == 16 * 23 - 48
    assert record.quality["negative_count"] == 0
    assert record.quality["duplicate_key_count"] == 0
    # 0 채움 없음: 관측된 (월,동) 조합 수만큼만 rows가 존재한다.
    assert len(record.rows) == 48
    assert {(r["base_ym"], r["dong_code"]) for r in record.rows} == {
        (r["base_ym"], r["dong_code"]) for r in fixture
    }


def test_create_dataset_sales_rejects_unmapped_dong_name(
    monkeypatch: pytest.MonkeyPatch, isolated_settings: Settings
) -> None:
    unmapped_fixture = _load_fixture("bc_dong_industry_sales_unmapped.json")
    monkeypatch.setattr(pg_reader, "fetch_aggregate", lambda *a, **kw: unmapped_fixture)

    with pytest.raises(MappingError) as exc_info:
        snapshot.create_dataset("observed_sales_krw")

    assert exc_info.value.unmapped == ["가짜동"]


def test_create_dataset_sales_no_zero_fill_across_gap(
    monkeypatch: pytest.MonkeyPatch, isolated_settings: Settings
) -> None:
    bc_fixture = _load_fixture("bc_dong_industry_sales.json")
    monkeypatch.setattr(pg_reader, "fetch_aggregate", lambda *a, **kw: bc_fixture)

    record = snapshot.create_dataset("observed_sales_krw")

    assert record.row_count == len(bc_fixture) == 42
    assert record.quality["month_count"] == 14
    # 202209·202210은 원천에 없다 — 채워 넣지 않는다.
    assert 202209 not in {r["base_ym"] for r in record.rows}
    assert 202210 not in {r["base_ym"] for r in record.rows}
    # R2-1(v0.2): 소비 스냅샷은 소비 원천만 반영 — 방문객 교차 컬럼이 없다.
    sample = next(r for r in record.rows if r["base_ym"] == 202201 and r["dong_code"] == "45190250")
    assert "nonlocal_visitors" not in sample


def test_load_dataset_round_trips(monkeypatch: pytest.MonkeyPatch, isolated_settings: Settings) -> None:
    fixture = _load_fixture("kt_monthly_visitors.json")
    monkeypatch.setattr(pg_reader, "fetch_all", lambda *a, **kw: fixture)

    created = snapshot.create_dataset("nonlocal_visitors")
    loaded = snapshot.load_dataset(created.dataset_id)

    assert loaded.dataset_id == created.dataset_id
    assert loaded.row_count == created.row_count
    assert loaded.rows == created.rows


def test_load_dataset_raises_when_missing() -> None:
    with pytest.raises(DatasetNotFound):
        snapshot.load_dataset("ds-doesnotexist")


def test_upsert_keeps_created_at_on_regeneration(
    monkeypatch: pytest.MonkeyPatch, isolated_settings: Settings
) -> None:
    fixture = _load_fixture("kt_monthly_visitors.json")
    monkeypatch.setattr(pg_reader, "fetch_all", lambda *a, **kw: fixture)

    first = snapshot.create_dataset("nonlocal_visitors")
    second = snapshot.create_dataset("nonlocal_visitors")

    assert first.created_at == second.created_at
