"""데이터셋 API 통합 테스트(R1-6, R6) — 봉투·인증·에러 status를 TestClient로 검증."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.core.settings import Settings
from src.realdata import pg_reader, snapshot

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "realdata"
_DATASETS_URL = "/api/v3/realdata/datasets"


def _load_fixture(name: str) -> list[dict[str, Any]]:
    return json.loads((_FIXTURE_DIR / name).read_text(encoding="utf-8"))


@pytest.fixture()
def isolated_dataset_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(realdata_dataset_dir=tmp_path / "datasets")
    monkeypatch.setattr(snapshot, "get_settings", lambda: settings)


def test_create_dataset_requires_auth(client: TestClient, isolated_dataset_dir: None) -> None:
    response = client.post(_DATASETS_URL, json={"target": "nonlocal_visitors"})
    assert response.status_code == 401


def test_create_dataset_ok(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    isolated_dataset_dir: None,
) -> None:
    fixture = _load_fixture("kt_monthly_visitors.json")
    monkeypatch.setattr(pg_reader, "fetch_all", lambda *a, **kw: fixture)

    response = client.post(_DATASETS_URL, json={"target": "nonlocal_visitors"}, headers=auth_headers)
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "ok"
    assert body["data"]["dataset_id"].startswith("ds-")
    assert body["data"]["row_count"] == len(fixture)
    assert "rows" not in body["data"]
    assert body["provenance"]["model_id"] == "namwon-nonlocal-visitors-next-month"


def test_create_dataset_mapping_error_is_status_error_not_http_error(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    isolated_dataset_dir: None,
) -> None:
    unmapped = _load_fixture("bc_dong_industry_sales_unmapped.json")
    monkeypatch.setattr(pg_reader, "fetch_aggregate", lambda *a, **kw: unmapped)

    response = client.post(_DATASETS_URL, json={"target": "observed_sales_krw"}, headers=auth_headers)
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "error"
    assert "가짜동" in body["message"]


def test_create_dataset_pg_unavailable_is_status_error(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    isolated_dataset_dir: None,
) -> None:
    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise pg_reader.RealdataUnavailable("PG 연결 실패(테스트)")

    monkeypatch.setattr(pg_reader, "fetch_all", _boom)

    response = client.post(_DATASETS_URL, json={"target": "nonlocal_visitors"}, headers=auth_headers)
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "error"
    assert "PG 연결 실패" in body["message"]


def test_get_dataset_not_found_is_404(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.get(f"{_DATASETS_URL}/ds-doesnotexist", headers=auth_headers)
    assert response.status_code == 404


def test_list_and_get_dataset_round_trip(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    isolated_dataset_dir: None,
) -> None:
    fixture = _load_fixture("kt_monthly_visitors.json")
    monkeypatch.setattr(pg_reader, "fetch_all", lambda *a, **kw: fixture)

    created = client.post(_DATASETS_URL, json={"target": "nonlocal_visitors"}, headers=auth_headers).json()
    dataset_id = created["data"]["dataset_id"]

    listed = client.get(_DATASETS_URL, headers=auth_headers).json()
    assert dataset_id in {d["dataset_id"] for d in listed["data"]["datasets"]}

    summary = client.get(f"{_DATASETS_URL}/{dataset_id}", headers=auth_headers).json()
    assert summary["status"] == "ok"
    assert "rows" not in summary["data"]

    with_rows = client.get(
        f"{_DATASETS_URL}/{dataset_id}", params={"include_rows": "true"}, headers=auth_headers
    ).json()
    assert len(with_rows["data"]["rows"]) == len(fixture)
