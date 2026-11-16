"""공표 통계 조회 라우트(관광지점 월별 입장객·생활인구 분기) — DB 대신 합성 행을 돌려주는 fetch_all로 검증.

데모(mock)·실데이터 표시 모두 같은 공표 값을 읽으므로 `/realdata/` 데모 차단 대상이 아니다."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api.v3 import public_stats as public_stats_module
from src.realdata.pg_reader import RealdataUnavailable

_SPOT_URL = "/api/v3/public-stats/spot-visitors"
_LIVING_URL = "/api/v3/public-stats/living-population"

_SPOT_ROWS = {
    "합성관광지": [{"base_ym": 202412, "visitors_total": 30}, {"base_ym": 202501, "visitors_total": 40}],
    "합성섬": [],
}


def _fake_fetch(calls: list[tuple[str, dict[str, Any]]]):
    def fetch(table: str, columns: list[str], *, where: dict[str, Any], order_by: list[str]) -> list[dict[str, Any]]:
        calls.append((table, where))
        if table == public_stats_module.SPOT_TABLE:
            return _SPOT_ROWS.get(where["spot_name"], [])
        return [
            {"base_year": 2025, "quarter": 1, "registered_population": "1.5", "stay_population": "4.0", "unit": "만 명"},
            {"base_year": 2025, "quarter": 2, "registered_population": "1.4", "stay_population": "5.5", "unit": "만 명"},
        ]

    return fetch


@pytest.mark.parametrize("url", [f"{_SPOT_URL}?spot=a", f"{_LIVING_URL}?region=a"])
def test_public_stats_require_auth(client: TestClient, url: str) -> None:
    assert client.get(url).status_code == 401


def test_spot_visitors_returns_monthly_series_per_spot(
    client: TestClient, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(public_stats_module, "fetch_all", _fake_fetch(calls))

    body = client.get(f"{_SPOT_URL}?spot=합성관광지&spot=합성섬&spot=합성관광지", headers=auth_headers).json()

    assert body["status"] == "ok"
    assert body["data"]["spots"] == [
        {"spot_name": "합성관광지", "points": [{"ym": "2024-12", "value": 30}, {"ym": "2025-01", "value": 40}]},
        {"spot_name": "합성섬", "points": []},
    ]
    assert [w["spot_name"] for _, w in calls] == ["합성관광지", "합성섬"]  # 중복 지점은 한 번만 읽는다
    assert body["provenance"]["dataset_id"] == "ds_14_kto_spot_monthly_visitors"


def test_spot_visitors_rejects_too_many_spots(client: TestClient, auth_headers: dict[str, str]) -> None:
    query = "&".join(f"spot=s{i}" for i in range(public_stats_module.MAX_SPOTS + 1))
    assert client.get(f"{_SPOT_URL}?{query}", headers=auth_headers).status_code == 422


def test_living_population_returns_quarters_in_published_unit(
    client: TestClient, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(public_stats_module, "fetch_all", _fake_fetch(calls))

    body = client.get(f"{_LIVING_URL}?region= 합성군 ", headers=auth_headers).json()

    assert body["status"] == "ok"
    assert body["data"] == {
        "region": "합성군",
        "unit": "만 명",
        "points": [
            {"year": 2025, "quarter": 1, "registered": 1.5, "stay": 4.0},
            {"year": 2025, "quarter": 2, "registered": 1.4, "stay": 5.5},
        ],
    }
    assert calls == [(public_stats_module.LIVING_TABLE, {"sigungu_name": "합성군"})]


def test_living_population_rejects_mixed_units(
    client: TestClient, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [{"base_year": 2025, "quarter": q, "registered_population": "1", "stay_population": "1", "unit": u}
            for q, u in ((1, "만 명"), (2, "명"))]
    monkeypatch.setattr(public_stats_module, "fetch_all", lambda *a, **k: rows)

    body = client.get(f"{_LIVING_URL}?region=합성군", headers=auth_headers).json()
    assert body["status"] == "error" and "단위" in body["message"]


@pytest.mark.parametrize("error", [RealdataUnavailable("XOPS_PG_DSN이 설정되지 않았습니다."), RuntimeError("relation does not exist")])
def test_public_stats_report_error_envelope_when_db_unavailable(
    client: TestClient, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def broken(*_a: Any, **_k: Any) -> list[dict[str, Any]]:
        raise error

    monkeypatch.setattr(public_stats_module, "fetch_all", broken)
    for url in (f"{_SPOT_URL}?spot=합성관광지", f"{_LIVING_URL}?region=합성군"):
        body = client.get(url, headers=auth_headers).json()
        assert body["status"] == "error" and body["data"] is None
        assert "relation" not in body["message"]  # DB 원문은 화면 응답에 싣지 않는다


@pytest.mark.parametrize("mode", ["mock", "real"])
def test_public_stats_answer_in_both_display_modes(
    client: TestClient, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(public_stats_module, "fetch_all", _fake_fetch(calls))
    headers = {**auth_headers, "X-Data-Mode": mode}
    assert client.get(f"{_LIVING_URL}?region=합성군", headers=headers).json()["status"] == "ok"
    assert client.get(f"{_SPOT_URL}?spot=합성관광지", headers=headers).json()["status"] == "ok"
