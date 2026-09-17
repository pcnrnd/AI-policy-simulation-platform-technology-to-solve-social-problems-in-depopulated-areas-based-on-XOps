"""GET /api/v3/dataops/catalog?live=true 통합 테스트.

데모 표시 OFF 화면이 "실제로 적재된 소스"만 고르려면 목록 응답이 등록 여부와 적재 여부를
구분해 줘야 한다. 이 파일은 그 계약 세 가지를 본다: 기본 off, live=true 정상, DSN 부재.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.dataops import liveness

_CATALOG = "/api/v3/dataops/catalog"


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    liveness.reset_cache()


def test_default_does_not_query_storage(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """live 파라미터가 없으면 저장소를 치지 않고 live_rows 는 비어 있다(기존 계약 유지)."""
    calls: list[str] = []
    monkeypatch.setattr(liveness, "_count", lambda schema: calls.append(schema["id"]) or 1)

    body = client.get(_CATALOG).json()

    assert calls == []
    assert body, "시드 카탈로그가 비어 있으면 안 된다"
    assert all(source["live_rows"] is None for source in body)


def test_live_true_annotates_row_counts(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """live=true 면 소스마다 실적재 행수가 붙는다."""
    monkeypatch.setattr(liveness, "_count", lambda schema: 1234)

    body = client.get(_CATALOG, params={"live": "true"}).json()

    assert body
    assert all(source["live_rows"] == 1234 for source in body)


def test_live_true_without_dsn_reports_null_not_zero(client: TestClient) -> None:
    """DSN이 설정되지 않은 환경에서는 0이 아니라 null 이다 — 미적재로 오인하면 안 된다.

    테스트 설정에는 XOPS_PG_DSN 등이 없으므로 실제 degrade 경로를 그대로 탄다.
    """
    body = client.get(_CATALOG, params={"live": "true"}).json()

    assert body
    assert all(source["live_rows"] is None for source in body)


def test_live_true_keeps_search_filter(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """검색(q)과 함께 써도 결과가 줄어들 뿐 주석은 그대로 붙는다."""
    monkeypatch.setattr(liveness, "_count", lambda schema: 5)

    body = client.get(_CATALOG, params={"live": "true", "q": "인구이동"}).json()

    assert body
    assert len(body) < len(client.get(_CATALOG).json())
    assert all(source["live_rows"] == 5 for source in body)


def test_annotation_does_not_leak_into_plain_listing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """live 조회 뒤 기본 조회를 해도 카탈로그 시드가 오염되지 않는다."""
    monkeypatch.setattr(liveness, "_count", lambda schema: 99)
    client.get(_CATALOG, params={"live": "true"})

    body = client.get(_CATALOG).json()

    assert all(source["live_rows"] is None for source in body)
