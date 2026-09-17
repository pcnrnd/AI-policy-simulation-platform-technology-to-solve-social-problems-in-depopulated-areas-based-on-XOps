"""데모 표시 OFF 기준 DataOps e2e — 카탈로그(실적재) → 스키마 → 토큰 → API 빌드 → 호출.

데모 토글은 프런트 CSS 전용이라 백엔드에는 개념이 없다. 그래서 "데모 OFF 상태"란 곧
**실적재된 소스만 고르고, 그 소스로 API를 만들어 호출했을 때 실 저장소 행이 돌아오는가**
이다. 이 파일은 그 한 바퀴를 STEP ①→②→③ 순서 그대로 고정한다.

실 컨테이너 대신 어댑터를 주입해 CI에서도 돌아가게 하고, 실 DB 왕복은 별도 스모크로 남긴다
(`test_dataops_real_db.py` 와 같은 방침).
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.dataops import liveness as liveness_module
from src.dataops import service as service_module
from src.dataops.results import ExecutionResult

_CATALOG = "/api/v3/dataops/catalog"
# 실적재가 확인된 소스(pg ext_kt_namwon_monthly_dong_visitors, 실측 1,334행)
_LIVE_SOURCE = "ds_11_kt_namwon_monthly_dong_visitors"
# 등록만 되어 있고 적재되지 않은 것으로 취급할 소스
_EMPTY_SOURCE = "ds_05_smartfarm"

_REAL_ROWS = [
    {
        "base_ym": 201901,
        "sido_name": "전라북도",
        "sigungu_name": "남원시",
        "dong_name": "도통동",
        "local_visitors": "442036",
    },
    {
        "base_ym": 201901,
        "sido_name": "전라북도",
        "sigungu_name": "남원시",
        "dong_name": "죽항동",
        "local_visitors": "324726.5",
    },
]


class _StubAdapter:
    """실 저장소가 응답한 것처럼 행·총계를 돌려주는 어댑터."""

    name = "PostgreSQLAdapter"

    def __init__(self, rows: list[dict[str, Any]], total: int) -> None:
        self._result = ExecutionResult(rows=rows, total=total, affected_rows=None, executed=True)
        self.requests: list[Any] = []

    def execute(self, request: Any) -> ExecutionResult:
        self.requests.append(request)
        return self._result


@pytest.fixture(autouse=True)
def _clear_live_cache() -> None:
    liveness_module.reset_cache()


@pytest.fixture()
def live_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """실적재 1건 · 미적재 1건 · 나머지 확인 불가 — 데모 OFF 목록의 세 가지 상태를 모두 만든다."""
    counts = {_LIVE_SOURCE: 1334, _EMPTY_SOURCE: 0}
    monkeypatch.setattr(liveness_module, "_count", lambda schema: counts.get(schema["id"]))


def test_demo_off_full_round_trip(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, live_counts: None
) -> None:
    """STEP ① 실적재 소스 선택 → ② 스키마 → ③ 토큰·API 호출까지 한 바퀴."""
    # ── STEP ① 카탈로그: 실적재 행수를 함께 받는다 ──
    catalog = client.get(_CATALOG, params={"live": "true"}).json()
    by_id = {s["id"]: s for s in catalog}

    assert by_id[_LIVE_SOURCE]["live_rows"] == 1334, "적재된 소스는 실제 행수가 실려야 한다"
    assert by_id[_EMPTY_SOURCE]["live_rows"] == 0, "미적재 소스는 0 — 프런트가 목록에서 제외한다"
    assert by_id["ds_09_welfare_facility"]["live_rows"] is None, "확인 불가는 0과 구분된다"

    # 프런트가 데모 OFF에서 실제로 고르는 목록 = live_rows != 0
    selectable = [s["id"] for s in catalog if s["live_rows"] != 0]
    assert _LIVE_SOURCE in selectable and _EMPTY_SOURCE not in selectable

    # ── STEP ② 스키마: 선택한 소스의 컬럼 정의가 그대로 온다 ──
    schema = client.get(f"{_CATALOG}/{_LIVE_SOURCE}").json()
    assert schema["object"] == "ext_kt_namwon_monthly_dong_visitors"
    columns = [c["name"] for c in schema["columns"]]
    assert columns, "스키마 표가 그릴 컬럼이 있어야 한다"
    # 실제 pg information_schema 와 같은 이름·순서(감사 실측으로 확인된 선두 컬럼)
    assert columns[:3] == ["base_ym", "sido_name", "sido_code"]

    # ── STEP ③-1 토큰 발급 ──
    token_body = client.post(f"/api/v3/dataops/token/{_LIVE_SOURCE}").json()
    assert token_body["token_type"] == "Bearer"
    assert "data:read" in token_body["scope"]
    headers = {"Authorization": f"Bearer {token_body['access_token']}"}

    # ── STEP ③-2 API 호출: 실 저장소 행이 돌아온다 ──
    adapter = _StubAdapter(_REAL_ROWS, total=1334)
    monkeypatch.setattr(service_module, "get_adapter", lambda s: adapter)

    response = client.get(f"/api/v3/dataops/{_LIVE_SOURCE}", params={"page_size": 2}, headers=headers)
    body = response.json()

    assert response.status_code == 200
    assert body["source_kind"] == "database", "실행된 조회는 database 로 표기된다"
    assert body["rows"] == _REAL_ROWS
    assert body["pagination"]["total"] == 1334
    # 화면에 그대로 뜨는 생성 SQL이 실제 테이블을 가리킨다
    assert "ext_kt_namwon_monthly_dong_visitors" in body["generated_query"]
    assert adapter.requests, "어댑터가 실제로 호출되어야 한다"


def test_unauthenticated_call_is_rejected(client: TestClient, live_counts: None) -> None:
    """토큰 없이 같은 경로를 부르면 막힌다 — STEP ③ 인증 단계가 장식이 아니다."""
    assert client.get(f"/api/v3/dataops/{_LIVE_SOURCE}").status_code == 401


def test_overview_total_matches_counted_sources(
    client: TestClient, live_counts: None
) -> None:
    """같은 실측 출처를 Overview 롤업도 쓴다 — 카탈로그와 합계가 어긋나지 않는다."""
    summary = client.get("/api/v3/overview/summary").json()

    assert summary["archive_rows_total"] == 1334, "센 소스(1334 + 0)만 합산된다"
    assert summary["archive_rows_counted"] == 2
    assert summary["archive_rows_unknown"] == summary["source_count"] - 2
