"""타 부서 연동 권한 — 조회 전용 토큰 발급과 쓰기 경로 인증.

조회 전용 자격증명(`XOPS_READONLY_CLIENT_ID/SECRET`)으로 받은 토큰은 `data:read`만 담는다.
오케스트레이션 쓰기 4개와 드리프트 자동 재학습은 `data:write`를 요구한다(일반 조회는 그대로 공개).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from src.api import dependencies
from src.auth.jwt import decode_jwt
from src.core.settings import Settings

_RO_HEADERS = {"X-Client-Id": "dept-readonly", "X-Client-Secret": "ro-secret"}


@pytest.fixture()
def readonly_configured(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    settings = Settings(readonly_client_id="dept-readonly", readonly_client_secret="ro-secret")
    monkeypatch.setattr(dependencies, "get_settings", lambda: settings)
    yield


@pytest.fixture()
def readonly_headers(client: TestClient, readonly_configured: None) -> dict[str, str]:
    token = client.post("/api/v3/dataops/token", headers=_RO_HEADERS).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


# ── 발급 ───────────────────────────────────────────────────
@pytest.mark.parametrize(
    "path",
    [
        "/api/v3/dataops/token",
        "/api/v3/dataops/token/ds_08_admin_boundary",
    ],
)
def test_readonly_credentials_issue_read_scope_only(
    client: TestClient, readonly_configured: None, path: str
) -> None:
    body = client.post(path, headers=_RO_HEADERS).json()
    assert body["scope"] == "data:read"
    assert decode_jwt(body["access_token"])["scope"] == "data:read"


@pytest.mark.parametrize("path", ["/api/v3/dataops/oauth2", "/api/v3/dataops/oauth2/ds_08_admin_boundary"])
def test_readonly_credentials_oauth2_read_scope_only(
    client: TestClient, readonly_configured: None, path: str
) -> None:
    body = client.post(path, headers=_RO_HEADERS).json()
    assert body["scope"] == "data:read"
    assert decode_jwt(body["access_token"])["scope"] == "data:read"


def test_readonly_wrong_secret_rejected_even_in_dev(client: TestClient, readonly_configured: None) -> None:
    r = client.post(
        "/api/v3/dataops/token", headers={"X-Client-Id": "dept-readonly", "X-Client-Secret": "nope"}
    )
    assert r.status_code == 401


def test_default_issue_keeps_read_write_scope(client: TestClient, readonly_configured: None) -> None:
    # 조회 전용 자격증명이 아닌 기존 dev 발급은 계약 그대로(우리 UI가 쓴다).
    body = client.post("/api/v3/dataops/token").json()
    assert body["scope"] == "data:read data:write"


# ── 조회 전용 토큰의 권한 ─────────────────────────────────────
@pytest.mark.parametrize(
    "path",
    [
        "/api/v3/realdata/models",
        "/api/v3/realdata/datasets",
        "/api/v3/realdata/training-runs",
        "/api/v3/realdata/dong-map",
    ],
)
def test_readonly_token_reads_realdata(client: TestClient, readonly_headers: dict[str, str], path: str) -> None:
    assert client.get(path, headers=readonly_headers).status_code == 200


def test_readonly_token_cannot_start_training(client: TestClient, readonly_headers: dict[str, str]) -> None:
    r = client.post(
        "/api/v3/realdata/training-runs",
        headers=readonly_headers,
        json={"model_id": "namwon-sales", "dataset_id": "x"},
    )
    assert r.status_code == 401


# ── 오케스트레이션 쓰기 4개 ──────────────────────────────────
_WRITES = [
    ("POST", "/api/v3/orchestration/events", {"model_id": "population-forecast", "trigger": "manual"}),
    (
        "POST",
        "/api/v3/orchestration/pipelines",
        {"id": "PL-AUTH-TEST", "name": "권한 시험", "model_id": "population-forecast"},
    ),
    ("DELETE", "/api/v3/orchestration/pipelines/PL-POP-RETRAIN-01", None),
    ("POST", "/api/v3/orchestration/pipelines/PL-POP-RETRAIN-01/run", {"trigger": "manual"}),
]


@pytest.mark.parametrize(("method", "path", "body"), _WRITES)
def test_orchestration_writes_require_token(
    client: TestClient, method: str, path: str, body: dict[str, object] | None
) -> None:
    assert client.request(method, path, json=body).status_code == 401


@pytest.mark.parametrize(("method", "path", "body"), _WRITES)
def test_orchestration_writes_reject_readonly_token(
    client: TestClient,
    readonly_headers: dict[str, str],
    method: str,
    path: str,
    body: dict[str, object] | None,
) -> None:
    assert client.request(method, path, json=body, headers=readonly_headers).status_code == 401


def test_orchestration_reads_stay_public(client: TestClient) -> None:
    for path in ("/models", "/pipelines", "/runs"):
        assert client.get(f"/api/v3/orchestration{path}").status_code == 200


# ── 드리프트: 조회는 공개, 자동 재학습만 쓰기 권한 ───────────────
_DRIFT_RETRAIN = {"drifted": "true", "include_seed": "true", "model_id": "population-forecast"}


def test_drift_read_stays_public(client: TestClient) -> None:
    r = client.get("/api/v3/monitoring/drift", params={"drifted": "true", "include_seed": "true"})
    assert r.status_code == 200
    assert r.json()["drifted"] is True


def test_drift_auto_retrain_requires_write(client: TestClient, readonly_headers: dict[str, str]) -> None:
    params = {**_DRIFT_RETRAIN, "auto_retrain": "true"}
    assert client.get("/api/v3/monitoring/drift", params=params).status_code == 401
    assert client.get("/api/v3/monitoring/drift", params=params, headers=readonly_headers).status_code == 401


def test_drift_auto_retrain_with_write_token_fires(client: TestClient, auth_headers: dict[str, str]) -> None:
    params = {**_DRIFT_RETRAIN, "auto_retrain": "true"}
    r = client.get("/api/v3/monitoring/drift", params=params, headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["retrain"] is not None


def test_post_drift_auto_retrain_requires_write(client: TestClient, auth_headers: dict[str, str]) -> None:
    body = {"reference": [0.25, 0.25, 0.25, 0.25], "current": [0.7, 0.1, 0.1, 0.1]}
    params = {"model_id": "population-forecast", "auto_retrain": "true"}
    assert client.post("/api/v3/monitoring/drift", json=body).status_code == 200  # 판정만은 공개
    assert client.post("/api/v3/monitoring/drift", json=body, params=params).status_code == 401
    assert client.post("/api/v3/monitoring/drift", json=body, params=params, headers=auth_headers).status_code == 200
