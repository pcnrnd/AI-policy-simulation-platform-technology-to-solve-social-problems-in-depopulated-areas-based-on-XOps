"""데모(mock)·실데이터(real)·헤더 없음 × include_seed × 인증 유무 읽기 매트릭스 — 데모 기록 누출 0.

헤더가 없으면 실데이터로 본다. include_seed 는 mock 안에서만 의미가 있고, 인증 성공이 mock 조회 허용이
아니다. 목록·단건(ID 직접 접근)·로그·데이터·토큰 경로를 한 번에 확인한다.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from main import app

MOCK = {"X-Data-Mode": "mock"}
OFF_MODES = [{}, {"X-Data-Mode": "real"}]  # 헤더 없음, 실데이터 명시
SEED_SOURCE = "ds_01_resident_registry"


@pytest.fixture(scope="module")
def mock_assets() -> dict[str, Any]:
    """mock 모드로 학습데이터 등록 → 파이프라인 → 학습 1회를 만들고 그 ID들을 돌려준다."""
    client = TestClient(app, headers=MOCK)
    token = client.post("/api/v3/dataops/token").json()["access_token"]
    auth = {"Authorization": f"Bearer {token}"}
    source = next(row for row in client.get("/api/v3/modelops/available-datasets", headers=auth).json()["data"]
                  if row["id"] == "mock_namwon")
    dataset = client.post("/api/v3/modelops/datasets", headers=auth, json={
        "name": "매트릭스 확인용", "source_dataset_id": source["id"], "source_hash": source["content_hash"]}).json()
    pipeline = client.post("/api/v3/modelops/pipelines", headers=auth, json={
        "id": "matrix-namwon-sim", "name": "매트릭스 확인용", "model_type": "simulation",
        "dataset_id": dataset["id"], "dataset_hash": dataset["content_hash"]}).json()
    job = client.post(f"/api/v3/modelops/pipelines/{pipeline['id']}/run", headers=auth).json()
    deadline = time.monotonic() + 30
    while client.get(f"/api/v3/modelops/jobs/{job['id']}", headers=auth).json()["state"] not in ("saved", "failed"):
        assert time.monotonic() < deadline
        time.sleep(0.05)
    run = client.post("/api/v3/orchestration/pipelines/PL-POP-RETRAIN-01/run", headers=auth, json={"trigger": "manual"}).json()
    return {"auth": auth, "dataset": dataset, "pipeline": pipeline, "job": job, "run_id": run["run_id"]}


@pytest.mark.parametrize("off", OFF_MODES, ids=["headerless", "real"])
@pytest.mark.parametrize("with_auth", [False, True], ids=["anon", "auth"])
def test_list_reads_hide_demo_records(mock_assets: dict[str, Any], off: dict[str, str], with_auth: bool) -> None:
    client = TestClient(app)
    headers = {**off, **(mock_assets["auth"] if with_auth else {})}
    seed = {"include_seed": "true"}
    catalog = client.get("/api/v3/dataops/catalog", params=seed, headers=headers).json()
    assert not any(row.get("is_seed") for row in catalog)
    overview = client.get("/api/v3/overview/summary", params=seed, headers=headers).json()
    assert overview["source_kind"] != "mock" and not any(s["id"] == SEED_SOURCE for s in overview["sources"])
    assert (overview["model"] or {}).get("metrics_source") != "seed"
    for path in ("/api/v3/orchestration/models", "/api/v3/orchestration/pipelines", "/api/v3/orchestration/runs"):
        rows = client.get(path, params=seed, headers=headers).json()
        assert all(row.get("metrics_source") != "seed" and row.get("run_id") != mock_assets["run_id"]
                   and row.get("id") not in ("PL-POP-RETRAIN-01", "matrix-namwon-sim") for row in rows), path
    for path in ("/api/v3/monitoring/metrics", "/api/v3/monitoring/drift", "/api/v3/monitoring/explain"):
        for model_id in (None, "population-forecast"):
            params = {**seed, **({"model_id": model_id} if model_id else {})}
            assert client.get(path, params=params, headers=headers).json()["source"] is None, path
    if with_auth:
        assert client.get("/api/v3/dataops/apis", headers=headers).json() == [] or all(
            row.get("mode", "real") == "real" for row in client.get("/api/v3/dataops/apis", headers=headers).json())
        modelops = client.get("/api/v3/modelops/catalog", headers=headers).json()
        assert modelops["mode"] == "real"
        ids = {mock_assets["dataset"]["id"], mock_assets["pipeline"]["id"], mock_assets["job"]["id"]}
        for key in ("instances", "datasets", "versions", "pipelines", "jobs", "audit"):
            assert not ids & {row["id"] for row in modelops[key]}
        assert all(row["source"] == "real" for row in client.get("/api/v3/modelops/available-datasets", headers=headers).json()["data"])
        assert not ids & {row["id"] for row in client.get("/api/v3/modelops/datasets", headers=headers).json()["data"]}


@pytest.mark.parametrize("off", OFF_MODES, ids=["headerless", "real"])
def test_demo_ids_are_not_reachable_directly(mock_assets: dict[str, Any], off: dict[str, str]) -> None:
    client = TestClient(app)
    auth = {**off, **mock_assets["auth"]}
    job = mock_assets["job"]
    instance_id, version_id = job["instance_id"], job["version_id"]
    for method, path in (
        ("GET", f"/api/v3/dataops/catalog/{SEED_SOURCE}"),
        ("POST", f"/api/v3/dataops/token/{SEED_SOURCE}"),
        ("POST", f"/api/v3/dataops/oauth2/{SEED_SOURCE}"),
        ("GET", f"/api/v3/dataops/{SEED_SOURCE}"),
        ("GET", f"/api/v3/orchestration/runs/{mock_assets['run_id']}"),
        ("GET", f"/api/v3/orchestration/runs/{mock_assets['run_id']}/logs"),
        ("GET", f"/api/v3/modelops/jobs/{job['id']}"),
        ("GET", f"/api/v3/modelops/instances/{instance_id}/versions"),
        ("GET", f"/api/v3/modelops/instances/{instance_id}/versions/{version_id}/artifact"),
        ("GET", f"/api/v3/modelops/instances/{instance_id}/versions/{version_id}/inputs"),
    ):
        response = client.request(method, path, headers=auth)
        assert response.status_code == 404, (method, path, response.status_code)
    predict = client.post(f"/api/v3/modelops/instances/{instance_id}/versions/{version_id}/predict", headers=auth, json={
        "region_id": "namwon", "target": "population_change_pct", "dataset_id": mock_assets["dataset"]["id"],
        "dataset_hash": mock_assets["dataset"]["content_hash"]})
    assert predict.status_code == 404


def test_same_reads_are_visible_in_mock_mode(mock_assets: dict[str, Any]) -> None:
    """양성 대조 — 같은 기록은 mock 모드에서 보인다(차단이 기능 제거가 아님)."""
    client = TestClient(app, headers=MOCK)
    auth = mock_assets["auth"]
    assert client.get(f"/api/v3/dataops/catalog/{SEED_SOURCE}").status_code == 200
    assert client.get(f"/api/v3/dataops/{SEED_SOURCE}", headers=auth).status_code == 200
    assert client.get(f"/api/v3/orchestration/runs/{mock_assets['run_id']}").status_code == 200
    assert client.get(f"/api/v3/modelops/jobs/{mock_assets['job']['id']}", headers=auth).status_code == 200
    assert client.get("/api/v3/monitoring/metrics").json()["source"] in ("seed", "measured")
