from uuid import uuid4

from src.core import db
from src.dataops import summary
from src.mlops.monitoring import sources


def mode_headers(auth, mode):
    return {**auth, "X-Data-Mode": mode}


def test_catalog_modes_are_disjoint_and_override_legacy_include_seed(client, auth_headers):
    mock = client.get("/api/v3/dataops/catalog?include_seed=false", headers=mode_headers(auth_headers, "mock"))
    real = client.get("/api/v3/dataops/catalog?include_seed=true", headers=mode_headers(auth_headers, "real"))
    assert mock.headers["X-Data-Mode"] == "mock"
    assert all(row["is_seed"] for row in mock.json())
    assert all(not row["is_seed"] for row in real.json())
    assert not {row["id"] for row in mock.json()} & {row["id"] for row in real.json()}
    assert client.get("/api/v3/dataops/ds_11_kt_namwon_monthly_dong_visitors", headers=mode_headers(auth_headers, "mock")).status_code == 404
    assert client.get("/api/v3/dataops/ds_01_resident_registry", headers=mode_headers(auth_headers, "real")).status_code == 404
    # 헤더 없는 기존 클라이언트의 계약은 유지한다.
    all_rows = client.get("/api/v3/dataops/catalog?include_seed=true").json()
    assert {row["id"] for row in mock.json()} | {row["id"] for row in real.json()} == {row["id"] for row in all_rows}


def test_mock_does_not_touch_database_adapter_or_liveness(client, auth_headers, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("real database was accessed in mock mode")
    monkeypatch.setattr(summary, "live_rows_for", forbidden)
    overview = client.get("/api/v3/overview/summary", headers=mode_headers(auth_headers, "mock"))
    assert overview.status_code == 200
    assert overview.json()["source_kind"] == "mock"
    result = client.get("/api/v3/dataops/ds_01_resident_registry", headers=mode_headers(auth_headers, "mock"))
    assert result.status_code == 200
    assert result.json()["source_kind"] == "mock"
    assert "rows" not in result.json()


def test_real_cannot_read_mock_metrics_or_trigger_mock_training(client, auth_headers, monkeypatch):
    monkeypatch.setattr(sources, "measured_metrics", lambda _: {"history": {"f1": [0.99]}, "latest": {"f1": 0.99}})
    h = mode_headers(auth_headers, "real")
    result = client.get("/api/v3/monitoring/metrics?model_id=population-forecast&include_seed=true", headers=h)
    assert result.json()["history"] == {}
    assert result.json()["source"] is None
    assert client.post("/api/v3/orchestration/events", headers=h, json={"model_id": "population-forecast", "trigger": "manual"}).status_code == 400
    assert client.get("/api/v3/realdata/models", headers=mode_headers(auth_headers, "mock")).status_code == 404
    assert client.get("/api/v3/dataops/catalog", headers=mode_headers(auth_headers, "invalid")).status_code == 400


def test_user_sources_and_built_apis_are_bound_to_creation_mode(client, auth_headers):
    source_id = "phaseb_test_" + uuid4().hex[:8]
    mock, real = mode_headers(auth_headers, "mock"), mode_headers(auth_headers, "real")
    schema = {"id": source_id, "label": "학습 연계", "source": "RDB · PostgreSQL", "object": "tb_training", "description": "등록", "columns": [{"name": "id", "type": "int", "description": "키"}]}
    created = client.post("/api/v3/dataops/catalog", headers=mock, json=schema)
    assert created.status_code == 201
    assert client.get("/api/v3/dataops/catalog/" + source_id, headers=real).status_code == 404
    api_id = "phaseb_api_" + uuid4().hex[:8]
    body = {"id": api_id, "name": "등록 API", "source_id": source_id, "method": "GET"}
    registered = client.post("/api/v3/dataops/apis", headers=mock, json=body)
    assert registered.status_code == 201
    assert api_id in {row["id"] for row in client.get("/api/v3/dataops/apis", headers=mock).json()}
    assert api_id not in {row["id"] for row in client.get("/api/v3/dataops/apis", headers=real).json()}
    assert client.delete("/api/v3/dataops/apis/" + api_id, headers=real).status_code == 404
    assert client.delete("/api/v3/dataops/catalog/" + source_id, headers=real).status_code == 404
    assert client.delete("/api/v3/dataops/apis/" + api_id, headers=mock).status_code == 200
    assert client.delete("/api/v3/dataops/catalog/" + source_id, headers=mock).status_code == 200


def test_real_storage_unavailable_does_not_return_stub_rows(client, auth_headers, monkeypatch):
    from src.dataops.service import DataService
    from src.dataops.results import ExecutionResult
    monkeypatch.setattr(DataService, "_run", staticmethod(lambda **_: ExecutionResult.not_executed("unit unavailable")))
    response = client.get("/api/v3/dataops/ds_11_kt_namwon_monthly_dong_visitors", headers=mode_headers(auth_headers, "real"))
    assert response.status_code == 200
    assert response.json()["source_kind"] == "in-memory"
    assert response.json()["result_rows"] == 0
    assert response.json()["pagination"]["total"] is None


def test_demo_alert_detection_calculates_distribution_without_model_mutation(client, auth_headers, monkeypatch):
    from src.api.v3 import monitoring
    from src.core.seed import get_seed
    from src.mlops.monitoring.drift import DriftDetector

    def forbidden():
        raise AssertionError("읽기 전용 경보 판정이 모델 재학습/승급을 호출함")
    monkeypatch.setattr(monitoring, "get_registry", forbidden)
    params = {"drifted": "true", "include_seed": "true", "auto_retrain": "false"}
    detected = client.get("/api/v3/monitoring/drift", params=params, headers=mode_headers(auth_headers, "mock")).json()
    dist = get_seed()["drift_distribution"]
    expected = DriftDetector().detect(dist["reference"], dist["current_drifted"])
    assert detected["source"] == "seed" and detected["drifted"] is True
    assert detected["psi"] == expected.psi and detected["kl"] == expected.kl
    assert detected["retrain"] is None
    normal = client.get("/api/v3/monitoring/drift", params={**params, "drifted": "false"}, headers=mode_headers(auth_headers, "mock")).json()
    assert normal["drifted"] is False and normal["retrain"] is None
    real = client.get("/api/v3/monitoring/drift", params=params, headers=mode_headers(auth_headers, "real")).json()
    assert real["source"] is None and real["reference"] == [] and "psi" not in real
