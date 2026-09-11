"""R3/R4 통합 테스트 — TestClient로 라우트·인증·봉투 status·409/400/model_required를 검증한다.

작성자 A의 models/snapshot/errors 모듈이 없어 `src.realdata.jobs`/`src.realdata.candidates`의
`_import_*` 간접화 지점을 monkeypatch로 스텁 대체한다.
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.core.db import _conn
from src.realdata import candidates, jobs

_MODEL_ID = "namwon-nonlocal-visitors-next-month"
_UNKNOWN_MODEL_ID = "not-a-real-model"
_BASE = "/api/v3/realdata"


class FakeRealdataError(Exception):
    pass


class FakeInsufficientData(FakeRealdataError):
    pass


class FakeDatasetNotFound(FakeRealdataError):
    pass


class FakeErrors:
    RealdataError = FakeRealdataError
    InsufficientData = FakeInsufficientData
    DatasetNotFound = FakeDatasetNotFound


class FakeDataset:
    rows: list[dict[str, Any]] = []
    content_hash = "hash-x"


class FakeSnapshot:
    def load_dataset(self, dataset_id: str) -> FakeDataset:
        return FakeDataset()


class FakeOutcome:
    def __init__(self, version: str) -> None:
        self.model_id = _MODEL_ID
        self.version = version
        self.dataset_id = "ds-abc123def456"
        self.artifact_path = "/tmp/artifact.json"
        self.metrics = {"mae": 1.0, "rmse": 1.5, "wape": 0.1}
        self.baseline = {"name": "yoy", "mae": 2.0, "rmse": 2.5, "wape": 0.2}
        self.eval_period = {"from": 202307, "to": 202309, "n": 6}


_FEATURES = ["y_lag1", "y_lag2"]
_MEANS = {"y_lag1": 100.0, "y_lag2": 90.0}
_STDS = {"y_lag1": 10.0, "y_lag2": 9.0}
_COEF = [1.0, 2.0]
_INTERCEPT = 5.0


def _manual_predict(row: dict[str, float]) -> float:
    return _INTERCEPT + sum(_COEF[i] * (row[f] - _MEANS[f]) / _STDS[f] for i, f in enumerate(_FEATURES))


class FakeModels:
    def train(self, model_id: str, dataset_id: str, version: str) -> FakeOutcome:
        return FakeOutcome(version)

    def load_artifact(self, model_id: str, version: str) -> dict[str, Any]:
        return {
            "model_id": model_id,
            "version": version,
            "features": _FEATURES,
            "coef": _COEF,
            "intercept": _INTERCEPT,
            "train_means": _MEANS,
            "train_stds": _STDS,
        }

    def build_feature_rows(self, dataset: Any, features: list[str]) -> list[dict[str, float]]:
        return [{"y_lag1": 110.0, "y_lag2": 95.0}]

    def predict(self, artifact: dict[str, Any], row: dict[str, float]) -> float:
        return _manual_predict(row)


@pytest.fixture(autouse=True)
def _clean_tables():
    conn = _conn()
    for table in ("rd_training_jobs", "rd_model_candidates", "rd_active_models", "rd_datasets"):
        conn.execute(f"DELETE FROM {table}")
    conn.commit()
    jobs._model_locks.clear()
    yield
    for table in ("rd_training_jobs", "rd_model_candidates", "rd_active_models", "rd_datasets"):
        conn.execute(f"DELETE FROM {table}")
    conn.commit()
    jobs._model_locks.clear()


def _wait_terminal(client: TestClient, headers: dict[str, str], job_id: str, timeout: float = 2.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"{_BASE}/training-runs/{job_id}", headers=headers).json()
        if body["data"]["state"] in ("saved", "failed", "cancelled"):
            return body
        time.sleep(0.01)
    raise AssertionError("job이 시간 내 종결 상태에 도달하지 못했습니다.")


def test_training_run_requires_auth(client: TestClient) -> None:
    response = client.post(f"{_BASE}/training-runs", json={"model_id": _MODEL_ID, "dataset_id": "ds-abc123def456"})
    assert response.status_code == 401


def test_training_run_unknown_model_returns_404(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.post(
        f"{_BASE}/training-runs",
        json={"model_id": _UNKNOWN_MODEL_ID, "dataset_id": "ds-abc123def456"},
        headers=auth_headers,
    )
    assert response.status_code == 404


def test_training_run_success_flow(client: TestClient, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jobs, "_import_models", lambda: FakeModels())
    monkeypatch.setattr(jobs, "_import_snapshot", lambda: FakeSnapshot())
    monkeypatch.setattr(jobs, "_import_errors", lambda: FakeErrors)

    response = client.post(
        f"{_BASE}/training-runs",
        json={"model_id": _MODEL_ID, "dataset_id": "ds-abc123def456"},
        headers=auth_headers,
    )
    assert response.status_code == 202
    job_id = response.json()["data"]["job_id"]

    body = _wait_terminal(client, auth_headers, job_id)
    assert body["status"] == "ok"
    assert body["data"]["state"] == "saved"


def test_training_run_conflict_returns_409(client: TestClient, auth_headers: dict[str, str]) -> None:
    lock = jobs._lock_for(_MODEL_ID)
    assert lock.acquire(blocking=False)
    try:
        response = client.post(
            f"{_BASE}/training-runs",
            json={"model_id": _MODEL_ID, "dataset_id": "ds-abc123def456"},
            headers=auth_headers,
        )
        assert response.status_code == 409
    finally:
        lock.release()


def test_list_models_returns_two_fixed_models(client: TestClient, auth_headers: dict[str, str]) -> None:
    body = client.get(f"{_BASE}/models", headers=auth_headers).json()
    assert body["status"] == "ok"
    model_ids = {entry["model_id"] for entry in body["data"]}
    assert model_ids == {
        "namwon-nonlocal-visitors-next-month",
        "namwon-observed-sales-next-month",
    }
    assert all(entry["active_version"] is None for entry in body["data"])


def test_candidates_empty_when_none_registered(client: TestClient, auth_headers: dict[str, str]) -> None:
    body = client.get(f"{_BASE}/models/{_MODEL_ID}/candidates", headers=auth_headers).json()
    assert body["status"] == "empty"
    assert body["data"] == []


def _seed_candidate(*, mae: float = 5.0, baseline_mae: float = 10.0, quality: dict[str, Any] | None = None) -> None:
    conn = _conn()
    conn.execute(
        "INSERT INTO rd_datasets (dataset_id, spec_json, quality_json, observed_from, observed_to, row_count, content_hash, file_path, created_at) "
        "VALUES ('ds-abc123def456', '{}', ?, 202301, 202310, 10, 'hash-1', '/tmp/x.json', '2026-01-01T00:00:00+00:00')",
        (json.dumps(quality or {"mapping_matched": 23, "mapping_total": 23, "duplicate_keys": 0}),),
    )
    metrics = {"mae": mae, "rmse": mae * 1.2, "wape": 0.1, "eval_period": {"from": 202307, "to": 202309, "n": 6}}
    baseline = {"name": "yoy", "mae": baseline_mae, "rmse": baseline_mae * 1.2, "wape": 0.2}
    conn.execute(
        "INSERT INTO rd_model_candidates (model_id, version, dataset_id, artifact_path, metrics_json, baseline_json, status) "
        "VALUES (?, 'v1', 'ds-abc123def456', 'artifact-path', ?, ?, 'candidate')",
        (_MODEL_ID, json.dumps(metrics), json.dumps(baseline)),
    )
    conn.commit()


def test_apply_rejected_returns_400_with_reasons(client: TestClient, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_candidate(mae=15.0, baseline_mae=10.0)  # MAE가 기준선보다 나쁨 → 조건③ 실패
    monkeypatch.setattr(candidates, "_import_models", lambda: FakeModels())
    monkeypatch.setattr(candidates, "_import_snapshot", lambda: FakeSnapshot())
    monkeypatch.setattr(candidates, "_import_errors", lambda: FakeErrors)

    response = client.post(f"{_BASE}/models/{_MODEL_ID}/candidates/v1/apply", headers=auth_headers)

    assert response.status_code == 400
    assert response.json()["detail"]["reasons"]


def test_evaluation_returns_model_required_without_active_model(client: TestClient, auth_headers: dict[str, str]) -> None:
    body = client.get(f"{_BASE}/models/{_MODEL_ID}/evaluation", headers=auth_headers).json()
    assert body["status"] == "model_required"


def test_drift_returns_model_required_without_active_model(client: TestClient, auth_headers: dict[str, str]) -> None:
    body = client.get(f"{_BASE}/models/{_MODEL_ID}/drift", headers=auth_headers).json()
    assert body["status"] == "model_required"


def test_explain_requires_base_ym_and_dong_code_query_params(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.get(f"{_BASE}/models/{_MODEL_ID}/explain", headers=auth_headers)
    assert response.status_code == 422


def test_get_training_run_returns_404_when_missing(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.get(f"{_BASE}/training-runs/job-does-not-exist", headers=auth_headers)
    assert response.status_code == 404


def test_list_training_runs_empty_then_ok(client: TestClient, auth_headers: dict[str, str]) -> None:
    empty_body = client.get(f"{_BASE}/training-runs", params={"model_id": _MODEL_ID}, headers=auth_headers).json()
    assert empty_body["status"] == "empty"

    _conn().execute(
        "INSERT INTO rd_training_jobs (job_id, model_id, dataset_id, state, requested_at) "
        "VALUES ('job-x', ?, 'ds-abc123def456', 'saved', '2026-01-01T00:00:00+00:00')",
        (_MODEL_ID,),
    )
    _conn().commit()

    ok_body = client.get(f"{_BASE}/training-runs", params={"model_id": _MODEL_ID}, headers=auth_headers).json()
    assert ok_body["status"] == "ok"
    assert len(ok_body["data"]) == 1


def test_apply_unknown_candidate_returns_404(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.post(f"{_BASE}/models/{_MODEL_ID}/candidates/no-such-version/apply", headers=auth_headers)
    assert response.status_code == 404


def test_restore_unknown_candidate_returns_404(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.post(f"{_BASE}/models/{_MODEL_ID}/restore/no-such-version", headers=auth_headers)
    assert response.status_code == 404


def _activate_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    """apply()가 4조건을 전부 통과하도록 seed하고 실제로 반영해 활성 모델을 만든다."""
    _seed_candidate(mae=5.0, baseline_mae=10.0)
    monkeypatch.setattr(candidates, "_import_models", lambda: FakeModels())
    monkeypatch.setattr(candidates, "_import_snapshot", lambda: FakeSnapshot())
    monkeypatch.setattr(candidates, "_import_errors", lambda: FakeErrors)
    candidates.apply(_MODEL_ID, "v1", "tester")


def test_evaluation_explain_drift_ok_with_active_model(client: TestClient, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    _activate_candidate(monkeypatch)

    from src.realdata import monitoring

    monkeypatch.setattr(monitoring, "evaluation", lambda model_id, version: {"status": "ok", "message": None, "data": {"validation": {}, "operational": {"kind": "pending", "pending_months": [202311]}}})
    monkeypatch.setattr(monitoring, "explain", lambda model_id, version, base_ym, dong_code: {"status": "ok", "message": None, "data": {"base_value": 5.0, "contributions": [], "prediction": 5.0, "reconstruction_check": 0.0}})
    monkeypatch.setattr(monitoring, "drift", lambda model_id, version: {"status": "ok", "message": None, "data": {"status": "none", "kind": None, "features": [], "target": None}})

    eval_body = client.get(f"{_BASE}/models/{_MODEL_ID}/evaluation", headers=auth_headers).json()
    assert eval_body["status"] == "ok"
    assert eval_body["provenance"]["version"] == "v1"

    explain_body = client.get(
        f"{_BASE}/models/{_MODEL_ID}/explain", params={"base_ym": 202310, "dong_code": "45190250"}, headers=auth_headers
    ).json()
    assert explain_body["status"] == "ok"

    drift_body = client.get(f"{_BASE}/models/{_MODEL_ID}/drift", headers=auth_headers).json()
    assert drift_body["status"] == "ok"
    assert drift_body["data"]["status"] == "none"


def test_recover_stale_jobs_runs_on_startup_lifespan() -> None:
    """`with TestClient(app) as client:` — lifespan을 명시적으로 트리거해 재시작 마감을 검증한다.

    이 저장소의 기존 conftest.client 픽스처는 컨텍스트 매니저 없이 TestClient(app)을 쓰므로
    lifespan(startup 이벤트)이 발화하지 않는다 — 이 테스트는 그 경로를 별도로 검증한다."""
    conn = _conn()
    conn.execute(
        "INSERT INTO rd_training_jobs (job_id, model_id, dataset_id, state, requested_at) "
        "VALUES ('job-stale-lifespan', ?, 'ds-abc123def456', 'training', '2026-01-01T00:00:00+00:00')",
        (_MODEL_ID,),
    )
    conn.commit()

    from main import app

    with TestClient(app):
        pass

    job = jobs.get_job("job-stale-lifespan")
    assert job is not None
    assert job["state"] == "failed"
    assert job["error"] == "restart"
