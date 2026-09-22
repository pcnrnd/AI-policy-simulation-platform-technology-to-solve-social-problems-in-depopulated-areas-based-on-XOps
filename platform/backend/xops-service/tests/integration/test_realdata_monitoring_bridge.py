"""데모 표시 OFF에서 /monitoring·/orchestration GET이 실데이터(rd_*)를 같은 스키마로 내려주는지.

검증 축 3개:
1. 실데이터 모델 id·회귀 지표가 실려 오고 시드 식별자(population-forecast, accuracy 계열,
   시드 파이프라인 id)가 응답에 남지 않는다.
2. 실데이터는 `data:read` 토큰이 있는 호출에만 실린다 — 무인증 GET은 기존 빈 응답 계약 그대로다.
3. 데모 표시 ON(`include_seed=true`)은 시드 경로 그대로다(실데이터가 끼어들지 않는다).

작성자 A의 models/snapshot 모듈 실호출(PG·아티팩트 파일) 대신 `src.realdata.monitoring`의
`_import_*` 간접화를 스텁으로 대체한다 — 어댑터가 옮겨 담는 계약만 검증하는 것이 목적이다.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.core.db import _conn
from src.mlops.monitoring import realdata_bridge
from src.realdata import monitoring as rd_monitoring

_MODEL_ID = "namwon-nonlocal-visitors-next-month"
_VERSION_OLD = "v20260911-00ff34"
_VERSION_NEW = "v20260918-00ff34"
_DATASET_ID = "ds-00ff3407ad9f"
_SEED_MODEL_ID = "population-forecast"

_FEATURES = ["y_lag1", "y_lag2"]
_MEANS = [100.0, 90.0]
_STDS = [10.0, 9.0]
_COEF = [3.0, -1.0]
_INTERCEPT = 5.0
_BASE_YM = 202310


class FakeRealdataError(Exception):
    pass


class FakeErrors:
    RealdataError = FakeRealdataError
    InsufficientData = FakeRealdataError
    DatasetNotFound = FakeRealdataError


class FakeDataset:
    rows: list[dict[str, Any]] = []


class FakeSnapshot:
    def load_dataset(self, dataset_id: str) -> FakeDataset:
        return FakeDataset()


def _feature_row(dong_code: str, base_ym: int, y_lag1: float) -> dict[str, Any]:
    return {
        "dong_code": dong_code,
        "base_ym": base_ym,
        "y": y_lag1 * 1.1,
        "y_lag1": y_lag1,
        "y_lag2": y_lag1 * 0.9,
    }


class FakeModels:
    """검증 구간(202308~202310) 앞뒤로 분포가 다른 행을 만들어 PSI 비교 구간을 성립시킨다."""

    def load_artifact(self, model_id: str, version: str) -> dict[str, Any]:
        return {
            "model_id": model_id,
            "version": version,
            "dataset_id": _DATASET_ID,
            "features": _FEATURES,
            "coef": _COEF,
            "intercept": _INTERCEPT,
            "train_means": _MEANS,
            "train_stds": _STDS,
            "observed_end_month": 202310,
            "forecast_month": 202311,
        }

    def build_feature_rows(self, dataset: Any, *, for_month: int | None = None) -> list[dict[str, Any]]:
        if for_month is not None:
            return [_feature_row("45190250", for_month, 130.0), _feature_row("45190310", for_month, 140.0)]
        rows: list[dict[str, Any]] = []
        for month in (202305, 202306, 202307):
            for index, dong in enumerate(("45190250", "45190310")):
                rows.append(_feature_row(dong, month, 100.0 + index * 5))
        for month in (202308, 202309, 202310):
            for index, dong in enumerate(("45190250", "45190310")):
                rows.append(_feature_row(dong, month, 150.0 + index * 5))
        return rows

    def predict(self, artifact: dict[str, Any], rows: list[dict[str, Any]]) -> list[float]:
        return [
            _INTERCEPT + sum(_COEF[i] * (row[f] - _MEANS[i]) / _STDS[i] for i, f in enumerate(_FEATURES))
            for row in rows
        ]


@pytest.fixture(autouse=True)
def _realdata_fixtures(monkeypatch: pytest.MonkeyPatch):
    """rd_* 테이블에 후보 2건·학습 job 1건을 넣고, A 모듈 호출을 스텁으로 돌린다."""
    monkeypatch.setattr(rd_monitoring, "_import_models", lambda: FakeModels())
    monkeypatch.setattr(rd_monitoring, "_import_snapshot", lambda: FakeSnapshot())
    monkeypatch.setattr(rd_monitoring, "_import_errors", lambda: FakeErrors)

    conn = _conn()
    for table in ("rd_training_jobs", "rd_model_candidates", "rd_active_models"):
        conn.execute(f"DELETE FROM {table}")
    for version, mae in ((_VERSION_OLD, 8019.018987), (_VERSION_NEW, 7675.788289)):
        conn.execute(
            "INSERT INTO rd_model_candidates "
            "(model_id, version, dataset_id, artifact_path, metrics_json, baseline_json, status) "
            "VALUES (?, ?, ?, ?, ?, ?, 'candidate')",
            (
                _MODEL_ID,
                version,
                _DATASET_ID,
                f"/tmp/{version}.json",
                json.dumps({"mae": mae, "rmse": 14964.3, "wape": 0.12, "eval_period": {"from": 202308, "to": 202310, "n": 69}}),
                json.dumps({"name": "yoy_or_prev_month", "mae": 6921.123188, "rmse": 10443.4, "wape": 0.108}),
            ),
        )
    conn.execute(
        "INSERT INTO rd_training_jobs "
        "(job_id, model_id, dataset_id, state, requested_at, started_at, finished_at, candidate_version) "
        "VALUES ('job-20260918-f9b3c2', ?, ?, 'saved', '2026-09-18T06:30:28+00:00', "
        "'2026-09-18T06:30:28+00:00', '2026-09-18T06:30:29+00:00', ?)",
        (_MODEL_ID, _DATASET_ID, _VERSION_NEW),
    )
    conn.commit()
    yield
    for table in ("rd_training_jobs", "rd_model_candidates", "rd_active_models"):
        conn.execute(f"DELETE FROM {table}")
    conn.commit()


# ── /monitoring ─────────────────────────────────────────────
def test_metrics_demo_off_returns_realdata_regression_series(client: TestClient, auth_headers: dict[str, str]) -> None:
    body = client.get(
        "/api/v3/monitoring/metrics",
        params={"model_id": _MODEL_ID, "include_seed": "false"},
        headers=auth_headers,
    ).json()

    assert body["source"] == "realdata"
    assert body["model_id"] == _MODEL_ID
    assert body["metrics_kind"] == "regression"
    # 후보 2건이 학습 순서(버전 오름차순)로 쌓인다.
    assert body["labels"] == [_VERSION_OLD, _VERSION_NEW]
    assert body["history"]["mae"] == [8019.018987, 7675.788289]
    assert body["latest"]["baseline_mae"] == 6921.123188
    # 시드 식별자는 남지 않는다.
    serialized = json.dumps(body, ensure_ascii=False)
    assert "accuracy" not in serialized and _SEED_MODEL_ID not in serialized


def test_drift_demo_off_returns_realdata_distribution(client: TestClient, auth_headers: dict[str, str]) -> None:
    body = client.get(
        "/api/v3/monitoring/drift",
        params={"model_id": _MODEL_ID, "include_seed": "false"},
        headers=auth_headers,
    ).json()

    assert body["source"] == "realdata"
    assert body["feature"] == "y_lag1"
    assert len(body["buckets"]) == len(body["reference"]) == len(body["current"]) == 10
    assert isinstance(body["psi"], float)
    assert body["drifted"] is (body["psi"] >= body["psi_threshold"] or body["kl"] >= body["kl_threshold"])


def test_explain_demo_off_returns_linear_shap(client: TestClient, auth_headers: dict[str, str]) -> None:
    body = client.get(
        "/api/v3/monitoring/explain",
        params={"model_id": _MODEL_ID, "include_seed": "false"},
        headers=auth_headers,
    ).json()

    assert body["source"] == "realdata"
    assert body["base_ym"] == _BASE_YM
    assert [f["feature"] for f in body["features"]] == _FEATURES  # |phi| 내림차순
    assert all(isinstance(f["value"], float) for f in body["features"])


def test_monitoring_without_token_keeps_empty_contract(client: TestClient) -> None:
    """무인증 공개 GET은 실데이터를 싣지 않는다 — 기존 빈 응답 계약 그대로."""
    metrics = client.get("/api/v3/monitoring/metrics", params={"model_id": _MODEL_ID, "include_seed": "false"}).json()
    assert metrics["source"] is None and metrics["history"] == {}

    drift = client.get("/api/v3/monitoring/drift", params={"model_id": _MODEL_ID, "include_seed": "false"}).json()
    assert drift["source"] is None and drift["buckets"] == []


def test_monitoring_demo_on_stays_seed(client: TestClient, auth_headers: dict[str, str]) -> None:
    """데모 표시 ON은 토큰이 있어도 시드 경로 그대로다."""
    body = client.get(
        "/api/v3/monitoring/metrics",
        params={"model_id": _MODEL_ID, "include_seed": "true"},
        headers=auth_headers,
    ).json()
    assert body["source"] == "seed"
    assert "accuracy" in body["history"]


# ── /orchestration ──────────────────────────────────────────
def test_runs_demo_off_includes_training_job(client: TestClient, auth_headers: dict[str, str]) -> None:
    rows = client.get("/api/v3/orchestration/runs", params={"include_seed": "false"}, headers=auth_headers).json()

    realdata_rows = [r for r in rows if r.get("source") == "realdata"]
    assert len(realdata_rows) == 1
    run = realdata_rows[0]
    assert run["run_id"] == "job-20260918-f9b3c2"
    assert run["state"] == "succeeded"
    assert run["model_id"] == _MODEL_ID
    assert run["active_version"] == _VERSION_NEW
    assert run["candidate_metrics"]["mae"] == 7675.788289
    assert all(_SEED_MODEL_ID != r.get("model_id") for r in rows)


def test_runs_without_token_excludes_realdata(client: TestClient) -> None:
    rows = client.get("/api/v3/orchestration/runs", params={"include_seed": "false"}).json()
    assert [r for r in rows if r.get("source") == "realdata"] == []


def test_pipelines_demo_off_lists_realdata_entry(client: TestClient, auth_headers: dict[str, str]) -> None:
    rows = client.get("/api/v3/orchestration/pipelines", params={"include_seed": "false"}, headers=auth_headers).json()

    entry = next(r for r in rows if r["id"] == realdata_bridge.pipeline_id_for(_MODEL_ID))
    assert entry["name"] == "남원 방문객 재학습(실데이터)"
    assert entry["candidate_version"] == _VERSION_NEW
    assert entry["dataset_id"] == _DATASET_ID
    assert entry["source"] == "realdata"


def test_pipelines_fresh_install_uses_snapshot_dataset(client: TestClient, auth_headers: dict[str, str]) -> None:
    """학습 job·후보가 아직 없어도 스냅샷이 있으면 행이 나오고 dataset_id가 실린다.

    dataset_id가 비면 화면의 [실행]이 영구히 잠겨 신선 설치에서 첫 학습을 시작할 수 없다.
    """
    conn = _conn()
    for table in ("rd_training_jobs", "rd_model_candidates", "rd_active_models"):
        conn.execute(f"DELETE FROM {table}")
    conn.execute(
        "INSERT OR REPLACE INTO rd_datasets "
        "(dataset_id, spec_json, quality_json, observed_from, observed_to, row_count, content_hash, file_path, created_at) "
        # created_at을 멀리 잡아 다른 테스트가 남긴 스냅샷보다 최신이 되게 한다(최신 1건을 고르는지도 함께 본다).
        "VALUES (?, ?, '{}', 202301, 202310, 69, '00ff3407ad9f', '/tmp/ds.json', '2999-01-01T00:00:00+00:00')",
        (_DATASET_ID, json.dumps({"target": "nonlocal_visitors", "model_id": _MODEL_ID})),
    )
    conn.commit()
    try:
        rows = client.get(
            "/api/v3/orchestration/pipelines", params={"include_seed": "false"}, headers=auth_headers
        ).json()
        entry = next(r for r in rows if r["id"] == realdata_bridge.pipeline_id_for(_MODEL_ID))
        assert entry["dataset_id"] == _DATASET_ID
        assert entry["candidate_version"] is None
    finally:
        conn.execute("DELETE FROM rd_datasets WHERE dataset_id = ?", (_DATASET_ID,))
        conn.commit()


def test_models_demo_off_lists_realdata_models(client: TestClient, auth_headers: dict[str, str]) -> None:
    rows = client.get("/api/v3/orchestration/models", params={"include_seed": "false"}, headers=auth_headers).json()

    realdata_rows = [r for r in rows if r.get("metrics_source") == "realdata"]
    assert [r["model_id"] for r in realdata_rows] == [_MODEL_ID]
    assert realdata_rows[0]["name"] == "남원 타지역 방문객(익월)"
    assert realdata_rows[0]["version"] == _VERSION_NEW


def test_run_logs_for_training_job(client: TestClient, auth_headers: dict[str, str]) -> None:
    body = client.get("/api/v3/orchestration/runs/job-20260918-f9b3c2/logs", headers=auth_headers).json()

    assert body["state"] == "succeeded"
    messages = [line["message"] for line in body["logs"]]
    assert any("학습 요청 접수" in m for m in messages)
    assert any(_VERSION_NEW in m for m in messages)


def test_run_logs_unknown_id_still_404(client: TestClient, auth_headers: dict[str, str]) -> None:
    assert client.get("/api/v3/orchestration/runs/run-does-not-exist/logs", headers=auth_headers).status_code == 404
