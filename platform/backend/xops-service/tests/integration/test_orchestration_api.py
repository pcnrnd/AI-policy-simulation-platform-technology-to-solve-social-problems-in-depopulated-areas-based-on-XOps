"""MLOps 오케스트레이션 API 통합 테스트."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_list_models(client: TestClient) -> None:
    models = client.get("/api/v3/orchestration/models").json()
    ids = {m["model_id"] for m in models}
    assert {"population-forecast", "living-population", "settlement-demand"} <= ids


def test_unknown_model_404(client: TestClient) -> None:
    r = client.post("/api/v3/orchestration/events", json={"model_id": "ghost", "trigger": "manual"})
    assert r.status_code == 404


def test_manual_event_promotes(client: TestClient) -> None:
    r = client.post(
        "/api/v3/orchestration/events",
        json={"model_id": "population-forecast", "trigger": "manual", "candidate_latency_ms": 120},
    ).json()
    assert r["state"] == "succeeded"
    assert r["evaluation"]["primary_metric"] == "f1"
    assert r["active_version"] == "v3.1"


def test_high_latency_rolls_back(client: TestClient) -> None:
    r = client.post(
        "/api/v3/orchestration/events",
        json={"model_id": "living-population", "trigger": "manual", "candidate_latency_ms": 250},
    ).json()
    assert r["state"] == "rolled_back"
    assert r["active_version"] == "v2.4"  # 직전 버전 유지


def test_rejected_when_candidate_worse(client: TestClient) -> None:
    r = client.post(
        "/api/v3/orchestration/events",
        json={
            "model_id": "settlement-demand",
            "trigger": "manual",
            "candidate_metrics": {"f1": 0.10},
        },
    ).json()
    assert r["state"] == "rejected"


def test_runs_recorded(client: TestClient) -> None:
    client.post("/api/v3/orchestration/events", json={"model_id": "population-forecast", "trigger": "manual"})
    runs = client.get("/api/v3/orchestration/runs").json()
    assert len(runs) >= 1
    assert "run_id" in runs[0]
