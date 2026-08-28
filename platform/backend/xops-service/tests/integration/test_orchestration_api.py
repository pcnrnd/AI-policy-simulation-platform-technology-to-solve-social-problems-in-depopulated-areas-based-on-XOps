"""MLOps 오케스트레이션 API 통합 테스트."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from fastapi.testclient import TestClient


def test_list_models(client: TestClient) -> None:
    models = client.get("/api/v3/orchestration/models").json()
    ids = {m["model_id"] for m in models}
    assert {"population-forecast", "vital-population", "settlement-demand"} <= ids


def test_unknown_model_404(client: TestClient) -> None:
    r = client.post("/api/v3/orchestration/events", json={"model_id": "ghost", "trigger": "manual"})
    assert r.status_code == 404


def test_manual_event_promotes(client: TestClient, reset_model: Callable[[str], None]) -> None:
    # 최초 재학습 상태로 고정 — 승급 후에는 아티팩트 실측치가 현행 기준이 되어 동점 반려된다.
    reset_model("population-forecast")
    r = client.post(
        "/api/v3/orchestration/events",
        json={"model_id": "population-forecast", "trigger": "manual", "candidate_latency_ms": 120},
    ).json()
    assert r["state"] == "succeeded"
    assert r["evaluation"]["primary_metric"] == "f1"
    assert r["active_version"] == "v3.1"
    # 후보 지표가 실제 학습에서 나왔음을 확인 — 파생 fallback이 아니다.
    assert r["training"]["source"] == "trained"
    assert r["training"]["dataset"]["rows"] > 0
    assert r["training"]["candidates_evaluated"] >= 2
    assert Path(r["artifact_path"]).is_file()
    # 기준선(절편-only)을 실제로 이겨서 승급했다.
    assert r["evaluation"]["candidate_value"] > r["evaluation"]["current_value"]
    assert r["candidate_metrics"]["f1"] == r["evaluation"]["candidate_value"]


def test_high_latency_rolls_back(client: TestClient, reset_model: Callable[[str], None]) -> None:
    # 이 테스트의 대상은 지연 초과 롤백이다. 후보 지표를 명시해 승급 경로를 결정적으로 만들어
    # 학습 결과와 무관하게 지연 판정에 도달시킨다.
    reset_model("vital-population")
    r = client.post(
        "/api/v3/orchestration/events",
        json={
            "model_id": "vital-population",
            "trigger": "manual",
            "candidate_metrics": {"f1": 0.99},
            "candidate_latency_ms": 250,
        },
    ).json()
    assert r["state"] == "rolled_back"
    assert r["active_version"] == "v2.4"  # 직전 버전 유지
    assert r["deploy"]["rolled_back"] is True


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
