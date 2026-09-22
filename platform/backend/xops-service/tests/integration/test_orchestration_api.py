"""MLOps 오케스트레이션 API 통합 테스트."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

import pytest
from fastapi.testclient import TestClient

from src.core import db
from src.mlops.orchestration import registry as registry_module


def test_list_models(client: TestClient) -> None:
    """레지스트리 3종은 데모 표시 ON 경로에서 나온다(기본값은 시드 지표 모델 제외)."""
    models = client.get("/api/v3/orchestration/models", params={"include_seed": "true"}).json()
    ids = {m["model_id"] for m in models}
    assert {"population-forecast", "vital-population", "settlement-demand"} <= ids


def test_demo_off_hides_seed_models_and_pipelines(client: TestClient) -> None:
    """기본값(데모 OFF)에서는 시드 파이프라인 3종과 시드 지표 모델이 목록에 없다."""
    pipelines = client.get("/api/v3/orchestration/pipelines").json()
    assert {p["id"] for p in pipelines} & {
        "PL-POP-RETRAIN-01",
        "PL-VITAL-RETRAIN-02",
        "PL-SETTLE-RETRAIN-03",
    } == set()

    models = client.get("/api/v3/orchestration/models").json()
    assert all(m["metrics_source"] != "seed" for m in models)

    on_pipelines = client.get("/api/v3/orchestration/pipelines", params={"include_seed": "true"}).json()
    assert {"PL-POP-RETRAIN-01", "PL-VITAL-RETRAIN-02", "PL-SETTLE-RETRAIN-03"} <= {p["id"] for p in on_pipelines}


def test_demo_off_hides_runs_from_seed_pipelines(client: TestClient) -> None:
    """시드 파이프라인 실행은 행 자체는 실행 산출물이지만 학습 입력이 시드라 목록에서 뺀다."""
    client.post("/api/v3/orchestration/pipelines/PL-POP-RETRAIN-01/run", json={"trigger": "manual"})

    off = client.get("/api/v3/orchestration/runs").json()
    on = client.get("/api/v3/orchestration/runs", params={"include_seed": "true"}).json()

    assert all(r.get("pipeline_id") != "PL-POP-RETRAIN-01" for r in off)
    assert any(r.get("pipeline_id") == "PL-POP-RETRAIN-01" for r in on)


def test_unknown_model_404(client: TestClient) -> None:
    r = client.post("/api/v3/orchestration/events", json={"model_id": "ghost", "trigger": "manual"})
    assert r.status_code == 404


def test_manual_event_promotes(
    client: TestClient, reset_model: Callable[[str], None], await_run: Callable[..., dict]
) -> None:
    # 최초 재학습 상태로 고정 — 승급 후에는 아티팩트 실측치가 현행 기준이 되어 동점 반려된다.
    reset_model("population-forecast")
    accepted = client.post(
        "/api/v3/orchestration/events",
        json={"model_id": "population-forecast", "trigger": "manual", "candidate_latency_ms": 120},
    )
    assert accepted.status_code == 202
    r = await_run(client, accepted.json())
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


def test_high_latency_rolls_back(
    client: TestClient, reset_model: Callable[[str], None], await_run: Callable[..., dict]
) -> None:
    # 이 테스트의 대상은 지연 초과 롤백이다. 후보 지표를 명시해 승급 경로를 결정적으로 만들어
    # 학습 결과와 무관하게 지연 판정에 도달시킨다.
    reset_model("vital-population")
    r = await_run(
        client,
        client.post(
            "/api/v3/orchestration/events",
            json={
                "model_id": "vital-population",
                "trigger": "manual",
                "candidate_metrics": {"f1": 0.99},
                "candidate_latency_ms": 250,
            },
        ).json(),
    )
    assert r["state"] == "rolled_back"
    assert r["active_version"] == "v2.4"  # 직전 버전 유지
    assert r["deploy"]["rolled_back"] is True


def test_rejected_when_candidate_worse(client: TestClient, await_run: Callable[..., dict]) -> None:
    r = await_run(
        client,
        client.post(
            "/api/v3/orchestration/events",
            json={
                "model_id": "settlement-demand",
                "trigger": "manual",
                "candidate_metrics": {"f1": 0.10},
            },
        ).json(),
    )
    assert r["state"] == "rejected"


def test_runs_recorded(client: TestClient, await_run: Callable[..., dict]) -> None:
    await_run(
        client,
        client.post(
            "/api/v3/orchestration/events", json={"model_id": "population-forecast", "trigger": "manual"}
        ).json(),
    )
    # /events 발화는 파이프라인에 매이지 않은 실행이라 데모 ON 경로에서 조회한다.
    runs = client.get("/api/v3/orchestration/runs", params={"include_seed": "true"}).json()
    assert len(runs) >= 1
    assert "run_id" in runs[0]


# ── ML 파이프라인 등록 → 실행 → 실행 로그 (데모 OFF 경로) ──


def test_pipeline_register_run_logs_roundtrip(
    client: TestClient, reset_model: Callable[[str], None], await_run: Callable[..., dict]
) -> None:
    """등록한 파이프라인으로 실행하면 실행 레코드·단계·로그가 저장소에 남고 조회된다."""
    reset_model("settlement-demand")
    definition = {
        "id": "PL-TEST-RT-01",
        "name": "왕복 테스트 파이프라인",
        "model_id": "settlement-demand",
        "trigger_policy": "수동",
        "experiment": "EXP-RT-001",
    }
    client.delete(f"/api/v3/orchestration/pipelines/{definition['id']}")

    created = client.post("/api/v3/orchestration/pipelines", json=definition)
    assert created.status_code == 201
    assert created.json()["model_id"] == "settlement-demand"

    # 등록 목록에 현행/다음 후보 버전이 함께 실린다 (후보가 현행과 같아 실행이 잠기지 않도록).
    listed = {p["id"]: p for p in client.get("/api/v3/orchestration/pipelines").json()}
    assert definition["id"] in listed
    assert listed[definition["id"]]["candidate_version"] != listed[definition["id"]]["base_version"]

    # id 중복 등록 거부 · 없는 모델 거부
    assert client.post("/api/v3/orchestration/pipelines", json=definition).status_code == 409
    assert client.post(
        "/api/v3/orchestration/pipelines", json={**definition, "id": "PL-TEST-GHOST", "model_id": "ghost"}
    ).status_code == 404

    accepted = client.post(
        f"/api/v3/orchestration/pipelines/{definition['id']}/run", json={"trigger": "manual"}
    )
    assert accepted.status_code == 202
    run = await_run(client, accepted.json())
    assert run["pipeline_id"] == definition["id"]
    assert run["state"] == "succeeded"
    assert [s["stage"] for s in run["stages"]] == ["queued", "preparing", "training", "evaluating", "deploying"]
    assert run["started_at"] and run["finished_at"]

    # 실행 이력이 파이프라인별로 조회된다 (프로세스 메모리가 아니라 SQLite에서).
    history = client.get("/api/v3/orchestration/runs", params={"pipeline_id": definition["id"]}).json()
    assert [r["run_id"] for r in history] == [run["run_id"]]

    logs = client.get(f"/api/v3/orchestration/runs/{run['run_id']}/logs").json()
    assert logs["state"] == "succeeded"
    messages = [entry["message"] for entry in logs["logs"]]
    assert any("재학습 이벤트 접수" in m for m in messages)
    assert any("[training]" in m for m in messages)
    assert any("[evaluating]" in m for m in messages)
    assert any("[deploying]" in m for m in messages)

    assert client.get("/api/v3/orchestration/runs/RUN-NONE/logs").status_code == 404
    assert client.delete(f"/api/v3/orchestration/pipelines/{definition['id']}").status_code == 200
    assert client.post(f"/api/v3/orchestration/pipelines/{definition['id']}/run").status_code == 404
    # 파이프라인을 지워도 실행 이력은 남는다.
    assert client.get("/api/v3/orchestration/runs", params={"pipeline_id": definition["id"]}).json()


# ── 비동기 실행(202) ────────────────────────────────────────


def test_event_is_accepted_and_persisted_before_completion(
    client: TestClient, await_run: Callable[..., dict]
) -> None:
    """접수 즉시 진행 중 실행이 저장된다 — 프로세스가 중간에 죽어도 흔적이 남는다."""
    accepted = client.post(
        "/api/v3/orchestration/events", json={"model_id": "population-forecast", "trigger": "manual"}
    )
    assert accepted.status_code == 202
    run_id = accepted.json()["run_id"]
    assert db.get_run(run_id) is not None
    assert client.get(f"/api/v3/orchestration/runs/{run_id}").status_code == 200
    assert client.get("/api/v3/orchestration/runs/RUN-NONE").status_code == 404

    assert await_run(client, accepted.json())["run_id"] == run_id  # 뒤 테스트로 실행이 새지 않게 마감


def test_second_event_on_same_model_conflicts_while_running(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """모델별 락 — 실행이 끝나기 전 같은 모델을 다시 부르면 409."""
    started = threading.Event()
    release = threading.Event()

    def _slow_trigger(self, **kwargs):  # type: ignore[no-untyped-def]
        started.set()
        release.wait(10.0)
        return None

    monkeypatch.setattr(registry_module.ModelRegistry, "trigger", _slow_trigger)
    first = client.post(
        "/api/v3/orchestration/events", json={"model_id": "vital-population", "trigger": "manual"}
    )
    assert first.status_code == 202
    assert started.wait(10.0)

    second = client.post(
        "/api/v3/orchestration/events", json={"model_id": "vital-population", "trigger": "manual"}
    )
    assert second.status_code == 409

    release.set()
    registry_module.recover_stale_runs()  # 스텁이 남긴 진행 중 행 정리


def test_recover_stale_runs_closes_running_records() -> None:
    """기동 시 진행 중으로 남은 실행은 failed(restart)로 마감된다."""
    db.append_run({"run_id": "RUN-STALE-0001", "model_id": "population-forecast", "state": "running"})

    assert registry_module.recover_stale_runs() >= 1

    closed = db.get_run("RUN-STALE-0001")
    assert closed is not None
    assert closed["state"] == "failed"
    assert closed["reason"] == "restart"
