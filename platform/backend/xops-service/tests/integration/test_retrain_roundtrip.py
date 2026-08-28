"""재학습 → 실제 학습 → 승급 → 배포 왕복 통합 테스트.

결정적 후보 파생을 실측 학습으로 대체한 뒤의 계약을 고정한다:
승급은 **보장되지 않고** 같은 평가 프로토콜로 기준선/직전 아티팩트를 이겨야 성립한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from fastapi.testclient import TestClient

from src.core.settings import get_settings
from src.mlops.orchestration import orchestrator as orchestrator_module
from src.mlops.training import trainer as trainer_module

_MODEL = "population-forecast"
_EVENTS = "/api/v3/orchestration/events"


def _trigger(client: TestClient, **payload: Any) -> dict[str, Any]:
    body = {"model_id": _MODEL, "trigger": "manual", **payload}
    response = client.post(_EVENTS, json=body)
    assert response.status_code == 200, response.text
    result: dict[str, Any] = response.json()
    return result


def _entry(client: TestClient, model_id: str = _MODEL) -> dict[str, Any]:
    models = client.get("/api/v3/orchestration/models").json()
    return next(m for m in models if m["model_id"] == model_id)


def test_retrain_promote_deploy_round_trip(client: TestClient, reset_model: Callable[[str], None]) -> None:
    """1회차: 학습 실측 후보가 기준선을 이겨 승급·배포되고 아티팩트·버전이 영속화된다."""
    reset_model(_MODEL)
    before = _entry(client)
    assert before["version"] == "v3.0-R3"
    assert before["metrics_source"] == "seed"

    run = _trigger(client, candidate_latency_ms=120)

    # 상태머신이 전 단계를 통과했다.
    assert run["state"] == "succeeded"
    assert [stage["stage"] for stage in run["stages"]] == [
        "queued",
        "preparing",
        "training",
        "evaluating",
        "deploying",
    ]
    # 후보 지표가 실제 학습 산출물이다.
    assert run["training"]["source"] == "trained"
    assert run["training"]["dataset"]["protocol"] == "leave-one-out"
    assert run["training"]["latency_ms"] > 0.0
    # 승급 판정은 학습 기준선(절편-only) 대비 f1 비교로 이뤄졌다.
    assert run["evaluation"]["primary_metric"] == "f1"
    assert run["evaluation"]["current_value"] == run["training"]["baseline_metrics"]["f1"]
    assert run["evaluation"]["candidate_value"] > run["evaluation"]["current_value"]
    # 배포는 canary → full 로 진행됐다.
    assert [stage["stage"] for stage in run["deploy"]["stages"]] == ["canary", "full"]
    assert run["deploy"]["deployed"] is True
    assert run["active_version"] == "v3.1"

    # 아티팩트 파일에 모델 계수와 실측 지표가 남았다.
    artifact = json.loads(Path(run["artifact_path"]).read_text(encoding="utf-8"))
    assert artifact["version"] == "v3.1"
    assert artifact["metrics"] == run["candidate_metrics"]
    assert len(artifact["model"]["coefficients"]) == len(artifact["model"]["feature_names"])

    # 레지스트리가 승급 버전과 실측 지표를 노출한다(시드 상수 대체).
    after = _entry(client)
    assert after["version"] == "v3.1"
    assert after["next_version"] == "v3.2"
    assert after["metrics_source"] == "trained"
    assert after["metrics"] == run["candidate_metrics"]
    assert after["metrics"] != before["metrics"]

    # 실행 이력에 남는다.
    runs = client.get("/api/v3/orchestration/runs").json()
    assert any(entry["run_id"] == run["run_id"] for entry in runs)


def test_second_retrain_on_same_data_is_rejected(client: TestClient, reset_model: Callable[[str], None]) -> None:
    """2회차: 승급 기준이 실측 아티팩트로 올라가므로 같은 시드 재학습은 동점 반려된다(래칫)."""
    reset_model(_MODEL)
    first = _trigger(client, candidate_latency_ms=120)
    assert first["state"] == "succeeded"

    second = _trigger(client, candidate_latency_ms=120)
    assert second["state"] == "rejected"
    # 현행 기준이 시드가 아니라 1회차 실측치다.
    assert second["evaluation"]["current_value"] == first["candidate_metrics"]["f1"]
    assert second["evaluation"]["candidate_value"] == first["candidate_metrics"]["f1"]
    # 반려는 운영 버전을 바꾸지 않는다.
    assert second["active_version"] == "v3.1"
    assert _entry(client)["version"] == "v3.1"


def test_rollback_keeps_promoted_version(client: TestClient, reset_model: Callable[[str], None]) -> None:
    """승급 후 지연 초과 배포는 직전 버전을 유지하고 버전을 올리지 않는다."""
    reset_model(_MODEL)
    assert _trigger(client, candidate_latency_ms=120)["state"] == "succeeded"

    # 후보를 명시해 승급을 통과시키고 지연만 임계 초과로 만든다.
    rolled = _trigger(client, candidate_metrics={"f1": 0.99}, candidate_latency_ms=250)
    assert rolled["state"] == "rolled_back"
    assert rolled["active_version"] == "v3.1"
    assert [stage["stage"] for stage in rolled["deploy"]["stages"]] == ["canary", "rollback"]
    # 롤백은 승급을 확정하지 않는다 — 운영 버전은 v3.1에 머문다.
    assert _entry(client)["version"] == "v3.1"


def test_measured_latency_drives_canary_without_explicit_value(
    client: TestClient, reset_model: Callable[[str], None]
) -> None:
    """지연을 명시하지 않으면 학습 시 실측한 추론 지연으로 헬스체크한다."""
    reset_model(_MODEL)
    run = _trigger(client)
    assert run["state"] == "succeeded"
    measured = run["training"]["latency_ms"]
    assert run["deploy"]["stages"][0]["latency_ms"] == measured
    assert measured > 0.0


def test_explicit_metrics_do_not_become_the_measured_baseline(
    client: TestClient, reset_model: Callable[[str], None]
) -> None:
    """명시 주입 지표로 승급해도 다음 판정의 기준선은 검증된 실측치만 쓴다."""
    reset_model(_MODEL)
    injected = _trigger(client, candidate_metrics={"f1": 0.99}, candidate_latency_ms=120)
    assert injected["state"] == "succeeded"
    # 표시용으로는 노출하되 출처를 명시한다.
    entry = _entry(client)
    assert entry["metrics"]["f1"] == 0.99
    assert entry["metrics_source"] == "explicit"

    # 다음 학습은 0.99가 아니라 학습 기준선과 비교된다 — 검증 안 된 값이 문턱이 되지 않는다.
    following = _trigger(client, candidate_latency_ms=120)
    assert following["training"]["source"] == "trained"
    assert following["evaluation"]["current_value"] == following["training"]["baseline_metrics"]["f1"]
    assert following["state"] == "succeeded"


def test_explicit_candidate_skips_training_and_artifact_write(
    client: TestClient, reset_model: Callable[[str], None]
) -> None:
    """명시 후보가 오면 학습을 아예 하지 않는다 — 쓰지 않을 결과를 계산·기록하지 않는다."""
    reset_model(_MODEL)
    run = _trigger(client, candidate_metrics={"f1": 0.95})

    assert run["state"] == "succeeded"
    assert run["training"] == {"source": "explicit"}  # 학습 요약이 없다 = 학습을 돌리지 않았다
    assert next(s for s in run["stages"] if s["stage"] == "training")["status"] == "skipped"
    # 아티팩트 경로도 파일도 만들지 않는다(DB 지표와 파일 지표가 어긋날 여지 제거).
    assert run["artifact_path"] is None
    assert not (get_settings().model_artifact_dir / _MODEL / "v3.1.json").exists()
    # 명시 지표가 그대로 승급 판정에 쓰인다.
    assert run["candidate_metrics"] == {"f1": 0.95}
    assert run["evaluation"]["candidate_value"] == 0.95
    # 학습 기준선이 없으므로 현행 기준은 시드 상수 그대로다.
    assert run["evaluation"]["current_value"] == 0.884
    # 지연도 실측이 없으므로 기본 상수를 쓴다.
    assert run["deploy"]["stages"][0]["latency_ms"] == 120.0


def test_corrupt_seed_does_not_break_the_endpoint(
    client: TestClient,
    reset_model: Callable[[str], None],
    monkeypatch: Any,
) -> None:
    """손상된 시드(0·null)는 경계에서 ValueError로 걸러져 500이 아니라 fallback으로 흐른다."""
    reset_model(_MODEL)
    broken = {"regions": [{"id": "x", "history": [1000, 0, 980], "riskIndex": None, "birthRate": 0.8, "agingIndex": 33.0}]}
    monkeypatch.setattr(trainer_module, "get_seed", lambda: broken)

    response = client.post(_EVENTS, json={"model_id": _MODEL, "trigger": "manual"})
    assert response.status_code == 200  # ZeroDivisionError/TypeError로 500이 되지 않는다
    run = response.json()
    assert run["training"]["source"] == "derived"
    assert run["artifact_path"] is None
    assert run["state"] == "succeeded"


def test_non_array_regions_seed_does_not_break_the_endpoint(
    client: TestClient,
    reset_model: Callable[[str], None],
    monkeypatch: Any,
) -> None:
    """regions 자체가 배열이 아닌 시드(None·숫자)도 경계에서 걸러진다."""
    reset_model(_MODEL)
    monkeypatch.setattr(trainer_module, "get_seed", lambda: {"regions": None})

    response = client.post(_EVENTS, json={"model_id": _MODEL, "trigger": "manual"})
    assert response.status_code == 200  # TypeError로 500이 되지 않는다
    run = response.json()
    assert run["training"]["source"] == "derived"
    assert run["artifact_path"] is None


def test_training_failure_falls_back_to_derived_candidate(
    client: TestClient,
    reset_model: Callable[[str], None],
    monkeypatch: Any,
) -> None:
    """학습이 불가능해도 상태머신은 멈추지 않고, 어느 경로를 탔는지 기록한다."""
    reset_model(_MODEL)

    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("시드 시계열 손상 시뮬레이션")

    monkeypatch.setattr(orchestrator_module.Trainer, "train", _boom)
    run = _trigger(client)

    assert run["training"]["source"] == "derived"
    assert run["artifact_path"] is None
    assert next(s for s in run["stages"] if s["stage"] == "training")["status"] == "skipped"
    # 파생 후보는 시드 지표 대비 개선폭이 고정이라 승급된다(안전망 경로).
    assert run["state"] == "succeeded"
    assert run["candidate_metrics"]["f1"] == 0.912  # 시드 0.884 + 0.028
    # 지연도 실측이 없으므로 기본 상수를 쓴다.
    assert run["deploy"]["stages"][0]["latency_ms"] == 120.0
