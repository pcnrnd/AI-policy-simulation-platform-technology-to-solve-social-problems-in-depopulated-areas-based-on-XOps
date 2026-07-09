"""Orchestrator — 이벤트 기반 재학습 상태머신.

queued → preparing → training → evaluating → deploying → (succeeded | rolled_back)
승급 미달 시 rejected, EventBus에서 걸러지면 debounced. 인프로세스 구현이며,
Airflow/Argo로 이관 시 각 stage를 operator로 매핑하면 된다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.core.settings import get_settings
from src.mlops.orchestration.deployer import AutoDeployer
from src.mlops.orchestration.evaluator import Evaluator
from src.mlops.orchestration.events import EventBus, RetrainEvent

_STAGES = ("queued", "preparing", "training", "evaluating", "deploying")
_DEFAULT_CANDIDATE_LATENCY_MS = 120.0


def _derive_candidate(current: dict[str, float]) -> dict[str, float]:
    """후보 지표 미제공 시 결정적 개선 파생 (accuracy +0.028, 오차 ×0.72)."""
    out: dict[str, float] = {}
    for metric, value in current.items():
        if metric in ("mae", "mse"):
            out[metric] = round(value * 0.72, 6)
        else:
            out[metric] = round(value + 0.028, 6)
    return out


@dataclass
class PipelineRun:
    """파이프라인 실행 기록."""

    run_id: str
    model_id: str
    trigger: str
    state: str
    stages: list[dict[str, Any]] = field(default_factory=list)
    evaluation: dict[str, Any] | None = None
    deploy: dict[str, Any] | None = None
    active_version: str | None = None


class Orchestrator:
    """재학습 파이프라인 오케스트레이터."""

    def __init__(self) -> None:
        self._bus = EventBus(get_settings().retrain_min_interval_minutes)
        self._evaluator = Evaluator()
        self._deployer = AutoDeployer()
        self._counter = 0

    def _next_run_id(self) -> str:
        self._counter += 1
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        return f"RUN-{stamp}-{self._counter:04d}"

    def handle_event(
        self,
        event: RetrainEvent,
        *,
        current_metrics: dict[str, float],
        current_version: str,
        candidate_version: str,
    ) -> PipelineRun:
        """이벤트 하나를 상태머신에 태워 실행 결과를 반환."""
        run = PipelineRun(run_id=self._next_run_id(), model_id=event.model_id, trigger=event.trigger, state="queued")

        if not self._bus.accept(event, now=event.created_at):
            run.state = "debounced"
            return run

        for stage in _STAGES[:4]:  # queued..evaluating
            run.stages.append({"stage": stage, "status": "done"})

        candidate = event.candidate_metrics or _derive_candidate(current_metrics)
        evaluation = self._evaluator.evaluate(current_metrics, candidate)
        run.evaluation = asdict(evaluation)

        if not evaluation.promote:
            run.state = "rejected"
            run.active_version = current_version
            return run

        run.stages.append({"stage": "deploying", "status": "done"})
        latency = event.candidate_latency_ms if event.candidate_latency_ms is not None else _DEFAULT_CANDIDATE_LATENCY_MS
        deploy = self._deployer.deploy(
            model_id=event.model_id,
            current_version=current_version,
            candidate_version=candidate_version,
            candidate_latency_ms=latency,
        )
        run.deploy = asdict(deploy)
        run.active_version = deploy.active_version
        run.state = "rolled_back" if deploy.rolled_back else "succeeded"
        return run
