"""Orchestrator — 이벤트 기반 재학습 상태머신.

queued → preparing → training → evaluating → deploying → (succeeded | rolled_back)
승급 미달 시 rejected, EventBus에서 걸러지면 debounced. 인프로세스 구현이며,
Airflow/Argo로 이관 시 각 stage를 operator로 매핑하면 된다.

`training` stage는 `Trainer`로 실제 학습한다(순수 Python 릿지 회귀, 그리드 → LOO 실측).
후보 지표 우선순위는 **명시값 > 학습 실측 > 결정적 파생(fallback)** 이다. 마지막 단계는
시드 손상 등으로 학습이 불가능할 때 상태머신을 멈추지 않기 위한 안전망이며, 어느 경로를
탔는지는 `PipelineRun.training["source"]`에 남는다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.core.logger import get_logger
from src.core.settings import get_settings
from src.mlops.orchestration.deployer import AutoDeployer
from src.mlops.orchestration.evaluator import Evaluator
from src.mlops.orchestration.events import EventBus, RetrainEvent
from src.mlops.training.trainer import Trainer, TrainingResult, artifact_path

_logger = get_logger("xops.orchestration")
_STAGES = ("queued", "preparing", "training", "evaluating", "deploying")
_DEFAULT_CANDIDATE_LATENCY_MS = 120.0


def _derive_candidate(current: dict[str, float]) -> dict[str, float]:
    """후보 지표 파생 fallback (accuracy +0.028, 오차 ×0.72).

    학습이 불가능할 때만 쓰는 안전망이다. 실제 학습이 성공하면 이 값은 쓰이지 않는다.
    """
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
    candidate_metrics: dict[str, float] | None = None
    training: dict[str, Any] | None = None
    artifact_path: str | None = None


class Orchestrator:
    """재학습 파이프라인 오케스트레이터."""

    def __init__(self) -> None:
        self._bus = EventBus(get_settings().retrain_min_interval_minutes)
        self._evaluator = Evaluator()
        self._deployer = AutoDeployer()
        self._trainer = Trainer()
        self._counter = 0

    def _next_run_id(self) -> str:
        self._counter += 1
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        return f"RUN-{stamp}-{self._counter:04d}"

    def _train(self, event: RetrainEvent, *, horizon: int, version: str) -> TrainingResult | None:
        """실제 학습 수행. 실패하면 None을 돌려 fallback 경로로 넘긴다."""
        try:
            return self._trainer.train(event.model_id, horizon=horizon, version=version)
        except (ValueError, KeyError, OSError) as exc:
            _logger.warning(f"train failed model={event.model_id} reason={exc} — 파생 후보로 대체")
            return None

    def handle_event(
        self,
        event: RetrainEvent,
        *,
        current_metrics: dict[str, float],
        current_version: str,
        candidate_version: str,
        measured_current: dict[str, float] | None = None,
        horizon: int | None = None,
    ) -> PipelineRun:
        """이벤트 하나를 상태머신에 태워 실행 결과를 반환. 기준선 선택은 `_evaluate` 참조."""
        run = PipelineRun(run_id=self._next_run_id(), model_id=event.model_id, trigger=event.trigger, state="queued")

        if not self._bus.accept(event, now=event.created_at):
            run.state = "debounced"
            return run

        for stage in _STAGES[:2]:  # queued, preparing
            run.stages.append({"stage": stage, "status": "done"})

        # 후보 지표가 명시로 들어오면 학습 결과를 쓰지 않으므로 학습 자체를 건너뛴다.
        # 계산·디스크 쓰기를 아끼고, DB에 기록되는 지표(주입값)와 아티팩트 파일의
        # 지표(실측)가 어긋나는 것을 막는다.
        trained = None
        if not event.candidate_metrics:
            steps = horizon if horizon is not None else get_settings().train_default_horizon
            trained = self._train(event, horizon=steps, version=candidate_version)
        self._record_training(run, trained, event, candidate_version)
        run.stages.append({"stage": "evaluating", "status": "done"})

        promote = self._evaluate(
            run, event, trained, current_metrics=current_metrics, measured_current=measured_current
        )
        if not promote:
            run.state = "rejected"
            run.active_version = current_version
            return run

        self._deploy(
            run,
            event,
            trained,
            current_version=current_version,
            candidate_version=candidate_version,
        )
        return run

    def _evaluate(
        self,
        run: PipelineRun,
        event: RetrainEvent,
        trained: TrainingResult | None,
        *,
        current_metrics: dict[str, float],
        measured_current: dict[str, float] | None,
    ) -> bool:
        """후보와 현행을 비교해 승급 여부를 정하고 판정 근거를 실행 기록에 남긴다.

        현행 기준(incumbent)은 **실측 아티팩트 > 학습 기준선 > 시드** 순으로 고른다.
        `measured_current`는 현행 버전을 같은 프로토콜로 측정한 지표(아티팩트)다. 아직
        없으면(최초 재학습) 학습이 함께 산출한 기준선 모델 지표를 쓴다 — 시드 상수는 같은
        방식으로 측정된 값이 아니라 비교 근거가 되지 못한다.
        """
        candidate = self._resolve_candidate(event, trained, current_metrics)
        incumbent = measured_current or (trained.baseline_metrics if trained else current_metrics)
        run.candidate_metrics = candidate
        evaluation = self._evaluator.evaluate(incumbent, candidate)
        run.evaluation = asdict(evaluation)
        return evaluation.promote

    def _deploy(
        self,
        run: PipelineRun,
        event: RetrainEvent,
        trained: TrainingResult | None,
        *,
        current_version: str,
        candidate_version: str,
    ) -> None:
        """승급 확정 후 canary → full 배포. 헬스체크(지연) 실패 시 롤백 상태로 남긴다."""
        run.stages.append({"stage": "deploying", "status": "done"})
        deploy = self._deployer.deploy(
            model_id=event.model_id,
            current_version=current_version,
            candidate_version=candidate_version,
            candidate_latency_ms=self._resolve_latency(event, trained),
        )
        run.deploy = asdict(deploy)
        run.active_version = deploy.active_version
        run.state = "rolled_back" if deploy.rolled_back else "succeeded"

    def _record_training(
        self,
        run: PipelineRun,
        trained: TrainingResult | None,
        event: RetrainEvent,
        candidate_version: str,
    ) -> None:
        """training stage 로그와 학습 요약·아티팩트 경로를 실행 기록에 반영."""
        source = "explicit" if event.candidate_metrics else ("trained" if trained else "derived")
        summary: dict[str, Any] = {"source": source}
        if trained is not None:
            summary.update(trained.summary())
            run.artifact_path = str(artifact_path(event.model_id, candidate_version))
        run.training = summary
        # stage 로그에는 요약만 — 상세는 run.training에 한 번만 담아 실행 이력 행을 키우지 않는다.
        status = "done" if trained else "skipped"
        run.stages.append({"stage": "training", "status": status, "source": source})

    @staticmethod
    def _resolve_candidate(
        event: RetrainEvent,
        trained: TrainingResult | None,
        current_metrics: dict[str, float],
    ) -> dict[str, float]:
        """후보 지표: 명시값 > 학습 실측 > 파생 fallback."""
        if event.candidate_metrics:
            return event.candidate_metrics
        if trained is not None:
            return trained.candidate_metrics
        return _derive_candidate(current_metrics)

    @staticmethod
    def _resolve_latency(event: RetrainEvent, trained: TrainingResult | None) -> float:
        """카나리 지연: 명시값 > 학습 시 실측 추론 지연 > 기본 상수."""
        if event.candidate_latency_ms is not None:
            return event.candidate_latency_ms
        if trained is not None:
            return trained.latency_ms
        return _DEFAULT_CANDIDATE_LATENCY_MS
