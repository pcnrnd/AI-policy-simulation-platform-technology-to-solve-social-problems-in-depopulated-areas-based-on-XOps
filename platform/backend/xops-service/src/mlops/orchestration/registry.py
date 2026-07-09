"""ModelRegistry — 인프로세스 모델 스토어 + 오케스트레이터 + 실행 이력의 단일 출처.

모니터링(드리프트 감지)과 오케스트레이션(재학습) 라우터가 공유한다. 이로써
"모니터링이 성능저하를 감지하면 오케스트레이션이 재학습" 서사를 한 곳에서 연결한다.
실제 Model Store(MLflow 등) 연동은 후속(provider seam).
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from typing import Any

from src.core.exceptions import SourceNotFoundError
from src.core.logger import get_logger
from src.mlops.orchestration.events import RetrainEvent
from src.mlops.orchestration.orchestrator import Orchestrator, PipelineRun

_logger = get_logger("xops.orchestration")

# 프론트 레지스트리 3종 대응 시드.
_SEED_STORE: dict[str, dict[str, Any]] = {
    "population-forecast": {
        "version": "v3.0-R3",
        "next_version": "v3.1",
        "metrics": {"accuracy": 0.892, "f1": 0.884, "precision": 0.891, "recall": 0.878, "mse": 0.041, "mae": 0.125},
    },
    "living-population": {
        "version": "v2.4",
        "next_version": "v2.5",
        "metrics": {"accuracy": 0.861, "f1": 0.852, "precision": 0.858, "recall": 0.847, "mse": 0.058, "mae": 0.147},
    },
    "settlement-demand": {
        "version": "v1.7",
        "next_version": "v1.8",
        "metrics": {"accuracy": 0.834, "f1": 0.821, "precision": 0.829, "recall": 0.814, "mse": 0.071, "mae": 0.166},
    },
}


class ModelRegistry:
    """운영 모델 상태와 재학습 파이프라인 실행을 관리."""

    def __init__(self, store: dict[str, dict[str, Any]]) -> None:
        self._store = store
        self._orchestrator = Orchestrator()
        self._runs: list[PipelineRun] = []

    def models(self) -> list[dict[str, Any]]:
        return [{"model_id": mid, **info} for mid, info in self._store.items()]

    def runs(self) -> list[PipelineRun]:
        return list(self._runs)

    def trigger(
        self,
        *,
        model_id: str,
        trigger: str = "manual",
        candidate_metrics: dict[str, float] | None = None,
        candidate_latency_ms: float | None = None,
    ) -> PipelineRun:
        """재학습 이벤트를 상태머신에 태우고 승급 성공 시 버전을 갱신."""
        model = self._store.get(model_id)
        if model is None:
            raise SourceNotFoundError(f"등록된 모델이 아닙니다: {model_id}")

        event = RetrainEvent(
            model_id=model_id,
            trigger=trigger,
            candidate_metrics=candidate_metrics or {},
            candidate_latency_ms=candidate_latency_ms,
        )
        run = self._orchestrator.handle_event(
            event,
            current_metrics=model["metrics"],
            current_version=model["version"],
            candidate_version=model["next_version"],
        )
        self._runs.append(run)
        if run.state == "succeeded":
            model["version"] = model["next_version"]
        _logger.info(f"retrain model={model_id} trigger={trigger} run={run.run_id} state={run.state}")
        return run


@lru_cache
def get_registry() -> ModelRegistry:
    """레지스트리 싱글톤 (시드 스토어의 복사본으로 초기화)."""
    return ModelRegistry(deepcopy(_SEED_STORE))
