"""ModelRegistry — 인프로세스 모델 스토어 + 오케스트레이터 + 실행 이력의 단일 출처.

모니터링(드리프트 감지)과 오케스트레이션(재학습) 라우터가 공유한다. 이로써
"모니터링이 성능저하를 감지하면 오케스트레이션이 재학습" 서사를 한 곳에서 연결한다.

모델 스토어는 SQLite(`model_versions`·`model_artifacts`)에 영속화된다 — 카탈로그(사용자
등록 소스)와 같은 방식이다. 학습된 모델 계수는 로컬 아티팩트 파일에 있고 DB에는 그 경로와
실측 지표만 둔다. MLflow/MinIO 없이 재학습→승급→배포가 완결되며, 필요해지면
provider seam으로 갈아끼울 수 있다.
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import asdict
from functools import lru_cache
from typing import Any

from src.core import db
from src.core.exceptions import SourceNotFoundError
from src.core.logger import get_logger
from src.core.settings import get_settings
from src.mlops.orchestration.events import RetrainEvent
from src.mlops.orchestration.orchestrator import Orchestrator, PipelineRun

_logger = get_logger("xops.orchestration")
_VERSION_PATTERN = re.compile(r"^v(\d+)\.(\d+)")

# 프론트 레지스트리 3종 대응 시드.
# horizon(예측 지평, 기간)은 학습 문제의 난이도를 정한다 — 지평이 길수록 어려워 지표가 낮다.
# metrics는 최초 학습 전 표시용 시드값이며, 승급 후에는 아티팩트의 실측 지표로 대체된다.
_SEED_STORE: dict[str, dict[str, Any]] = {
    "population-forecast": {
        "version": "v3.0-R3",
        "horizon": 1,
        "metrics": {"accuracy": 0.892, "f1": 0.884, "precision": 0.891, "recall": 0.878, "mse": 0.041, "mae": 0.125},
    },
    "vital-population": {
        "version": "v2.4",
        "horizon": 2,
        "metrics": {"accuracy": 0.861, "f1": 0.852, "precision": 0.858, "recall": 0.847, "mse": 0.058, "mae": 0.147},
    },
    "settlement-demand": {
        "version": "v1.7",
        "horizon": 3,
        "metrics": {"accuracy": 0.834, "f1": 0.821, "precision": 0.829, "recall": 0.814, "mse": 0.071, "mae": 0.166},
    },
}


def next_version(current: str) -> str:
    """현재 버전의 마이너를 1 올린다 — `v3.0-R3`→`v3.1`, `v3.1`→`v3.2`.

    고정 상수를 쓰면 두 번째 승급이 같은 버전을 재기록해 아티팩트가 덮어써진다.
    패턴이 맞지 않으면 접미사 `-next`를 붙여 최소한 충돌은 피한다.
    """
    match = _VERSION_PATTERN.match(current)
    if match is None:
        return f"{current}-next"
    major, minor = int(match.group(1)), int(match.group(2))
    return f"v{major}.{minor + 1}"


class ModelRegistry:
    """운영 모델 상태와 재학습 파이프라인 실행을 관리."""

    def __init__(self, store: dict[str, dict[str, Any]]) -> None:
        self._store = store  # 시드(지표·지평). 현재 버전·아티팩트·실행 이력은 SQLite가 소스오브트루스.
        self._orchestrator = Orchestrator()

    def _version(self, model_id: str, seed_version: str) -> str:
        return db.get_model_version(model_id) or seed_version

    @staticmethod
    def _artifact_metrics(model_id: str, version: str) -> tuple[dict[str, float], str] | None:
        """현행 버전 아티팩트의 (승급 지표, 출처). 없으면 None."""
        artifact = db.get_model_artifact(model_id, version)
        if artifact is None:
            return None
        metrics: dict[str, float] = artifact.get("metrics", {})
        if not metrics:
            return None
        return metrics, str(artifact.get("metrics_source", "trained"))

    def _measured(self, model_id: str, version: str) -> dict[str, float] | None:
        """승급 판정의 현행 기준이 될 **실측** 지표 — 학습으로 측정된 것만 신뢰한다.

        명시 주입(`candidate_metrics`)이나 파생 fallback으로 승급된 값은 우리 평가
        프로토콜로 측정된 값이 아니므로 기준선으로 쓰지 않는다. 검증되지 않은 수치가
        영구적인 승급 문턱이 되는 것을 막는다(표시용으로는 그대로 노출한다).
        """
        found = self._artifact_metrics(model_id, version)
        if found is None:
            return None
        metrics, source = found
        return metrics if source == "trained" else None

    def models(self) -> list[dict[str, Any]]:
        """등록된 운영 모델과 현재 버전/지표. 승급 이력이 있으면 실측 지표를 노출한다."""
        out: list[dict[str, Any]] = []
        for model_id, info in self._store.items():
            version = self._version(model_id, info["version"])
            found = self._artifact_metrics(model_id, version)
            metrics, source = found if found is not None else (info["metrics"], "seed")
            out.append(
                {
                    **info,
                    "model_id": model_id,
                    "version": version,
                    "next_version": next_version(version),
                    "metrics": metrics,
                    "metrics_source": source,
                }
            )
        return out

    def runs(self) -> list[dict[str, Any]]:
        return db.list_runs()

    def trigger(
        self,
        *,
        model_id: str,
        trigger: str = "manual",
        candidate_metrics: dict[str, float] | None = None,
        candidate_latency_ms: float | None = None,
    ) -> PipelineRun:
        """재학습 이벤트를 상태머신에 태우고 승급 성공 시 버전·아티팩트를 갱신(SQLite 영속화)."""
        model = self._store.get(model_id)
        if model is None:
            raise SourceNotFoundError(f"등록된 모델이 아닙니다: {model_id}")

        event = RetrainEvent(
            model_id=model_id,
            trigger=trigger,
            candidate_metrics=candidate_metrics or {},
            candidate_latency_ms=candidate_latency_ms,
        )
        current_version = self._version(model_id, model["version"])
        run = self._orchestrator.handle_event(
            event,
            current_metrics=model["metrics"],
            measured_current=self._measured(model_id, current_version),
            current_version=current_version,
            candidate_version=next_version(current_version),
            horizon=model.get("horizon", get_settings().train_default_horizon),
        )
        db.append_run(asdict(run))
        if run.state == "succeeded":
            self._promote(model_id, run)
        _logger.info(f"retrain model={model_id} trigger={trigger} run={run.run_id} state={run.state}")
        return run

    @staticmethod
    def _promote(model_id: str, run: PipelineRun) -> None:
        """승급 확정 — 운영 버전 갱신 + 아티팩트 메타 기록."""
        version = run.active_version
        if version is None:
            return
        training = run.training or {}
        db.set_model_version(model_id, version)
        db.set_model_artifact(
            model_id,
            version,
            {
                "model_id": model_id,
                "version": version,
                "run_id": run.run_id,
                "metrics": run.candidate_metrics or {},
                # "trained"만 다음 판정의 기준선으로 신뢰한다 — `_measured` 참조.
                "metrics_source": str(training.get("source", "trained")),
                "training": training,
                "artifact_path": run.artifact_path,
            },
        )


@lru_cache
def get_registry() -> ModelRegistry:
    """레지스트리 싱글톤 (시드 스토어의 복사본으로 초기화)."""
    return ModelRegistry(deepcopy(_SEED_STORE))
