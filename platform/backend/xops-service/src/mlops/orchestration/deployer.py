"""AutoDeployer — canary → full 배포, 헬스체크 실패 시 롤백.

승급 후보의 예측 지연(latency)이 rollback_latency_ms(기본 200ms)를 초과하면 롤백:
직전 버전을 유지하고 트래픽을 전환하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.core.settings import get_settings


@dataclass(frozen=True)
class DeployResult:
    """배포 결과 — 최종 상태와 단계 로그."""

    deployed: bool
    rolled_back: bool
    active_version: str
    stages: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""


class AutoDeployer:
    """canary → full 승급 배포기."""

    def deploy(
        self, *, model_id: str, current_version: str, candidate_version: str, candidate_latency_ms: float
    ) -> DeployResult:
        threshold = get_settings().rollback_latency_ms
        stages: list[dict[str, Any]] = [{"stage": "canary", "version": candidate_version, "latency_ms": candidate_latency_ms}]

        if candidate_latency_ms > threshold:
            stages.append({"stage": "rollback", "version": current_version})
            return DeployResult(
                deployed=False,
                rolled_back=True,
                active_version=current_version,
                stages=stages,
                reason=f"헬스체크 실패: latency {candidate_latency_ms}ms > 임계 {threshold}ms — 직전 버전 유지",
            )

        stages.append({"stage": "full", "version": candidate_version})
        return DeployResult(
            deployed=True,
            rolled_back=False,
            active_version=candidate_version,
            stages=stages,
            reason=f"헬스체크 통과: latency {candidate_latency_ms}ms ≤ 임계 {threshold}ms",
        )
