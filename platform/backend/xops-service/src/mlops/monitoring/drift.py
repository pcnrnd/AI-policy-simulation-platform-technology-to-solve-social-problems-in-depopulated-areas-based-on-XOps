"""DriftDetector — PSI / KL-Divergence 실제 계산.

임계: PSI ≥ 0.2, KL ≥ 0.1 (Notion 명세, settings로 조정). 분포는 카운트/퍼센트 어느 쪽이든
받아 확률로 정규화한다. 0 확률은 log 발산을 막기 위해 epsilon으로 clip.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from src.core.settings import get_settings

_EPS = 1e-6


def _normalize(dist: Sequence[float]) -> list[float]:
    total = sum(dist)
    if total <= 0:
        raise ValueError("분포 합이 0보다 커야 합니다.")
    return [max(x / total, _EPS) for x in dist]


def population_stability_index(reference: Sequence[float], current: Sequence[float]) -> float:
    """PSI = Σ (cur - ref) * ln(cur / ref)."""
    if len(reference) != len(current):
        raise ValueError("reference와 current 버킷 수가 같아야 합니다.")
    ref, cur = _normalize(reference), _normalize(current)
    return sum((c - r) * math.log(c / r) for r, c in zip(ref, cur))


def kl_divergence(reference: Sequence[float], current: Sequence[float]) -> float:
    """KL(ref || cur) = Σ ref * ln(ref / cur)."""
    if len(reference) != len(current):
        raise ValueError("reference와 current 버킷 수가 같아야 합니다.")
    ref, cur = _normalize(reference), _normalize(current)
    return sum(r * math.log(r / c) for r, c in zip(ref, cur))


@dataclass(frozen=True)
class DriftResult:
    """드리프트 판정 결과."""

    psi: float
    kl: float
    psi_threshold: float
    kl_threshold: float
    drifted: bool


class DriftDetector:
    """분포 비교로 드리프트를 판정."""

    def detect(self, reference: Sequence[float], current: Sequence[float]) -> DriftResult:
        settings = get_settings()
        psi = population_stability_index(reference, current)
        kl = kl_divergence(reference, current)
        drifted = psi >= settings.psi_threshold or kl >= settings.kl_threshold
        return DriftResult(
            psi=round(psi, 6),
            kl=round(kl, 6),
            psi_threshold=settings.psi_threshold,
            kl_threshold=settings.kl_threshold,
            drifted=drifted,
        )
