"""MLOps 도메인 단위 테스트 — 지표·드리프트·이상치·평가·배포·이벤트."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.mlops.monitoring.drift import DriftDetector, kl_divergence, population_stability_index
from src.mlops.monitoring.explain import ExplainabilityModule
from src.mlops.monitoring.metrics import MetricCollector
from src.mlops.monitoring.outliers import OutlierDetector
from src.mlops.orchestration.deployer import AutoDeployer
from src.mlops.orchestration.evaluator import Evaluator
from src.mlops.orchestration.events import EventBus, RetrainEvent


# ── metrics ──
def test_regression_metrics() -> None:
    out = MetricCollector().regression([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])
    assert out["mse"] == pytest.approx(1 / 3, abs=1e-6)
    assert out["mae"] == pytest.approx(1 / 3, abs=1e-6)


def test_classification_metrics_perfect() -> None:
    out = MetricCollector().classification([1, 0, 1, 0], [1, 0, 1, 0])
    assert out == {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_metrics_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        MetricCollector().regression([1.0], [1.0, 2.0])


# ── drift ──
def test_identical_distribution_no_drift() -> None:
    ref = [10, 20, 30, 40]
    result = DriftDetector().detect(ref, ref)
    assert result.psi == pytest.approx(0.0, abs=1e-9)
    assert result.drifted is False


def test_shifted_distribution_flags_drift() -> None:
    result = DriftDetector().detect([10, 20, 30, 40], [40, 30, 20, 10])
    assert result.psi > 0.2
    assert result.drifted is True


def test_psi_and_kl_nonnegative() -> None:
    assert population_stability_index([1, 1, 1], [3, 1, 1]) >= 0
    assert kl_divergence([1, 1, 1], [3, 1, 1]) >= 0


# ── outliers ──
def test_zscore_flags_extreme() -> None:
    # z-score가 임계(3.0)를 넘으려면 정상 표본이 충분해야 한다.
    result = OutlierDetector().zscore([10.0] * 30 + [500.0])
    assert any(o["value"] == 500.0 for o in result.outliers)


def test_iqr_flags_extreme() -> None:
    result = OutlierDetector().iqr([1, 2, 3, 4, 5, 100])
    assert any(o["value"] == 100 for o in result.outliers)


# ── evaluator ──
def test_evaluator_promotes_on_f1() -> None:
    r = Evaluator().evaluate({"f1": 0.80, "accuracy": 0.90}, {"f1": 0.85, "accuracy": 0.80})
    assert r.promote is True
    assert r.primary_metric == "f1"


def test_evaluator_lower_is_better_for_mae() -> None:
    r = Evaluator().evaluate({"mae": 0.20}, {"mae": 0.10})
    assert r.promote is True and r.primary_metric == "mae"


def test_evaluator_no_common_metric() -> None:
    r = Evaluator().evaluate({"foo": 1.0}, {"bar": 2.0})
    assert r.promote is False and r.primary_metric is None


# ── deployer ──
def test_deploy_promotes_when_latency_ok() -> None:
    d = AutoDeployer().deploy(model_id="m", current_version="v1", candidate_version="v2", candidate_latency_ms=120)
    assert d.deployed is True and d.active_version == "v2"


def test_deploy_rolls_back_on_high_latency() -> None:
    d = AutoDeployer().deploy(model_id="m", current_version="v1", candidate_version="v2", candidate_latency_ms=250)
    assert d.rolled_back is True and d.active_version == "v1"


# ── event bus ──
def test_manual_event_always_accepted() -> None:
    bus = EventBus(30)
    now = datetime.now(timezone.utc)
    assert bus.accept(RetrainEvent("m", "manual"), now=now) is True
    assert bus.accept(RetrainEvent("m", "manual"), now=now) is True


def test_drift_event_debounced_within_interval() -> None:
    bus = EventBus(30)
    now = datetime.now(timezone.utc)
    assert bus.accept(RetrainEvent("m", "drift"), now=now) is True
    assert bus.accept(RetrainEvent("m", "drift"), now=now + timedelta(minutes=5)) is False
    assert bus.accept(RetrainEvent("m", "drift"), now=now + timedelta(minutes=31)) is True


def test_explain_backend_and_ranking() -> None:
    mod = ExplainabilityModule()
    assert mod.backend in ("shap", "pure-python-fallback")
    ranked = mod.rank_features({"a": [0.1, 0.3], "b": [-0.5, -0.5]})
    assert ranked[0]["feature"] == "b"  # |−0.5| > |0.2|
