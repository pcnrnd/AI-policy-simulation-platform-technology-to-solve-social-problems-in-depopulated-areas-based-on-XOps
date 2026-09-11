"""학습·평가·아티팩트 재로딩 단위 테스트(R2) — insufficient_data, YYYYMM 산술, predict 재현."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.core.settings import Settings
from src.realdata import features, models, pg_reader, snapshot
from src.realdata.errors import InsufficientData, RealdataError

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "realdata"
_MODEL_ID = "namwon-nonlocal-visitors-next-month"


def _load_fixture(name: str) -> list[dict[str, Any]]:
    return json.loads((_FIXTURE_DIR / name).read_text(encoding="utf-8"))


@pytest.fixture()
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    settings = Settings(realdata_dataset_dir=tmp_path / "datasets", model_artifact_dir=tmp_path / "models")
    monkeypatch.setattr(snapshot, "get_settings", lambda: settings)
    monkeypatch.setattr(models, "get_settings", lambda: settings)
    return settings


@pytest.fixture()
def trained_outcome(monkeypatch: pytest.MonkeyPatch, isolated_env: Settings) -> tuple[models.TrainOutcome, str]:
    fixture = _load_fixture("kt_monthly_visitors.json")
    monkeypatch.setattr(pg_reader, "fetch_all", lambda *a, **kw: fixture)
    dataset = snapshot.create_dataset("nonlocal_visitors")
    outcome = models.train(_MODEL_ID, dataset.dataset_id, version="v0-test")
    return outcome, dataset.dataset_id


def test_train_produces_expected_observed_and_forecast_month(
    trained_outcome: tuple[models.TrainOutcome, str]
) -> None:
    outcome, _ = trained_outcome
    assert outcome.observed_end_month == 202304
    assert outcome.forecast_month == 202305  # 일반적인(연도 안 넘는) YYYYMM+1
    assert outcome.model_id == _MODEL_ID
    assert outcome.train_rows > 0
    assert set(outcome.metrics) == {"mae", "rmse", "wape"}
    assert set(outcome.baseline) == {"name", "mae", "rmse", "wape"}
    assert outcome.eval_period["from"] == 202302
    assert outcome.eval_period["to"] == 202304
    assert outcome.eval_period["n"] > 0


def test_forecast_month_wraps_december_to_january(
    monkeypatch: pytest.MonkeyPatch, isolated_env: Settings
) -> None:
    fixture = _load_fixture("kt_monthly_visitors.json")
    # 관측 종료월이 12월로 끝나도록 전체를 8개월 밀어 재구성(202304 -> 202312).
    shifted = [{**row, "base_ym": features.add_month(row["base_ym"], 8)} for row in fixture]
    monkeypatch.setattr(pg_reader, "fetch_all", lambda *a, **kw: shifted)

    dataset = snapshot.create_dataset("nonlocal_visitors")
    outcome = models.train(_MODEL_ID, dataset.dataset_id, version="v0-dec")

    assert outcome.observed_end_month == 202312
    assert outcome.forecast_month == 202401


def test_load_artifact_matches_written_file(trained_outcome: tuple[models.TrainOutcome, str]) -> None:
    outcome, _ = trained_outcome
    artifact = models.load_artifact(outcome.model_id, outcome.version)

    assert artifact["model_id"] == outcome.model_id
    assert artifact["observed_end_month"] == outcome.observed_end_month
    assert artifact["forecast_month"] == outcome.forecast_month
    assert artifact["train_rows"] == outcome.train_rows
    assert Path(outcome.artifact_path).exists()


def test_predict_reproduces_from_artifact_alone(
    monkeypatch: pytest.MonkeyPatch, isolated_env: Settings, trained_outcome: tuple[models.TrainOutcome, str]
) -> None:
    outcome, dataset_id = trained_outcome
    dataset = snapshot.load_dataset(dataset_id)
    artifact = models.load_artifact(outcome.model_id, outcome.version)

    forecast_rows = features.build_feature_rows(dataset, for_month=outcome.forecast_month)
    assert forecast_rows  # 최소 1개 동은 예측 가능해야 함

    first = models.predict(artifact, forecast_rows)
    second = models.predict(artifact, forecast_rows)

    assert first == second  # 결정적 — 재로딩해도 같은 입력엔 같은 출력
    assert all(isinstance(v, float) for v in first)


def test_train_raises_insufficient_data_below_12_months(
    monkeypatch: pytest.MonkeyPatch, isolated_env: Settings
) -> None:
    fixture = _load_fixture("kt_monthly_visitors.json")
    short_fixture = [r for r in fixture if r["base_ym"] < 202207]  # 6개월치만
    monkeypatch.setattr(pg_reader, "fetch_all", lambda *a, **kw: short_fixture)
    dataset = snapshot.create_dataset("nonlocal_visitors")

    with pytest.raises(InsufficientData):
        models.train(_MODEL_ID, dataset.dataset_id, version="v0-short")


def test_train_rejects_model_dataset_target_mismatch(
    trained_outcome: tuple[models.TrainOutcome, str]
) -> None:
    _, dataset_id = trained_outcome
    with pytest.raises(RealdataError):
        models.train("namwon-observed-sales-next-month", dataset_id, version="v0-mismatch")
