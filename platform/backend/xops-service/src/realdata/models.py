"""남원 23동 pooled Ridge 학습·평가·예측 — 기존 순수 파이썬 `RidgeRegressor` 재사용(R2).

v1은 고정 학습(≤T−3 표본) + T−2·T−1·T 순차 평가로 단순화한다(G0 §10-3). 같은 모델이
평가에도, 다음 달 예측 아티팩트에도 쓰인다 — 별도 "배포용 재학습" 단계는 두지 않는다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.settings import get_settings
from src.mlops.training.model import RidgeRegressor
from src.realdata import evaluate, snapshot
from src.realdata.errors import InsufficientData, RealdataError
from src.realdata.features import add_month, build_feature_rows, feature_names

_BASELINE_NAME = "yoy_or_prev_month"


@dataclass
class TrainOutcome:
    model_id: str
    version: str
    dataset_id: str
    artifact_path: str
    observed_end_month: int
    forecast_month: int
    features: list[str]
    metrics: dict[str, float]
    baseline: dict[str, Any]
    eval_period: dict[str, int]
    train_rows: int
    trained_at: str


def _artifact_path(model_id: str, version: str) -> Path:
    return get_settings().model_artifact_dir / model_id / f"{version}.json"


def train(model_id: str, dataset_id: str, version: str) -> TrainOutcome:
    """학습 + 3개월 순차 평가 + 아티팩트 기록. 선행 관측 부족 시 `InsufficientData`."""
    dataset = snapshot.load_dataset(dataset_id)
    expected_model_id = snapshot.TARGETS.get(dataset.target)
    if expected_model_id != model_id:
        raise RealdataError(
            f"model_id {model_id!r}가 dataset target {dataset.target!r}(기대 {expected_model_id!r})와 맞지 않습니다."
        )

    observed_months = {r["base_ym"] for r in dataset.rows}
    evaluate.require_min_observed_months(observed_months)

    observed_end_month = dataset.observed_to
    names = feature_names(dataset)
    feature_rows = build_feature_rows(dataset)

    train_cutoff = add_month(observed_end_month, -3)
    train_rows = [r for r in feature_rows if r["base_ym"] <= train_cutoff]
    if not train_rows:
        raise InsufficientData(
            f"학습 표본이 없습니다 — base_ym <= {train_cutoff}(observed_end_month-3) 구간에 유효한 관측이 없습니다."
        )

    eval_months = set(evaluate.eval_months(observed_end_month))
    eval_rows = [r for r in feature_rows if r["base_ym"] in eval_months]
    if not eval_rows:
        raise InsufficientData(
            f"평가 표본이 없습니다 — {sorted(eval_months)} 구간에 유효한 관측이 없습니다."
        )

    x_train = [[r[name] for name in names] for r in train_rows]
    y_train = [r["y"] for r in train_rows]
    # 승급 판정이 원척도 MAE이므로 학습도 MAE에 맞춘다(v0.3) — 최소제곱은 조건부 평균을 겨냥해
    # 규모가 큰 소수 행정동이 공유 기울기를 지배했다. 산출물 형식은 이전과 같다.
    model = RidgeRegressor.fit_absolute_error(x_train, y_train, ridge_lambda=1.0, feature_names=names)

    x_eval = [[r[name] for name in names] for r in eval_rows]
    y_eval = [r["y"] for r in eval_rows]
    y_pred = model.predict(x_eval)
    metrics = evaluate.metric_bundle(y_eval, y_pred)

    baseline_pred = [r["y_yoy"] if r["has_yoy"] else r["y_lag1"] for r in eval_rows]
    baseline_metrics = evaluate.metric_bundle(y_eval, baseline_pred)

    eval_period = {"from": min(eval_months), "to": max(eval_months), "n": len(eval_rows)}
    forecast_month = add_month(observed_end_month, 1)
    trained_at = datetime.now(timezone.utc).isoformat()

    model_dict = model.to_dict()
    artifact = {
        "model_id": model_id,
        "version": version,
        "dataset_id": dataset_id,
        "content_hash": dataset.content_hash,
        "features": names,
        "coef": model_dict["coefficients"],
        "intercept": model_dict["intercept"],
        "train_means": model_dict["means"],
        "train_stds": model_dict["stds"],
        "lambda": model_dict["ridge_lambda"],
        "eval": {"period": eval_period, "metrics": metrics, "baseline": {"name": _BASELINE_NAME, **baseline_metrics}},
        "observed_end_month": observed_end_month,
        "forecast_month": forecast_month,
        "train_rows": len(train_rows),
        "trained_at": trained_at,
    }

    path = _artifact_path(model_id, version)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")

    return TrainOutcome(
        model_id=model_id,
        version=version,
        dataset_id=dataset_id,
        artifact_path=str(path),
        observed_end_month=observed_end_month,
        forecast_month=forecast_month,
        features=names,
        metrics=metrics,
        baseline={"name": _BASELINE_NAME, **baseline_metrics},
        eval_period=eval_period,
        train_rows=len(train_rows),
        trained_at=trained_at,
    )


def load_artifact(model_id: str, version: str) -> dict[str, Any]:
    """`data/models/<model_id>/<version>.json` 로드."""
    path = _artifact_path(model_id, version)
    if not path.exists():
        raise RealdataError(f"아티팩트를 찾을 수 없습니다: {path}")
    doc: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return doc


def predict(artifact: dict[str, Any], feature_rows: list[dict[str, Any]]) -> list[float]:
    """아티팩트만으로 재계산(재로딩 검증) — feature_rows는 `artifact['features']` 순서로 값을 읽는다."""
    model = RidgeRegressor(
        coefficients=list(artifact["coef"]),
        intercept=artifact["intercept"],
        means=list(artifact["train_means"]),
        stds=list(artifact["train_stds"]),
        feature_names=list(artifact["features"]),
        ridge_lambda=artifact["lambda"],
    )
    names = artifact["features"]
    rows = [[row[name] for name in names] for row in feature_rows]
    return model.predict(rows)
