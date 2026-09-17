"""실측 모니터링 소스 — SQLite 실행 이력·학습 아티팩트에서만 값을 읽는다.

`mock_data.json` 시드는 여기서 읽지 않는다. 실측이 없으면 `None`을 돌려 호출자가 빈 상태를
그대로 노출하게 한다(0이나 시드로 메우면 없는 값이 실측치로 둔갑한다).

읽는 곳은 두 군데다.

- `runs` 테이블 — 재학습 실행 기록. `training.source == "trained"` 인 실행의
  `candidate_metrics` 가 우리 평가 프로토콜(LOO)로 **측정된** 6대 지표다.
- 학습 아티팩트 JSON(`run.artifact_path`) — 릿지 모델의 표준화 계수. 피처가 표준화돼 있어
  계수 자체가 부호를 보존한 선형 기여도이며, 이것이 SHAP 대체 설명의 실측 근거다.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.core import db
from src.core.logger import get_logger

_logger = get_logger("xops.monitoring.sources")

SERIES = ("accuracy", "f1", "precision", "recall", "mse", "mae")
# 승급되지 않은 실행도 지표는 실측이므로 추이에 남긴다. 걸러내는 것은 학습이 돌지 않은 실행뿐이다.
_TRAINED = "trained"


def _finite(value: Any) -> float | None:
    """유한한 수만 통과시킨다 — None·bool·NaN·문자열은 전부 값 없음으로 본다."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _trained_runs(model_id: str) -> list[dict[str, Any]]:
    """해당 모델의 실측 학습 실행만 시간순으로. `list_runs()` 가 이미 seq 오름차순이다."""
    out: list[dict[str, Any]] = []
    for run in db.list_runs():
        if run.get("model_id") != model_id:
            continue
        training = run.get("training") or {}
        if training.get("source") != _TRAINED:
            continue
        metrics = run.get("candidate_metrics") or {}
        values = {key: _finite(metrics.get(key)) for key in SERIES}
        if any(value is None for value in values.values()):
            continue
        out.append(
            {
                "run_id": run.get("run_id"),
                "state": run.get("state"),
                "metrics": {key: value for key, value in values.items() if value is not None},
                "latency_ms": _finite(training.get("latency_ms")),
            }
        )
    return out


def measured_metrics(model_id: str) -> dict[str, Any] | None:
    """실측 6대 지표 추이 — 실행 이력이 없으면 None.

    x축 라벨은 실행 ID다. 시각 라벨을 만들어 붙이면 없는 관측 주기를 있는 것처럼 보이게 한다.
    """
    runs = _trained_runs(model_id)
    if not runs:
        return None
    history = {key: [run["metrics"][key] for run in runs] for key in SERIES}
    latencies = [run["latency_ms"] for run in runs if run["latency_ms"] is not None]
    return {
        "history": history,
        "latest": {key: values[-1] for key, values in history.items()},
        "labels": [run["run_id"] for run in runs],
        "runs": [{"run_id": run["run_id"], "state": run["state"]} for run in runs],
        # 학습 시 실측한 1건 추론 지연. 없으면 None으로 두고 화면이 '측정 없음'을 쓴다.
        "latency_ms": latencies[-1] if latencies else None,
        "protocol": "leave-one-out",
    }


def _artifact_model(model_id: str, version: str) -> dict[str, Any] | None:
    """승급된 버전의 아티팩트 파일에서 학습된 모델 블록을 읽는다. 없거나 깨졌으면 None."""
    meta = db.get_model_artifact(model_id, version)
    if meta is None:
        return None
    raw_path = meta.get("artifact_path")
    if not raw_path:
        return None
    try:
        payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _logger.warning(f"artifact read failed model={model_id} version={version} reason={exc}")
        return None
    model = payload.get("model")
    return model if isinstance(model, dict) else None


def measured_features(model_id: str, version: str) -> list[dict[str, Any]] | None:
    """학습된 모델의 표준화 계수를 기여도로 — 아티팩트가 없으면 None.

    피처가 학습 시 표준화됐으므로 계수는 그대로 "1 표준편차 변화당 타깃 변화"이며 부호가
    유출(-)·완화(+) 방향을 그대로 나타낸다. |값| 내림차순으로 돌려준다.
    """
    model = _artifact_model(model_id, version)
    if model is None:
        return None
    names = model.get("feature_names")
    coefficients = model.get("coefficients")
    if not isinstance(names, list) or not isinstance(coefficients, list):
        return None
    ranked = [
        {"feature": str(name), "value": round(value, 6)}
        for name, raw in zip(names, coefficients)
        if (value := _finite(raw)) is not None
    ]
    if not ranked:
        return None
    ranked.sort(key=lambda row: abs(row["value"]), reverse=True)
    return ranked
