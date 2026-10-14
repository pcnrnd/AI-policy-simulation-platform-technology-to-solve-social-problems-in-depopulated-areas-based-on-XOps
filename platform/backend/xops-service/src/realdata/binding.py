"""모델과 스냅샷의 입력 범위를 검증한다."""

from typing import Any

from src.realdata.snapshot import TARGETS

MODEL_REGIONS = {model_id: "namwon" for model_id in TARGETS.values()}


def dataset_matches_model(dataset: dict[str, Any], model_id: str) -> bool:
    spec = dataset.get("spec") or {}
    return spec.get("model_id") == model_id and TARGETS.get(spec.get("target")) == model_id
