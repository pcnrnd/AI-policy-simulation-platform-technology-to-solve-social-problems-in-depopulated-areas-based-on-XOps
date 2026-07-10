"""SQLite 영속화 단위 테스트 (T8) — 새 인스턴스(=재시작)에서도 상태가 유지되는지."""

from __future__ import annotations

from copy import deepcopy

import pytest

from src.core.exceptions import SourceNotFoundError
from src.dataops.catalog import MetadataCatalog
from src.mlops.orchestration.registry import ModelRegistry, _SEED_STORE

_SCHEMA = {
    "id": "ds_persist_1",
    "label": "영속화 테스트",
    "source": "RDB · PostgreSQL",
    "object": "t_persist",
    "columns": [{"name": "c", "type": "int"}],
}


def test_user_source_survives_new_catalog_instance() -> None:
    MetadataCatalog([]).add(_SCHEMA)
    # 새 인스턴스(재시작 시뮬레이션)에서도 조회됨
    assert MetadataCatalog([]).get("ds_persist_1")["id"] == "ds_persist_1"
    MetadataCatalog([]).remove("ds_persist_1")
    # 삭제 후 새 인스턴스에서 조회 불가
    with pytest.raises(SourceNotFoundError):
        MetadataCatalog([]).get("ds_persist_1")


def test_model_version_and_runs_persist_across_instances() -> None:
    reg1 = ModelRegistry(deepcopy(_SEED_STORE))
    run = reg1.trigger(model_id="population-forecast", trigger="manual", candidate_latency_ms=120)
    assert run.state == "succeeded"

    reg2 = ModelRegistry(deepcopy(_SEED_STORE))  # 재시작
    version = next(m["version"] for m in reg2.models() if m["model_id"] == "population-forecast")
    assert version == "v3.1"
    assert len(reg2.runs()) >= 1
