"""SQLite 영속화 단위 테스트 (T8) — 새 인스턴스(=재시작)에서도 상태가 유지되는지."""

from __future__ import annotations

from copy import deepcopy
from typing import Callable

import pytest

from src.core.exceptions import SourceNotFoundError
from src.dataops.catalog import MetadataCatalog
from src.mlops.orchestration.registry import ModelRegistry, _SEED_STORE, next_version

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


def test_model_version_and_runs_persist_across_instances(reset_model: Callable[[str], None]) -> None:
    # 이 테스트의 대상은 영속화다. 후보 지표를 명시해 승급을 결정적으로 만들고(실측 학습은
    # 승급을 보장하지 않는다), 시작 버전을 v3.0-R3으로 고정해 승급 결과 버전을 단언한다.
    reset_model("population-forecast")
    reg1 = ModelRegistry(deepcopy(_SEED_STORE))
    run = reg1.trigger(
        model_id="population-forecast",
        trigger="manual",
        candidate_metrics={"f1": 0.99},
        candidate_latency_ms=120,
    )
    assert run.state == "succeeded"

    reg2 = ModelRegistry(deepcopy(_SEED_STORE))  # 재시작
    entry = next(m for m in reg2.models() if m["model_id"] == "population-forecast")
    assert entry["version"] == "v3.1"
    assert len(reg2.runs()) >= 1
    # 승급 지표가 아티팩트로 영속화돼 재시작 후에도 노출되고, 출처가 함께 기록된다
    # (명시 주입값은 표시는 하되 다음 판정의 실측 기준선으로는 쓰지 않는다).
    assert entry["metrics"]["f1"] == 0.99
    assert entry["metrics_source"] == "explicit"


def test_next_version_increments_minor_and_drops_suffix() -> None:
    # 고정 상수를 쓰면 두 번째 승급이 같은 버전을 재기록해 아티팩트가 덮어써진다.
    assert next_version("v3.0-R3") == "v3.1"
    assert next_version("v3.1") == "v3.2"
    assert next_version("v2.4") == "v2.5"
    assert next_version("v1.9") == "v1.10"
    # 패턴을 벗어나면 최소한 충돌만 피한다.
    assert next_version("release-candidate") == "release-candidate-next"


def test_repeated_promotion_keeps_advancing_versions(reset_model: Callable[[str], None]) -> None:
    """승급마다 버전이 올라가 아티팩트가 서로 덮어써지지 않는다."""
    reset_model("settlement-demand")
    registry = ModelRegistry(deepcopy(_SEED_STORE))
    versions = []
    for _ in range(3):
        run = registry.trigger(
            model_id="settlement-demand",
            trigger="manual",
            candidate_metrics={"f1": 0.99},
            candidate_latency_ms=10,
        )
        assert run.state == "succeeded"
        versions.append(run.active_version)
    assert versions == ["v1.8", "v1.9", "v1.10"]
