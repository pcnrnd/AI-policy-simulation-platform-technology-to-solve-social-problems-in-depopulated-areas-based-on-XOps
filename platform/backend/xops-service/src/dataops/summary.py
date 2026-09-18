"""Overview 롤업 — 소스 수와 **실제 적재 행수**를 집계한다.

행수는 저장소에서 직접 센 값(`liveness.live_rows_for`, 60초 TTL 캐시)이다. 예전에는
카탈로그의 `archive.rows` 메타데이터를 그대로 합산했는데, 그 값은 데모용 상수여서 실제와
크게 어긋났다(L0 감사 G3: 표시 2,555,386 대 실제 235,596, 주민등록은 1,248,000 대 60행).
메타데이터를 `source_kind="database"` 라벨과 함께 내보내면 시드를 실측치로 오인하게 된다.

세 상태를 구분한다.

- 센 값(`int`) — 그 소스는 실제로 저장소에 있고 행수를 확인했다. `source_kind="database"`.
- `None` — DSN 부재·드라이버 미설치·연결 실패로 **확인하지 못했다**. `source_kind="in-memory"`.
  합계에서 빠지고 `archive_rows_unknown` 으로 몇 건인지 알린다(0으로 단정하지 않는다).

DB가 내려가 있어도 롤업은 200 을 유지한다 — 전 소스가 `None` 이 될 뿐이다.
"""

from __future__ import annotations

from typing import Any

from src.core.settings import get_settings
from src.dataops.adapters import adapter_of
from src.dataops.catalog import get_catalog
from src.dataops.liveness import live_rows_for
from src.mlops.orchestration.registry import get_registry

# Overview 'F1-score' 카드가 보는 모델 — 프론트 POPULATION_MODEL_ID 와 같은 값.
OVERVIEW_MODEL_ID = "population-forecast"

DATABASE = "database"
IN_MEMORY = "in-memory"


def _rollup(schema: dict[str, Any]) -> dict[str, Any]:
    """소스 1건의 롤업 — 도넛(라벨·행수)과 출처 표기(db_adapter·source_kind)에 필요한 만큼.

    `archive_rows` 는 실제로 센 행수이며, 확인하지 못하면 `None` 이다(0 과 구분).
    `source_kind` 도 DSN 문자열 유무가 아니라 **실제로 세는 데 성공했는지**를 따른다.
    """
    archive = schema.get("archive") or {}
    counted = live_rows_for(schema)
    return {
        "id": schema.get("id"),
        "label": schema.get("label") or schema.get("id"),
        "source": schema.get("source"),
        "db_adapter": adapter_of(schema),
        "archive_rows": counted,
        "storage_tier": archive.get("tier"),
        "loaded_at": archive.get("loaded_at"),
        "source_kind": DATABASE if counted is not None else IN_MEMORY,
        "user_registered": bool(schema.get("user_registered")),
    }


def _model_snapshot(model_id: str = OVERVIEW_MODEL_ID, include_seed: bool = True) -> dict[str, Any] | None:
    """현행 운영 버전과 F1 — /orchestration/models 와 같은 출처를 그대로 합성한다.

    `include_seed=False` 면 지표가 시드인 모델은 아예 나오지 않으므로 None 이 되고, 화면은
    "측정값 없음"을 고른다.
    """
    for model in get_registry().models(include_seed=include_seed):
        if model.get("model_id") != model_id:
            continue
        metrics = model.get("metrics") or {}
        return {
            "model_id": model_id,
            "serving_version": model.get("version"),
            "f1": metrics.get("f1"),
            "metrics_source": model.get("metrics_source"),
        }
    return None


def build_overview_summary(include_seed: bool = True) -> dict[str, Any]:
    """Overview 지표 카드·아카이브 도넛용 롤업 (공개 조회, 소스별 실적재 행수 포함).

    `include_seed=False`(데모 표시 OFF)면 시드 소스와 시드 지표 모델을 빼고 센다 — 예전에는
    `source_count` 가 시드 7종을 포함한 12를 실측처럼 표시했다.
    """
    settings = get_settings()
    sources = [_rollup(s) for s in get_catalog().list_sources(include_seed=include_seed)]
    counted = [s["archive_rows"] for s in sources if s["archive_rows"] is not None]
    # 한 소스라도 실제로 세었으면 실 저장소 평면으로 본다. 개별 판정은 sources[].source_kind.
    return {
        "status": 200,
        "method": "GET",
        "endpoint": f"{settings.api_prefix}/overview/summary",
        "dataops_version": settings.dataops_version,
        "source_kind": DATABASE if counted else IN_MEMORY,
        "source_count": len(sources),
        # 확인한 소스만 더한다 — 확인 못 한 소스를 0으로 치면 합계가 실제보다 작다고 단정하게 된다.
        "archive_rows_total": sum(counted),
        "archive_rows_counted": len(counted),
        "archive_rows_unknown": len(sources) - len(counted),
        "sources": sources,
        "model": _model_snapshot(include_seed=include_seed),
    }
