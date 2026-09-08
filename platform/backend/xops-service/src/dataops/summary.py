"""Overview 롤업 — 카탈로그 메타데이터만으로 소스 수·아카이브 행수를 집계한다.

신규 저장소 쿼리를 하지 않는다. 행수는 카탈로그의 `archive.rows` 메타데이터이고, 실 저장소
연결 여부는 DSN 설정만 보고 판별한다(연결 시도·요청 왕복 없음). 따라서 DB가 내려가 있어도
이 롤업은 항상 응답한다 — 값의 출처는 기존 DataOps 봉투와 같은 `source_kind`
(`database`|`in-memory`)로 드러나고, 프론트는 이 필드로 실데이터/시드를 구분한다.
"""

from __future__ import annotations

from typing import Any

from src.core.settings import get_settings
from src.dataops.adapters import adapter_of, dsn_for
from src.dataops.catalog import get_catalog
from src.mlops.orchestration.registry import get_registry

# Overview 'F1-score' 카드가 보는 모델 — 프론트 POPULATION_MODEL_ID 와 같은 값.
OVERVIEW_MODEL_ID = "population-forecast"

DATABASE = "database"
IN_MEMORY = "in-memory"


def _archive_rows(schema: dict[str, Any]) -> int:
    """카탈로그 메타데이터의 적재 행수 — 숫자가 아니거나 없으면 0.

    사용자 등록 소스는 등록 시 행수를 받지 않으므로 0이다. SQLite 에 저장된 JSON 도
    신뢰 경계 밖이라 타입을 확인한다(도넛 합계가 TypeError 로 500 이 되지 않게).
    """
    rows = (schema.get("archive") or {}).get("rows")
    if isinstance(rows, bool) or not isinstance(rows, (int, float)):
        return 0
    return max(0, int(rows))


def _source_kind(schema: dict[str, Any]) -> str:
    """이 소스가 실 저장소로 라우팅되는지 — DSN 설정만 보고 판별한다."""
    return DATABASE if dsn_for(schema) else IN_MEMORY


def _rollup(schema: dict[str, Any]) -> dict[str, Any]:
    """소스 1건의 롤업 — 도넛(라벨·행수)과 출처 표기(db_adapter·source_kind)에 필요한 만큼."""
    archive = schema.get("archive") or {}
    return {
        "id": schema.get("id"),
        "label": schema.get("label") or schema.get("id"),
        "source": schema.get("source"),
        "db_adapter": adapter_of(schema),
        "archive_rows": _archive_rows(schema),
        "storage_tier": archive.get("tier"),
        "loaded_at": archive.get("loaded_at"),
        "source_kind": _source_kind(schema),
        "user_registered": bool(schema.get("user_registered")),
    }


def _model_snapshot(model_id: str = OVERVIEW_MODEL_ID) -> dict[str, Any] | None:
    """현행 운영 버전과 F1 — /orchestration/models 와 같은 출처를 그대로 합성한다."""
    for model in get_registry().models():
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


def build_overview_summary() -> dict[str, Any]:
    """Overview 지표 카드·아카이브 도넛용 롤업 (공개 메타데이터, 신규 저장소 쿼리 없음)."""
    settings = get_settings()
    sources = [_rollup(s) for s in get_catalog().list_sources()]
    # 하나라도 실 DSN 이 설정돼 있으면 실 저장소 평면으로 본다. 개별 판정은 sources[].source_kind.
    live = any(s["source_kind"] == DATABASE for s in sources)
    return {
        "status": 200,
        "method": "GET",
        "endpoint": f"{settings.api_prefix}/overview/summary",
        "dataops_version": settings.dataops_version,
        "source_kind": DATABASE if live else IN_MEMORY,
        "source_count": len(sources),
        "archive_rows_total": sum(s["archive_rows"] for s in sources),
        "sources": sources,
        "model": _model_snapshot(),
    }
