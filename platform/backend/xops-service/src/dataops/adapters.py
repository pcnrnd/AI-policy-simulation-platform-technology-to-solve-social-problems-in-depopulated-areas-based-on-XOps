"""DB Adapter 선택 — 프론트 dataopsApi.js(pickAdapter·adapterOf) 로직 포팅.

두 층으로 나뉜다.

- **표시용 이름**(`pick_adapter`·`adapter_of`): 어떤 Adapter/쿼리 언어로 라우팅되는지를
  응답 `db_adapter` 에 문자열로 노출한다. 프론트 계약이므로 값이 바뀌지 않는다.
- **실행 어댑터**(`get_adapter`): 실제 저장소에 질의하는 객체를 돌려주는 팩토리.
  DSN이 비어 있거나 드라이버가 없으면 `InMemoryAdapter` 로 degrade하므로,
  DB 없이도 기존 응답 계약이 그대로 유지된다.
"""

from __future__ import annotations

from typing import Any, Protocol

from src.core.logger import get_logger
from src.core.settings import get_settings
from src.dataops.results import ExecutionRequest, ExecutionResult

_logger = get_logger("xops.dataops.adapter")


def pick_adapter(source_id: str) -> str:
    """소스 id 휴리스틱으로 Adapter 결정 (공간=PostGIS, 문서=Mongo, 시계열=Timescale)."""
    if "complaints" in source_id:
        return "MongoAdapter (Document Store)"
    if "spatial" in source_id:
        return "PostGISAdapter (EPSG:4326)"
    if "smartfarm" in source_id:
        return "TimescaleDBAdapter (시계열)"
    if any(k in source_id for k in ("welfare", "industrial", "facility")):
        return "PostgreSQLAdapter"
    return "PostgreSQLAdapter (In-Memory Cache)"


def adapter_of(schema: dict[str, Any]) -> str:
    """스키마의 저장소 유형 문자열로 Adapter 결정 (사용자 등록 소스 대응, 없으면 id 휴리스틱)."""
    src = str(schema.get("source", ""))
    if "MongoDB" in src:
        return "MongoAdapter (Document Store)"
    if "PostGIS" in src:
        return "PostGISAdapter (EPSG:4326)"
    if "TimescaleDB" in src:
        return "TimescaleDBAdapter (시계열)"
    return pick_adapter(str(schema.get("id", "")))


class QueryAdapter(Protocol):
    """실제 저장소 어댑터 seam — 이 인터페이스를 만족하면 교체 가능."""

    name: str

    def execute(self, request: ExecutionRequest) -> ExecutionResult: ...


class InMemoryAdapter:
    """저장소 미연결 시 쓰는 어댑터 — 실행하지 않고 이유만 남긴다.

    서비스는 `executed=False` 를 보고 결정적 스텁 응답(총 행수·affected_rows)을 유지한다.
    """

    def __init__(self, name: str, reason: str = "실 저장소 DSN이 설정되지 않았습니다.") -> None:
        self.name = name
        self._reason = reason

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        del request  # 라우팅 결과만 알리고 질의는 하지 않는다.
        return ExecutionResult.not_executed(self._reason)


def dsn_for(schema: dict[str, Any]) -> str:
    """스키마의 저장소 유형에 맞는 DSN — 없으면 빈 문자열.

    PostgreSQL과 PostGIS는 같은 인스턴스(postgis 이미지)를 쓰므로 `pg_dsn` 을 공유한다.
    """
    settings = get_settings()
    source = str(schema.get("source", ""))
    if "MongoDB" in source:
        return settings.mongo_uri
    if "TimescaleDB" in source:
        return settings.timescale_dsn
    return settings.pg_dsn


def get_adapter(schema: dict[str, Any]) -> QueryAdapter:
    """실행 어댑터 팩토리 — DSN·드라이버가 갖춰졌을 때만 실 백엔드를 준다.

    import 실패(선택 의존성 미설치)와 DSN 부재를 모두 degrade 사유로 취급한다.
    """
    name = adapter_of(schema)
    dsn = dsn_for(schema)
    if not dsn:
        return InMemoryAdapter(name)

    settings = get_settings()
    try:
        from src.dataops.backends import MongoAdapter, SqlAdapter
    except ImportError as exc:  # pragma: no cover - backends 자체는 표준 라이브러리만 씀
        _logger.warning(f"backends import 실패 — In-Memory 로 degrade: {exc}")
        return InMemoryAdapter(name, f"백엔드 모듈을 불러올 수 없습니다: {exc}")

    kwargs = {"timeout": settings.db_timeout_seconds, "max_rows": settings.db_max_rows}
    if "MongoDB" in str(schema.get("source", "")):
        return MongoAdapter(name, dsn, **kwargs)
    return SqlAdapter(name, dsn, **kwargs)
