"""DB Adapter 선택 — 프론트 dataopsApi.js(pickAdapter·adapterOf) 로직 포팅.

실제 Postgres/PostGIS/Mongo/Timescale 연결은 후속(provider seam). 현재는 In-Memory로
결정적 결과를 반환하되, 어떤 Adapter/쿼리 언어로 라우팅되는지를 계약대로 노출한다.
"""

from __future__ import annotations

from typing import Any, Protocol

from src.dataops.query_builder import GeneratedQuery


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
    """실제 저장소 어댑터 seam — 후속 구현이 이 인터페이스를 만족하면 교체 가능."""

    name: str

    def execute(self, query: GeneratedQuery) -> dict[str, Any]: ...


class InMemoryAdapter:
    """결정적 In-Memory 실행 — 실제 저장소 연결 전 계약 검증용."""

    def __init__(self, name: str) -> None:
        self.name = name

    def execute(self, query: GeneratedQuery) -> dict[str, Any]:
        # 실제 행 반환은 후속(provider seam). 지금은 라우팅/쿼리 메타만 반환.
        return {"adapter": self.name, "query_language": query.lang, "executed": True}
