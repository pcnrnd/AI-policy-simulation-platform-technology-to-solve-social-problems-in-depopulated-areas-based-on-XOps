"""DataService — 메타검색 → Adapter 선택 → 쿼리 생성 → 안전 검증 → 표준 REST 응답.

프론트 dataopsApi.js(buildApiResponse) 응답 계약과 1:1. 사용자는 저장소를 알지 못해도
API로 CRUD/필터/정렬/페이징을 수행하며, 저장소 접근은 메타데이터로 추상화된다.
"""

from __future__ import annotations

from typing import Any

from src.core.logger import get_logger
from src.core.settings import get_settings
from src.dataops.adapters import adapter_of
from src.dataops.query_builder import build_query
from src.dataops.safety import assert_safe_filter, assert_safe_sort, assert_safe_sql

_logger = get_logger("xops.dataops")

# ponytail: 실제 Adapter 연결 전까지 총 행수는 결정적 스텁(프론트 계약값과 동일). 실 연결 시 COUNT(*)로 대체.
_TOTAL_ROWS = 1248


def _base_response(*, method: str, schema: dict[str, Any], adapter: str, payload: dict[str, Any], query: Any) -> dict[str, Any]:
    settings = get_settings()
    archive = schema.get("archive")
    range_ = schema.get("range")
    return {
        "status": 201 if method == "POST" else 200,
        "method": method,
        "endpoint": f"{settings.api_prefix}/dataops/{schema['id']}",
        "dataops_version": settings.dataops_version,
        "auth": {"authenticated": True, "sub": payload.get("sub"), "scope": payload.get("scope")},
        "db_adapter": adapter,
        "archive_meta": (
            {"storage_tier": archive["tier"], "retention": archive["retention"], "loaded_at": archive["loaded_at"]}
            if archive
            else None
        ),
        "range_scope": ({"column": range_["column"], "from": range_["from"], "to": range_["to"]} if range_ else None),
        "query_language": query.lang,
        "generated_query": query.text,
    }


def _get_extras(schema: dict[str, Any], filter_expr: str | None, sort: str | None, page: int, page_size: int) -> dict[str, Any]:
    total = _TOTAL_ROWS
    return {
        "query": {"filter": filter_expr or None, "sort": sort or None},
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": -(-total // page_size),  # ceil
        },
        "result_rows": max(0, min(page_size, total - (page - 1) * page_size)),
        "sample": {c["name"]: f"<{c['type']}>" for c in schema["columns"]},
    }


class DataService:
    """Data API Builder — 요청을 검증·라우팅하고 표준 REST 응답을 생성."""

    def execute(
        self,
        *,
        method: str,
        schema: dict[str, Any],
        payload: dict[str, Any],
        filter_expr: str | None = None,
        sort: str | None = None,
        page: int = 1,
        page_size: int | None = None,
    ) -> dict[str, Any]:
        settings = get_settings()
        page_size = page_size or settings.default_page_size

        columns = {c["name"] for c in schema["columns"]}
        assert_safe_filter(filter_expr, columns)
        assert_safe_sort(sort, columns)

        query = build_query(
            method=method, schema=schema, filter_expr=filter_expr, sort=sort, page=page, page_size=page_size
        )
        if query.lang == "SQL":
            assert_safe_sql(query.text)

        adapter = adapter_of(schema)
        _logger.info(f"dataops {method} source={schema['id']} adapter={adapter} lang={query.lang}")
        base = _base_response(method=method, schema=schema, adapter=adapter, payload=payload, query=query)

        if method == "GET":
            return {**base, **_get_extras(schema, filter_expr, sort, page, page_size)}
        if method == "DELETE":
            return {**base, "affected_rows": 1 if filter_expr else 0, "message": "Row(s) deleted via virtualized API."}
        return {**base, "affected_rows": 1, "message": f"{method} processed through Data API Builder (storage abstracted)."}
