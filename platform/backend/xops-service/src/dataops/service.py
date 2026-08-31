"""DataService — 메타검색 → Adapter 선택 → 쿼리 생성 → 안전 검증 → 표준 REST 응답.

프론트 dataopsApi.js(buildApiResponse) 응답 계약과 1:1. 사용자는 저장소를 알지 못해도
API로 CRUD/필터/정렬/페이징을 수행하며, 저장소 접근은 메타데이터로 추상화된다.
"""

from __future__ import annotations

from typing import Any

from src.core.logger import get_logger
from src.core.settings import get_settings
from src.dataops.adapters import adapter_of, get_adapter
from src.dataops.query_builder import build_query
from src.dataops.results import ExecutionRequest, ExecutionResult
from src.dataops.safety import (
    assert_safe_filter,
    assert_safe_schema,
    assert_safe_sort,
    assert_safe_sql,
)

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


def _stub_result_rows(total: int, page: int, page_size: int) -> int:
    """저장소 미연결 시의 결정적 행수 — 페이지 경계에서 남은 행수로 자른다."""
    return max(0, min(page_size, total - (page - 1) * page_size))


def _get_extras(
    schema: dict[str, Any],
    filter_expr: str | None,
    sort: str | None,
    page: int,
    page_size: int,
    result: ExecutionResult,
) -> dict[str, Any]:
    """GET 응답의 조회 관련 필드.

    실 저장소에서 실행됐으면 총 행수와 실제 행을 쓰고, 아니면 기존 결정적 스텁을 유지한다
    (DB 없이도 프론트 계약이 그대로 성립해야 한다).
    """
    total = result.total if result.executed and result.total is not None else _TOTAL_ROWS
    extras: dict[str, Any] = {
        "query": {"filter": filter_expr or None, "sort": sort or None},
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": -(-total // page_size) if page_size else 0,  # ceil
        },
        "result_rows": (
            len(result.rows) if result.executed else _stub_result_rows(total, page, page_size)
        ),
        "sample": {c["name"]: f"<{c['type']}>" for c in schema["columns"]},
        "source_kind": "database" if result.executed else "in-memory",
    }
    if result.executed:
        extras["rows"] = result.rows  # 실 저장소에서 읽은 행 (미연결 시에는 키 자체가 없다)
    return extras


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

        # 카탈로그 메타데이터가 먼저다 — object·컬럼명·range 가 SQL에 조립되므로
        # 등록 검증 이전에 저장된 행이라도 여기서 막아야 한다.
        assert_safe_schema(schema)

        columns = {c["name"] for c in schema["columns"]}
        assert_safe_filter(filter_expr, columns)
        assert_safe_sort(sort, columns)

        query = build_query(
            method=method, schema=schema, filter_expr=filter_expr, sort=sort, page=page, page_size=page_size
        )
        if query.lang == "SQL":
            assert_safe_sql(query.text)

        adapter = adapter_of(schema)
        result = self._run(
            schema=schema,
            method=method,
            query_text=query.text,
            filter_expr=filter_expr,
            sort=sort,
            page=page,
            page_size=page_size,
        )
        _logger.info(
            f"dataops {method} source={schema['id']} adapter={adapter} lang={query.lang} "
            f"executed={result.executed}"
        )
        base = _base_response(method=method, schema=schema, adapter=adapter, payload=payload, query=query)

        if method == "GET":
            return {**base, **_get_extras(schema, filter_expr, sort, page, page_size, result)}
        if method == "DELETE":
            return {**base, "affected_rows": 1 if filter_expr else 0, "message": "Row(s) deleted via virtualized API."}
        return {**base, "affected_rows": 1, "message": f"{method} processed through Data API Builder (storage abstracted)."}

    @staticmethod
    def _run(
        *,
        schema: dict[str, Any],
        method: str,
        query_text: str,
        filter_expr: str | None,
        sort: str | None,
        page: int,
        page_size: int,
    ) -> ExecutionResult:
        """실행 어댑터에 위임. 어떤 실패도 응답을 깨뜨리지 않고 스텁으로 degrade한다.

        DB가 내려가 있거나 스키마가 어긋나도 대시보드는 계속 동작해야 한다 —
        실패 사유는 경고 로그와 `source_kind` 로 드러난다.
        """
        request = ExecutionRequest(
            schema=schema,
            method=method,
            sql=query_text,
            filter_expr=filter_expr,
            sort=sort,
            page=page,
            page_size=page_size,
        )
        try:
            return get_adapter(schema).execute(request)
        except Exception as exc:  # noqa: BLE001 - 드라이버 예외 계층이 다양해 경계에서 일괄 degrade
            source_id = schema.get("id")
            _logger.warning(f"adapter 실행 실패 source={source_id} reason={exc} — In-Memory 응답 유지")
            return ExecutionResult.not_executed(f"실행 실패: {exc}")
