"""표준 SQL / MQL 생성 — 프론트 dataopsApi.js(buildSql·buildMql·buildQuery) 로직 포팅.

동일 요청 구성이 저장소 유형에 따라 SQL 또는 MQL로 변환됨을 보인다.
메타데이터 적재 범위(range)를 Adapter가 자동 주입한다(SQL BETWEEN / MQL $gte·$lte).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

Column = dict[str, Any]
Range = dict[str, Any] | None

_OPS_MQL = {">": "$gt", ">=": "$gte", "<": "$lt", "<=": "$lte", "!=": "$ne"}
_FILTER_RE = re.compile(r"^(\w+)\s*(>=|<=|!=|=|>|<)\s*(.+)$")


@dataclass(frozen=True)
class GeneratedQuery:
    """생성 결과 — 쿼리 언어와 텍스트."""

    lang: str  # "SQL" | "MQL"
    text: str


def _fmt_sql_value(value: Any) -> str:
    return str(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else f"'{value}'"


def _sql_where(range_: Range, filter_expr: str | None) -> str:
    parts: list[str] = []
    if range_:
        parts.append(
            f"{range_['column']} BETWEEN {_fmt_sql_value(range_['from'])} AND {_fmt_sql_value(range_['to'])}"
        )
    if filter_expr:
        parts.append(filter_expr)
    return f" WHERE {' AND '.join(parts)}" if parts else ""


def build_sql(
    *,
    method: str,
    table: str,
    columns: list[Column],
    range_: Range,
    filter_expr: str | None,
    sort: str | None,
    page: int,
    page_size: int,
) -> str:
    """메서드/필터/정렬/페이징으로부터 표준 SQL 생성 (range 자동 주입)."""
    col_list = ", ".join(c["name"] for c in columns)
    where = _sql_where(range_, filter_expr)
    order = f" ORDER BY {sort} DESC" if sort else ""
    offset = (page - 1) * page_size
    limit = f" LIMIT {page_size} OFFSET {offset}"

    if method == "POST":
        placeholders = ", ".join("?" for _ in columns)
        return f"INSERT INTO {table} ({col_list})\n  VALUES ({placeholders});"
    if method == "PUT":
        assigns = ", ".join(f"{c['name']} = ?" for c in columns)
        return f"UPDATE {table}\n  SET {assigns}{where};"
    if method == "PATCH":
        return f"UPDATE {table}\n  SET {columns[0]['name']} = ?{where};"
    if method == "DELETE":
        return f"DELETE FROM {table}{where};"
    return f"SELECT {col_list}\n  FROM {table}{where}{order}{limit};"


def _mongo_filter_of(filter_expr: str | None) -> str | None:
    if not filter_expr:
        return None
    m = _FILTER_RE.match(filter_expr)
    if not m:
        return f"/* 미해석 조건: {filter_expr} */"
    col, op, raw = m.group(1), m.group(2), m.group(3)
    try:
        val: Any = int(raw)
    except ValueError:
        try:
            val = float(raw)
        except ValueError:
            val = f'"{raw.strip(chr(39)).strip(chr(34))}"'
    return f"{col}: {val}" if op == "=" else f"{col}: {{ {_OPS_MQL[op]}: {val} }}"


def _mongo_match(range_: Range, filter_expr: str | None) -> str:
    parts: list[str] = []
    if range_:
        parts.append(
            f"{range_['column']}: {{ $gte: {json.dumps(range_['from'])}, $lte: {json.dumps(range_['to'])} }}"
        )
    f = _mongo_filter_of(filter_expr)
    if f:
        parts.append(f)
    return f"{{ {', '.join(parts)} }}"


def build_mql(
    *,
    method: str,
    collection: str,
    columns: list[Column],
    range_: Range,
    filter_expr: str | None,
    sort: str | None,
    page: int,
    page_size: int,
) -> str:
    """문서형 저장소용 MQL 생성."""
    match = _mongo_match(range_, filter_expr)
    doc_body = "{ " + ", ".join(f"{c['name']}: <{c['type']}>" for c in columns) + " }"
    sort_seg = f".sort({{ {sort}: -1 }})" if sort else ""
    skip = (page - 1) * page_size

    if method == "POST":
        return f"db.{collection}.insertOne(\n  {doc_body}\n);"
    if method == "PUT":
        return f"db.{collection}.updateMany(\n  {match},\n  {{ $set: {doc_body} }}\n);"
    if method == "PATCH":
        c0 = columns[0]
        return f"db.{collection}.updateMany(\n  {match},\n  {{ $set: {{ {c0['name']}: <{c0['type']}> }} }}\n);"
    if method == "DELETE":
        return f"db.{collection}.deleteMany({match});"
    return f"db.{collection}.find(\n  {match}\n){sort_seg}.skip({skip}).limit({page_size});"


def build_count_sql(*, table: str, range_: Range, filter_expr: str | None) -> str:
    """총 행수 조회 SQL — 페이징 total 을 실제 저장소에서 얻을 때 쓴다.

    본 조회와 같은 `_sql_where` 를 쓰므로 range·filter 범위가 어긋나지 않는다.
    """
    return f"SELECT COUNT(*) AS total FROM {table}{_sql_where(range_, filter_expr)};"


def _mongo_value(raw: str) -> Any:
    """MQL 값 문자열을 파이썬 값으로 — 표시용 `_mongo_filter_of` 와 같은 규칙."""
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw.strip("'").strip('"')


def build_mongo_query(*, range_: Range, filter_expr: str | None) -> dict[str, Any]:
    """pymongo 에 그대로 넘기는 필터 dict.

    표시용 MQL 문자열(`build_mql`)과 달리 이쪽은 드라이버 호출용이다. 문자열을 다시
    파싱하지 않고 같은 입력(range·filter)에서 각각 만들어 파서 이중화를 피한다.
    """
    query: dict[str, Any] = {}
    if range_:
        query[range_["column"]] = {"$gte": range_["from"], "$lte": range_["to"]}
    if filter_expr:
        match = _FILTER_RE.match(filter_expr.strip())
        if match is None:  # safety 가 이미 막지만, 방어적으로 무시한다.
            return query
        column, op, raw = match.group(1), match.group(2), match.group(3)
        value = _mongo_value(raw)
        query[column] = value if op == "=" else {_OPS_MQL[op]: value}
    return query


def is_document_store(schema: dict[str, Any]) -> bool:
    """문서형(NoSQL) 저장소 여부 — MQL 생성 대상."""
    return "MongoDB" in str(schema.get("source", ""))


def build_query(
    *,
    method: str,
    schema: dict[str, Any],
    filter_expr: str | None,
    sort: str | None,
    page: int,
    page_size: int,
) -> GeneratedQuery:
    """저장소 유형에 맞춰 SQL 또는 MQL 산출."""
    common = dict(
        method=method,
        columns=schema["columns"],
        range_=schema.get("range"),
        filter_expr=filter_expr,
        sort=sort,
        page=page,
        page_size=page_size,
    )
    if is_document_store(schema):
        return GeneratedQuery("MQL", build_mql(collection=schema["object"], **common))
    return GeneratedQuery("SQL", build_sql(table=schema["object"], **common))
