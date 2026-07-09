"""query_builder 단위 테스트 — SQL/MQL 생성 계약."""

from __future__ import annotations

from src.dataops.query_builder import build_query, is_document_store

_SQL_SCHEMA = {
    "id": "ds_01_resident_registry",
    "source": "RDB · PostgreSQL",
    "object": "tb_resident_movement",
    "range": {"column": "reg_date", "from": "20210101", "to": "20261231"},
    "columns": [{"name": "reg_date", "type": "VARCHAR(8)"}, {"name": "in_flow_count", "type": "INTEGER"}],
}
_MONGO_SCHEMA = {
    "id": "ds_07_civil_complaints",
    "source": "NoSQL · MongoDB",
    "object": "col_civil_complaints",
    "range": {"column": "seq", "from": 25032, "to": 53024},
    "columns": [{"name": "seq", "type": "int"}, {"name": "sentiment_score", "type": "float"}],
}


def test_get_sql_injects_range_and_paging() -> None:
    q = build_query(method="GET", schema=_SQL_SCHEMA, filter_expr=None, sort="reg_date", page=2, page_size=10)
    assert q.lang == "SQL"
    assert "BETWEEN '20210101' AND '20261231'" in q.text
    assert "ORDER BY reg_date DESC" in q.text
    assert "LIMIT 10 OFFSET 10" in q.text


def test_insert_sql_uses_placeholders() -> None:
    q = build_query(method="POST", schema=_SQL_SCHEMA, filter_expr=None, sort=None, page=1, page_size=20)
    assert q.text.startswith("INSERT INTO tb_resident_movement")
    assert q.text.count("?") == 2


def test_delete_sql_with_filter() -> None:
    q = build_query(method="DELETE", schema=_SQL_SCHEMA, filter_expr="in_flow_count > 100", page=1, page_size=20, sort=None)
    assert q.text.startswith("DELETE FROM tb_resident_movement WHERE")
    assert "in_flow_count > 100" in q.text


def test_mongo_get_generates_mql_range() -> None:
    q = build_query(method="GET", schema=_MONGO_SCHEMA, filter_expr=None, sort=None, page=1, page_size=20)
    assert q.lang == "MQL"
    assert "db.col_civil_complaints.find(" in q.text
    assert "seq: { $gte: 25032, $lte: 53024 }" in q.text
    assert ".skip(0).limit(20)" in q.text


def test_mongo_filter_translation() -> None:
    q = build_query(method="GET", schema=_MONGO_SCHEMA, filter_expr="sentiment_score < 0", sort=None, page=1, page_size=5)
    assert "sentiment_score: { $lt: 0 }" in q.text


def test_is_document_store() -> None:
    assert is_document_store(_MONGO_SCHEMA) is True
    assert is_document_store(_SQL_SCHEMA) is False
