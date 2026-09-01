"""query_builder 단위 테스트 — SQL/MQL 생성 계약."""

from __future__ import annotations

from itertools import product

from src.dataops import query_builder, safety
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


def test_builder_filter_regex_covers_every_safe_filter_shape() -> None:
    """safety가 허용한 filter를 builder가 놓쳐 WHERE를 무음 탈락시키지 않는다."""
    identifiers = ("column", "컬럼_1")
    operators = (">=", "<=", "!=", "=", ">", "<")
    values = ("'서울 특별시'", '""', "-42", "0.5", "서울_1")
    whitespace = ("", " ", "\t")

    for identifier, operator, value, left_gap, right_gap in product(
        identifiers, operators, values, whitespace, whitespace
    ):
        expression = f"{identifier}{left_gap}{operator}{right_gap}{value}"
        assert safety._FILTER_RE.fullmatch(expression) is not None, expression
        assert query_builder._FILTER_RE.fullmatch(expression) is not None, expression


# ── 쓰기 실행용 파라미터 바인딩 SQL ──
def test_build_write_sql_post_binds_only_provided_columns() -> None:
    from src.dataops.query_builder import build_write_sql

    statement = build_write_sql(
        method="POST",
        table="tb_resident_movement",
        range_=_SQL_SCHEMA["range"],
        filter_expr=None,
        values={"reg_date": "20260101", "in_flow_count": 7},
    )
    assert statement is not None
    sql, params = statement
    assert sql == "INSERT INTO tb_resident_movement (reg_date, in_flow_count) VALUES (%s, %s);"
    assert params == ["20260101", 7]


def test_build_write_sql_update_appends_range_and_filter_params() -> None:
    from src.dataops.query_builder import build_write_sql

    statement = build_write_sql(
        method="PATCH",
        table="tb_resident_movement",
        range_=_SQL_SCHEMA["range"],
        filter_expr="in_flow_count > 100",
        values={"in_flow_count": 0},
    )
    assert statement is not None
    sql, params = statement
    assert sql == (
        "UPDATE tb_resident_movement SET in_flow_count = %s"
        " WHERE reg_date BETWEEN %s AND %s AND in_flow_count > %s;"
    )
    # filter 값은 리터럴 조립 없이 숫자로 강제 변환돼 바인딩된다.
    assert params == [0, "20210101", "20261231", 100]


def test_build_write_sql_delete_requires_filter() -> None:
    from src.dataops.query_builder import build_write_sql

    common = {"table": "tb_x", "range_": None, "values": None}
    assert build_write_sql(method="DELETE", filter_expr=None, **common) is None
    statement = build_write_sql(method="DELETE", filter_expr="a = 'x'", **common)
    assert statement == ("DELETE FROM tb_x WHERE a = %s;", ["x"])


def test_build_write_sql_returns_none_without_values() -> None:
    from src.dataops.query_builder import build_write_sql

    for method in ("POST", "PUT", "PATCH"):
        assert (
            build_write_sql(
                method=method, table="tb_x", range_=None, filter_expr="a = 1", values={}
            )
            is None
        )
    # PUT/PATCH 는 filter 가 없어도 실행하지 않는다(범위 전체 덮어쓰기 차단).
    assert (
        build_write_sql(method="PUT", table="tb_x", range_=None, filter_expr=None, values={"a": 1})
        is None
    )
