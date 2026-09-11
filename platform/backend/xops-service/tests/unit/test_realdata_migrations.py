"""rd_* 마이그레이션 단위 테스트 — 추가 전용, 2회 적용해도 idempotent, 기존 4테이블 무변경."""

from __future__ import annotations

import sqlite3

from src.core import db

_EXISTING_TABLES = {"user_sources", "model_versions", "runs", "model_artifacts"}
_RD_TABLES = {
    "rd_schema_migrations",
    "rd_datasets",
    "rd_training_jobs",
    "rd_model_candidates",
    "rd_active_models",
}


def _table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {row["name"] for row in rows}


def test_apply_rd_migrations_is_idempotent_and_keeps_existing_tables() -> None:
    conn = db._conn()

    db.init_db()  # 1회차 (conftest의 앱 import 시 이미 최초 1회는 실행됨)
    tables_after_first = _table_names(conn)

    db.init_db()  # 2회차 — idempotent
    tables_after_second = _table_names(conn)

    assert tables_after_first == tables_after_second
    assert _EXISTING_TABLES <= tables_after_second
    assert _RD_TABLES <= tables_after_second

    applied = conn.execute("SELECT id FROM rd_schema_migrations").fetchall()
    assert [row["id"] for row in applied] == ["rd_0001_init"]


def test_rd_model_candidates_has_composite_primary_key() -> None:
    conn = db._conn()
    columns = conn.execute("PRAGMA table_info(rd_model_candidates)").fetchall()
    pk_columns = [c["name"] for c in sorted((c for c in columns if c["pk"]), key=lambda c: c["pk"])]
    assert pk_columns == ["model_id", "version"]


def test_existing_four_tables_unchanged_by_rd_migration() -> None:
    conn = db._conn()
    expected = {
        "user_sources": {"id", "schema_json"},
        "model_versions": {"model_id", "version"},
        "runs": {"seq", "run_json"},
        "model_artifacts": {"model_id", "version", "artifact_json"},
    }
    for table, expected_columns in expected.items():
        columns = {c["name"] for c in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        assert columns == expected_columns
