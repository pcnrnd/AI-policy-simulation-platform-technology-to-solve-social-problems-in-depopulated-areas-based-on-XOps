"""SQLite 영속화 — 사용자 등록 소스·모델 버전·재학습 실행 이력.

in-memory 상태를 대체해 재시작에도 유지되고 멀티워커에서도 일관되게 한다(단일 파일 SQLite,
WAL 모드). 시드(mock_data.json)와 EventBus debounce는 대상 아님(debounce는 프로세스 국소).
값은 JSON 문자열로 저장한다.
"""

from __future__ import annotations

import json
import sqlite3
from functools import lru_cache
from typing import Any

from src.core.settings import get_settings


@lru_cache
def _conn() -> sqlite3.Connection:
    settings = get_settings()
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(settings.db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """테이블 생성 (idempotent)."""
    conn = _conn()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS user_sources (id TEXT PRIMARY KEY, schema_json TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS model_versions (model_id TEXT PRIMARY KEY, version TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS runs (seq INTEGER PRIMARY KEY AUTOINCREMENT, run_json TEXT NOT NULL);
        """
    )
    conn.commit()


# ── 사용자 등록 소스 ────────────────────────────────────────
def add_user_source(schema: dict[str, Any]) -> None:
    _conn().execute(
        "INSERT INTO user_sources (id, schema_json) VALUES (?, ?)",
        (schema["id"], json.dumps(schema, ensure_ascii=False)),
    )
    _conn().commit()


def get_user_source(source_id: str) -> dict[str, Any] | None:
    row = _conn().execute("SELECT schema_json FROM user_sources WHERE id = ?", (source_id,)).fetchone()
    return json.loads(row["schema_json"]) if row else None


def list_user_sources() -> list[dict[str, Any]]:
    rows = _conn().execute("SELECT schema_json FROM user_sources ORDER BY id").fetchall()
    return [json.loads(r["schema_json"]) for r in rows]


def delete_user_source(source_id: str) -> bool:
    cur = _conn().execute("DELETE FROM user_sources WHERE id = ?", (source_id,))
    _conn().commit()
    return cur.rowcount > 0


# ── 모델 버전 오버라이드 ────────────────────────────────────
def get_model_version(model_id: str) -> str | None:
    row = _conn().execute("SELECT version FROM model_versions WHERE model_id = ?", (model_id,)).fetchone()
    return row["version"] if row else None


def set_model_version(model_id: str, version: str) -> None:
    _conn().execute(
        "INSERT INTO model_versions (model_id, version) VALUES (?, ?) "
        "ON CONFLICT(model_id) DO UPDATE SET version = excluded.version",
        (model_id, version),
    )
    _conn().commit()


# ── 재학습 실행 이력 ────────────────────────────────────────
def append_run(run: dict[str, Any]) -> None:
    _conn().execute("INSERT INTO runs (run_json) VALUES (?)", (json.dumps(run, ensure_ascii=False),))
    _conn().commit()


def list_runs() -> list[dict[str, Any]]:
    rows = _conn().execute("SELECT run_json FROM runs ORDER BY seq").fetchall()
    return [json.loads(r["run_json"]) for r in rows]
