"""SQLite 영속화 — 사용자 등록 소스·모델 버전·재학습 실행 이력.

in-memory 상태를 대체해 재시작에도 유지되고 멀티워커에서도 일관되게 한다(단일 파일 SQLite,
WAL 모드). 시드(mock_data.json)와 EventBus debounce는 대상 아님(debounce는 프로세스 국소).
값은 JSON 문자열로 저장한다.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
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
        CREATE TABLE IF NOT EXISTS pipelines (
            id TEXT PRIMARY KEY,
            definition_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS model_artifacts (
            model_id TEXT NOT NULL,
            version TEXT NOT NULL,
            artifact_json TEXT NOT NULL,
            PRIMARY KEY (model_id, version)
        );
        """
    )
    conn.commit()
    apply_rd_migrations(conn)


# ── 실데이터 연계(rd_*) 마이그레이션 규약 ────────────────────
# 추가 전용: 새 마이그레이션은 RD_MIGRATIONS에 (id, SQL문 목록)을 덧붙인다.
# 기존 rd_* 테이블에 컬럼을 추가할 때는 `PRAGMA table_info(table)`로 존재 여부를
# 확인한 뒤 `ALTER TABLE ... ADD COLUMN`만 실행한다. DROP·데이터 재작성은 금지.
RD_MIGRATIONS: list[tuple[str, list[str]]] = [
    (
        "rd_0001_init",
        [
            """
            CREATE TABLE IF NOT EXISTS rd_datasets (
                dataset_id TEXT PRIMARY KEY,
                spec_json TEXT NOT NULL,
                quality_json TEXT NOT NULL,
                observed_from INTEGER,
                observed_to INTEGER,
                row_count INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                file_path TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS rd_training_jobs (
                job_id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                state TEXT NOT NULL,
                requested_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                error TEXT,
                candidate_version TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS rd_model_candidates (
                model_id TEXT NOT NULL,
                version TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                artifact_path TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                baseline_json TEXT NOT NULL,
                status TEXT NOT NULL,
                decided_at TEXT,
                decided_by TEXT,
                note TEXT,
                PRIMARY KEY (model_id, version)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS rd_active_models (
                model_id TEXT PRIMARY KEY,
                version TEXT NOT NULL,
                applied_at TEXT NOT NULL,
                previous_version TEXT
            )
            """,
        ],
    ),
]


def apply_rd_migrations(conn: sqlite3.Connection) -> None:
    """rd_* 마이그레이션 적용 — 적용 이력은 rd_schema_migrations에 기록, 기존 4테이블은 무변경."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS rd_schema_migrations (id TEXT PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    conn.commit()
    applied = {row["id"] for row in conn.execute("SELECT id FROM rd_schema_migrations").fetchall()}
    for migration_id, statements in RD_MIGRATIONS:
        if migration_id in applied:
            continue
        for statement in statements:
            conn.execute(statement)
        conn.execute(
            "INSERT INTO rd_schema_migrations (id, applied_at) VALUES (?, ?)",
            (migration_id, datetime.now(timezone.utc).isoformat()),
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


# ── 학습 아티팩트 (승급된 버전의 실측 지표·파일 경로) ───────
def set_model_artifact(model_id: str, version: str, artifact: dict[str, Any]) -> None:
    """승급된 버전의 아티팩트 메타를 기록. 모델 가중치는 로컬 파일에 있고 여기엔 경로만 둔다."""
    _conn().execute(
        "INSERT INTO model_artifacts (model_id, version, artifact_json) VALUES (?, ?, ?) "
        "ON CONFLICT(model_id, version) DO UPDATE SET artifact_json = excluded.artifact_json",
        (model_id, version, json.dumps(artifact, ensure_ascii=False)),
    )
    _conn().commit()


def get_model_artifact(model_id: str, version: str) -> dict[str, Any] | None:
    """특정 버전의 아티팩트 메타 — 없으면 None."""
    row = _conn().execute(
        "SELECT artifact_json FROM model_artifacts WHERE model_id = ? AND version = ?",
        (model_id, version),
    ).fetchone()
    return json.loads(row["artifact_json"]) if row else None


# ── 재학습 실행 이력 ────────────────────────────────────────
def append_run(run: dict[str, Any]) -> None:
    _conn().execute("INSERT INTO runs (run_json) VALUES (?)", (json.dumps(run, ensure_ascii=False),))
    _conn().commit()


def list_runs() -> list[dict[str, Any]]:
    rows = _conn().execute("SELECT run_json FROM runs ORDER BY seq").fetchall()
    return [json.loads(r["run_json"]) for r in rows]


def get_run(run_id: str) -> dict[str, Any] | None:
    """실행 하나를 run_id로. 이력이 작아 파이썬에서 훑는다 — 인덱스가 필요해지면 컬럼으로 승격."""
    for run in reversed(list_runs()):
        if run.get("run_id") == run_id:
            return run
    return None


# ── ML 파이프라인 정의 ──────────────────────────────────────
def add_pipeline(pipeline: dict[str, Any]) -> None:
    """등록 — 같은 id가 이미 있으면 sqlite3.IntegrityError."""
    _conn().execute(
        "INSERT INTO pipelines (id, definition_json, created_at) VALUES (?, ?, ?)",
        (
            pipeline["id"],
            json.dumps(pipeline, ensure_ascii=False),
            pipeline.get("created_at") or datetime.now(timezone.utc).isoformat(),
        ),
    )
    _conn().commit()


def get_pipeline(pipeline_id: str) -> dict[str, Any] | None:
    row = _conn().execute(
        "SELECT definition_json FROM pipelines WHERE id = ?", (pipeline_id,)
    ).fetchone()
    return json.loads(row["definition_json"]) if row else None


def list_pipelines() -> list[dict[str, Any]]:
    rows = _conn().execute("SELECT definition_json FROM pipelines ORDER BY created_at, id").fetchall()
    return [json.loads(r["definition_json"]) for r in rows]


def delete_pipeline(pipeline_id: str) -> bool:
    cur = _conn().execute("DELETE FROM pipelines WHERE id = ?", (pipeline_id,))
    _conn().commit()
    return cur.rowcount > 0
