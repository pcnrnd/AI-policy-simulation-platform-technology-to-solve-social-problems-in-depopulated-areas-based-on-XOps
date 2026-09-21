"""SQLite 영속화 — 사용자 등록 소스·모델 버전·재학습 실행 이력.

in-memory 상태를 대체해 재시작에도 유지되고 멀티워커에서도 일관되게 한다(단일 파일 SQLite,
WAL 모드). 시드(mock_data.json)와 EventBus debounce는 대상 아님(debounce는 프로세스 국소).
값은 JSON 문자열로 저장한다.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

from src.core.settings import get_settings


# 커넥션은 스레드마다 하나 — uvicorn 스레드풀 워커와 학습 job 백그라운드 스레드가 하나를 공유하면
# 트랜잭션도 공유돼 한 스레드의 commit이 다른 스레드의 미완 쓰기까지 함께 커밋한다.
# 스키마는 커넥션이 아니라 파일에 있으므로 새 스레드의 커넥션은 init_db를 다시 돌릴 필요가 없다.
# ponytail: WAL의 단일 writer 제약은 그대로라 동시 쓰기는 sqlite3 기본 busy timeout(5초)만큼 기다린다.
_local = threading.local()


def _conn() -> sqlite3.Connection:
    conn: sqlite3.Connection | None = getattr(_local, "conn", None)
    if conn is None:
        settings = get_settings()
        settings.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(settings.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        _local.conn = conn
    return conn


# 최초 1회 적재하는 기본 파이프라인 카탈로그 — 모델 레지스트리 3종과 1:1.
# 사용자가 지운 뒤 다시 살아나면 안 되므로 pipelines 테이블이 **없던** 최초 생성 시에만 넣는다.
# 후보 버전은 저장하지 않는다(현행 버전에서 매번 파생 — registry.pipelines 참조).
_SEED_PIPELINES: list[dict[str, Any]] = [
    {
        "id": "PL-POP-RETRAIN-01",
        "name": "인구이동 예측 재학습",
        "model_id": "population-forecast",
        "trigger_policy": "드리프트(PSI > 0.2)·성능 저하(Acc < 0.85) 자동 · 수동",
        "experiment": "EXP-POP-DECLINE-031",
    },
    {
        "id": "PL-VITAL-RETRAIN-02",
        "name": "생활인구 추정 재학습",
        "model_id": "vital-population",
        "trigger_policy": "주간 배치 (매주 월 02:00)",
        "experiment": "EXP-VITAL-POP-012",
    },
    {
        "id": "PL-SETTLE-RETRAIN-03",
        "name": "정주여건 수요예측 재학습",
        "model_id": "settlement-demand",
        "trigger_policy": "수동",
        "experiment": "EXP-SETTLE-DMD-007",
    },
]

# 데모(시드) 파이프라인 id — 데모 표시 OFF 목록에서 빼는 기준.
# 행에 표식 컬럼을 두지 않고 id 집합으로 가른다: `pipelines.id` 가 PRIMARY KEY 라 사용자가
# 같은 id로 등록할 수 없고, 이미 배포된 SQLite 를 건드리지 않아도 된다.
SEED_PIPELINE_IDS: frozenset[str] = frozenset(p["id"] for p in _SEED_PIPELINES)


def init_db() -> None:
    """테이블 생성 (idempotent) + 파이프라인 카탈로그 최초 1회 시드."""
    conn = _conn()
    fresh_pipelines = (
        conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='pipelines'").fetchone() is None
    )
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
        CREATE TABLE IF NOT EXISTS built_apis (
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
    if fresh_pipelines:
        for pipeline in _SEED_PIPELINES:
            add_pipeline(pipeline)
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


# ── 발급 API(Data API 빌드 결과) ────────────────────────────
def upsert_built_api(api: dict[str, Any]) -> dict[str, Any]:
    """빌드된 API 구성 저장 — 같은 id면 덮어쓴다(재빌드가 중복 행을 만들지 않게)."""
    created_at = api.get("created_at") or datetime.now(timezone.utc).isoformat()
    stored = {**api, "created_at": created_at}
    _conn().execute(
        "INSERT INTO built_apis (id, definition_json, created_at) VALUES (?, ?, ?) "
        "ON CONFLICT(id) DO UPDATE SET definition_json = excluded.definition_json",
        (stored["id"], json.dumps(stored, ensure_ascii=False), created_at),
    )
    _conn().commit()
    return stored


def list_built_apis() -> list[dict[str, Any]]:
    """최근 빌드 순."""
    rows = _conn().execute("SELECT definition_json FROM built_apis ORDER BY created_at DESC, id DESC").fetchall()
    return [json.loads(r["definition_json"]) for r in rows]


def delete_built_api(api_id: str) -> bool:
    cur = _conn().execute("DELETE FROM built_apis WHERE id = ?", (api_id,))
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
