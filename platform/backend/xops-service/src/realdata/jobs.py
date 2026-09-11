"""R3-1 실행 잡 — 상태 전이(queued→loading→training→evaluating→saved|failed|cancelled)와
모델별 동시성 락, 재시작 마감(recover_stale_jobs)을 관리한다.

작성자 A의 `src.realdata.{models,snapshot,errors}` 는 이 worktree에 없다(병행 개발).
`_import_models()`/`_import_snapshot()`/`_import_errors()` 간접화를 거쳐 지연 import하고,
테스트는 이 3개 함수를 monkeypatch해 스텁 모듈을 주입한다(성공·InsufficientData·예외 3케이스).
"""

from __future__ import annotations

import secrets
import threading
from datetime import datetime, timezone
from types import ModuleType
from typing import Any

from src.core.db import _conn

ACTIVE_STATES: tuple[str, ...] = ("queued", "loading", "training", "evaluating")


class JobConflict(Exception):
    """같은 model_id에 이미 활성 학습 작업이 있음 — API는 409로 변환한다(R3-1)."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_job_id() -> str:
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"job-{today}-{secrets.token_hex(3)}"


# ── A 모듈 지연 import 간접화 ────────────────────────────────
def _import_models() -> ModuleType:
    from src.realdata import models

    return models


def _import_snapshot() -> ModuleType:
    from src.realdata import snapshot

    return snapshot


def _import_errors() -> ModuleType:
    from src.realdata import errors

    return errors


# ── 모델별 락 ────────────────────────────────────────────────
_locks_guard = threading.Lock()
_model_locks: dict[str, threading.Lock] = {}


def _lock_for(model_id: str) -> threading.Lock:
    with _locks_guard:
        lock = _model_locks.get(model_id)
        if lock is None:
            lock = threading.Lock()
            _model_locks[model_id] = lock
        return lock


# ── SQLite 접근 (rd_training_jobs) ───────────────────────────
def _has_active_job(model_id: str) -> bool:
    placeholders = ",".join("?" for _ in ACTIVE_STATES)
    row = _conn().execute(
        f"SELECT 1 FROM rd_training_jobs WHERE model_id = ? AND state IN ({placeholders}) LIMIT 1",
        (model_id, *ACTIVE_STATES),
    ).fetchone()
    return row is not None


def _insert_job(job_id: str, model_id: str, dataset_id: str) -> None:
    _conn().execute(
        "INSERT INTO rd_training_jobs (job_id, model_id, dataset_id, state, requested_at) "
        "VALUES (?, ?, ?, 'queued', ?)",
        (job_id, model_id, dataset_id, _now_iso()),
    )
    _conn().commit()


def _update_job(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    set_clause = ", ".join(f"{key} = ?" for key in fields)
    _conn().execute(
        f"UPDATE rd_training_jobs SET {set_clause} WHERE job_id = ?",
        (*fields.values(), job_id),
    )
    _conn().commit()


def get_job(job_id: str) -> dict[str, Any] | None:
    """`GET /realdata/training-runs/{job_id}`."""
    row = _conn().execute("SELECT * FROM rd_training_jobs WHERE job_id = ?", (job_id,)).fetchone()
    return dict(row) if row else None


def list_jobs(model_id: str | None = None) -> list[dict[str, Any]]:
    """`GET /realdata/training-runs?model_id=` — 생략하면 전체(최신 우선)."""
    if model_id:
        rows = _conn().execute(
            "SELECT * FROM rd_training_jobs WHERE model_id = ? ORDER BY requested_at DESC",
            (model_id,),
        ).fetchall()
    else:
        rows = _conn().execute("SELECT * FROM rd_training_jobs ORDER BY requested_at DESC").fetchall()
    return [dict(row) for row in rows]


# ── 버전 명명: v<YYYYMMDD>-<content_hash[:6]>, 같은 dataset 재학습은 -r2 접미 ──
def _base_version(dataset_id: str) -> str:
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    content_part = dataset_id[3:9] if dataset_id.startswith("ds-") else dataset_id[:6]
    return f"v{today}-{content_part}"


def _next_version(model_id: str, dataset_id: str) -> str:
    base = _base_version(dataset_id)
    rows = _conn().execute(
        "SELECT version FROM rd_model_candidates WHERE model_id = ?", (model_id,)
    ).fetchall()
    existing = {row["version"] for row in rows}
    if base not in existing:
        return base
    suffix = 2
    while f"{base}-r{suffix}" in existing:
        suffix += 1
    return f"{base}-r{suffix}"


# ── 실행 ─────────────────────────────────────────────────────
def start_training(model_id: str, dataset_id: str) -> str:
    """`POST /realdata/training-runs` — 모델별 락을 non-blocking으로 시도, 실패 시 JobConflict(409)."""
    lock = _lock_for(model_id)
    if not lock.acquire(blocking=False):
        raise JobConflict(f"{model_id}에 이미 활성 학습 작업이 있습니다.")
    try:
        if _has_active_job(model_id):
            # 방어적 재확인 — 정상 경로에선 락이 유일한 게이트지만, 재시작 직후처럼
            # 락이 비어 있는데 DB만 활성 상태로 남는 경우(recover_stale_jobs 이전)를 대비한다.
            raise JobConflict(f"{model_id}에 이미 활성 학습 작업이 있습니다.")
        job_id = _new_job_id()
        _insert_job(job_id, model_id, dataset_id)
    except Exception:
        lock.release()
        raise

    thread = threading.Thread(target=_run_job, args=(job_id, model_id, dataset_id, lock), daemon=True)
    thread.start()
    return job_id


def _run_job(job_id: str, model_id: str, dataset_id: str, lock: threading.Lock) -> None:
    from src.realdata import candidates

    try:
        errors_mod = _import_errors()
        snapshot_mod = _import_snapshot()
        models_mod = _import_models()

        _update_job(job_id, state="loading", started_at=_now_iso())
        try:
            snapshot_mod.load_dataset(dataset_id)
        except errors_mod.DatasetNotFound as exc:
            _update_job(job_id, state="failed", finished_at=_now_iso(), error=f"dataset_not_found: {exc}")
            return
        except errors_mod.RealdataError as exc:
            _update_job(job_id, state="failed", finished_at=_now_iso(), error=str(exc))
            return

        _update_job(job_id, state="training")
        version = _next_version(model_id, dataset_id)
        try:
            outcome = models_mod.train(model_id, dataset_id, version)
        except errors_mod.InsufficientData as exc:
            _update_job(job_id, state="failed", finished_at=_now_iso(), error=f"insufficient_data: {exc}")
            return
        except errors_mod.RealdataError as exc:
            _update_job(job_id, state="failed", finished_at=_now_iso(), error=str(exc))
            return

        _update_job(job_id, state="evaluating")
        candidate_version = candidates.register_candidate(model_id, dataset_id, outcome)
        _update_job(job_id, state="saved", finished_at=_now_iso(), candidate_version=candidate_version)
    except Exception as exc:  # pragma: no cover — 예상 못한 실패의 안전망(상태 미고착 방지)
        _update_job(job_id, state="failed", finished_at=_now_iso(), error=f"unexpected: {exc}")
    finally:
        lock.release()


def recover_stale_jobs() -> int:
    """서버 기동 시 활성 상태(queued/loading/training/evaluating)로 남은 job을
    failed(error="restart")로 마감한다(R3-1). 반환값은 마감한 job 수(로그·테스트용)."""
    placeholders = ",".join("?" for _ in ACTIVE_STATES)
    rows = _conn().execute(
        f"SELECT job_id FROM rd_training_jobs WHERE state IN ({placeholders})", ACTIVE_STATES
    ).fetchall()
    now = _now_iso()
    for row in rows:
        _conn().execute(
            "UPDATE rd_training_jobs SET state = 'failed', error = 'restart', finished_at = ? WHERE job_id = ?",
            (now, row["job_id"]),
        )
    _conn().commit()
    return len(rows)
