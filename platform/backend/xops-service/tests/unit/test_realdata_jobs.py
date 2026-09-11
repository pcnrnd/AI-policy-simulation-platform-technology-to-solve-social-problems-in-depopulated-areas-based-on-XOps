"""R3-1 실행 잡 단위 테스트 — 409 충돌, 상태 전이, 재시작 마감, 3케이스(성공·InsufficientData·예외).

작성자 A의 models/snapshot/errors 모듈은 이 worktree에 없어(병행 개발), jobs.py의
`_import_models`/`_import_snapshot`/`_import_errors`를 monkeypatch로 스텁 대체한다.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from src.core.db import _conn
from src.realdata import jobs
from src.realdata.jobs import JobConflict

_MODEL_ID = "namwon-nonlocal-visitors-next-month"
_DATASET_ID = "ds-abc123def456"


class FakeRealdataError(Exception):
    pass


class FakeDatasetNotFound(FakeRealdataError):
    pass


class FakeInsufficientData(FakeRealdataError):
    pass


class FakeErrors:
    RealdataError = FakeRealdataError
    DatasetNotFound = FakeDatasetNotFound
    InsufficientData = FakeInsufficientData


class FakeDataset:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = [{"base_ym": 202301, "dong_code": "45190250", "y": 10.0}]
        self.content_hash = "fixed-hash"


class FakeSnapshot:
    def __init__(self, *, raise_not_found: bool = False) -> None:
        self._raise_not_found = raise_not_found

    def load_dataset(self, dataset_id: str) -> FakeDataset:
        if self._raise_not_found:
            raise FakeErrors.DatasetNotFound(dataset_id)
        return FakeDataset()


class FakeOutcome:
    def __init__(self, version: str) -> None:
        self.model_id = _MODEL_ID
        self.version = version
        self.dataset_id = _DATASET_ID
        self.artifact_path = "/tmp/fake-artifact.json"
        self.metrics = {"mae": 1.0, "rmse": 1.5, "wape": 0.1}
        self.baseline = {"name": "yoy", "mae": 2.0, "rmse": 2.5, "wape": 0.2}
        self.eval_period = {"from": 202307, "to": 202309, "n": 6}


class FakeModels:
    def __init__(self, *, raise_insufficient: bool = False, raise_error: bool = False) -> None:
        self._raise_insufficient = raise_insufficient
        self._raise_error = raise_error

    def train(self, model_id: str, dataset_id: str, version: str) -> FakeOutcome:
        if self._raise_insufficient:
            raise FakeErrors.InsufficientData("표본이 12개월 미만입니다.")
        if self._raise_error:
            raise FakeErrors.RealdataError("학습 중 알 수 없는 오류")
        return FakeOutcome(version)


def _patch_stubs(monkeypatch: pytest.MonkeyPatch, *, models: FakeModels, snapshot: FakeSnapshot) -> None:
    monkeypatch.setattr(jobs, "_import_models", lambda: models)
    monkeypatch.setattr(jobs, "_import_snapshot", lambda: snapshot)
    monkeypatch.setattr(jobs, "_import_errors", lambda: FakeErrors)


def _wait_terminal(job_id: str, timeout: float = 2.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = jobs.get_job(job_id)
        assert job is not None
        if job["state"] in ("saved", "failed", "cancelled"):
            return job
        time.sleep(0.01)
    raise AssertionError(f"job {job_id}가 시간 내 종결 상태에 도달하지 못했습니다.")


@pytest.fixture(autouse=True)
def _clean_tables() -> None:
    conn = _conn()
    conn.execute("DELETE FROM rd_training_jobs")
    conn.execute("DELETE FROM rd_model_candidates")
    conn.execute("DELETE FROM rd_active_models")
    conn.commit()
    jobs._model_locks.clear()
    yield
    conn.execute("DELETE FROM rd_training_jobs")
    conn.execute("DELETE FROM rd_model_candidates")
    conn.execute("DELETE FROM rd_active_models")
    conn.commit()
    jobs._model_locks.clear()


def test_start_training_success_transitions_to_saved_with_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_stubs(monkeypatch, models=FakeModels(), snapshot=FakeSnapshot())

    job_id = jobs.start_training(_MODEL_ID, _DATASET_ID)
    job = _wait_terminal(job_id)

    assert job["state"] == "saved"
    assert job["error"] is None
    assert job["candidate_version"]
    assert job["started_at"] is not None
    assert job["finished_at"] is not None

    row = _conn().execute(
        "SELECT status FROM rd_model_candidates WHERE model_id = ? AND version = ?",
        (_MODEL_ID, job["candidate_version"]),
    ).fetchone()
    assert row is not None and row["status"] == "candidate"


def test_start_training_insufficient_data_marks_failed_with_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_stubs(monkeypatch, models=FakeModels(raise_insufficient=True), snapshot=FakeSnapshot())

    job_id = jobs.start_training(_MODEL_ID, _DATASET_ID)
    job = _wait_terminal(job_id)

    assert job["state"] == "failed"
    assert job["error"].startswith("insufficient_data")
    assert job["candidate_version"] is None


def test_start_training_unexpected_exception_marks_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_stubs(monkeypatch, models=FakeModels(raise_error=True), snapshot=FakeSnapshot())

    job_id = jobs.start_training(_MODEL_ID, _DATASET_ID)
    job = _wait_terminal(job_id)

    assert job["state"] == "failed"
    assert "알 수 없는 오류" in job["error"]


def test_start_training_dataset_not_found_marks_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_stubs(monkeypatch, models=FakeModels(), snapshot=FakeSnapshot(raise_not_found=True))

    job_id = jobs.start_training(_MODEL_ID, _DATASET_ID)
    job = _wait_terminal(job_id)

    assert job["state"] == "failed"
    assert job["error"].startswith("dataset_not_found")


def test_second_start_training_conflicts_while_first_active(monkeypatch: pytest.MonkeyPatch) -> None:
    """락을 미리 잡아 두어 "활성 작업이 있는 동안"을 결정적으로 재현한다."""
    lock = jobs._lock_for(_MODEL_ID)
    assert lock.acquire(blocking=False)
    try:
        with pytest.raises(JobConflict):
            jobs.start_training(_MODEL_ID, _DATASET_ID)
    finally:
        lock.release()


def test_recover_stale_jobs_marks_active_states_failed_restart() -> None:
    conn = _conn()
    for state in jobs.ACTIVE_STATES:
        conn.execute(
            "INSERT INTO rd_training_jobs (job_id, model_id, dataset_id, state, requested_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (f"job-stale-{state}", _MODEL_ID, _DATASET_ID, state, "2026-01-01T00:00:00+00:00"),
        )
    conn.commit()

    recovered = jobs.recover_stale_jobs()

    assert recovered == len(jobs.ACTIVE_STATES)
    for state in jobs.ACTIVE_STATES:
        job = jobs.get_job(f"job-stale-{state}")
        assert job is not None
        assert job["state"] == "failed"
        assert job["error"] == "restart"
        assert job["finished_at"] is not None


def test_recover_stale_jobs_leaves_terminal_states_untouched() -> None:
    conn = _conn()
    conn.execute(
        "INSERT INTO rd_training_jobs (job_id, model_id, dataset_id, state, requested_at, error) "
        "VALUES ('job-done', ?, ?, 'saved', '2026-01-01T00:00:00+00:00', NULL)",
        (_MODEL_ID, _DATASET_ID),
    )
    conn.commit()

    jobs.recover_stale_jobs()

    job = jobs.get_job("job-done")
    assert job is not None
    assert job["state"] == "saved"
    assert job["error"] is None


def test_version_naming_appends_r2_suffix_for_same_dataset_same_day(monkeypatch: pytest.MonkeyPatch) -> None:
    base = jobs._base_version(_DATASET_ID)
    _conn().execute(
        "INSERT INTO rd_model_candidates (model_id, version, dataset_id, artifact_path, metrics_json, baseline_json, status) "
        "VALUES (?, ?, ?, '', '{}', '{}', 'candidate')",
        (_MODEL_ID, base, _DATASET_ID),
    )
    _conn().commit()

    assert jobs._next_version(_MODEL_ID, _DATASET_ID) == f"{base}-r2"
