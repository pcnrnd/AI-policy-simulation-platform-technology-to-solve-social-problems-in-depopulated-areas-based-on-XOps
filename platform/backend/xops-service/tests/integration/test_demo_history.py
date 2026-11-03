"""데모 표시용 운영 이력 재구성 — 새 설치 판정·명세 일치·중단 재개·동시 기동·시간 상한·기본 끔."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient

from main import app
from src.core import db
from src.core.settings import Settings, get_settings
from src.modelops import demo_history as history
from src.modelops import service, store

MOCK = {"X-Data-Mode": "mock"}
FAR = 10**9


@pytest.fixture()
def fresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """빈 SQLite·모델 경로 — 다른 테스트가 만든 mock 기록과 섞이지 않게 한다(새 설치)."""
    settings = get_settings()
    monkeypatch.setattr(settings, "db_path", tmp_path / "xops.db")
    monkeypatch.setattr(settings, "model_artifact_dir", tmp_path / "models")
    previous = getattr(db._local, "conn", None)
    db._local.conn = None
    db.init_db()
    store.init_db()
    yield tmp_path
    if db._local.conn is not None:
        db._local.conn.close()
    db._local.conn = previous


def _mock_rows() -> int:
    return db._conn().execute("SELECT COUNT(*) FROM mo_assets WHERE mode='mock'").fetchone()[0]


def _versions(key: str) -> list[str]:
    ids = history.owned_ids()
    return [ids[f"{key}:run{run}:version"] for run in range(history.RUNS)]


def test_flag_defaults_off_and_parses_env_values() -> None:
    assert Settings().demo_history is False
    assert Settings(demo_history="0").demo_history is False
    assert Settings(demo_history="1").demo_history is True


def test_startup_with_flag_off_writes_nothing(fresh: Path) -> None:
    history.startup()
    assert history.read_state() is None
    assert _mock_rows() == 0


def test_fresh_install_builds_manifest(fresh: Path) -> None:
    assert history.ensure(time.monotonic() + 60) == "completed"
    state = history.read_state()
    assert state["status"] == "completed" and state["integrity"]["problems"] == []
    assert state["integrity"]["audit_total"] == 37
    assert state["integrity"]["audit"] == history.EXPECTED_AUDIT
    # 포인터: 남원 생활인구는 3차 → 2차 복원(이전 3차), 나머지는 3차 활성·이전 2차
    by_id = {row["id"]: row for row in service.instances("mock")}
    ids = history.owned_ids()
    for combo in history.COMBOS:
        versions = _versions(combo["key"])
        instance = by_id[ids[f"{combo['key']}:instance"]]
        expected = (versions[1], versions[2]) if combo["rollback"] else (versions[2], versions[1])
        assert (instance["active_version"], instance["previous_version"]) == expected
    assert len(service.instances("mock")) == 4
    assert sum(1 for job in store.list_assets("mock", "job") if job["state"] == "saved") == 12
    assert len(store.list_assets("mock", "version")) == 12
    rollback = [row for row in store.list_assets("mock", "audit") if row["action"] == "version_rolled_back"]
    assert len(rollback) == 1 and rollback[0]["reason"] == history.ROLLBACK_REASON
    # 시각: 모두 기준 시각 이전, 작업마다 created ≤ started ≤ finished
    for job in store.list_assets("mock", "job"):
        assert job["created_at"] <= job["started_at"] <= job["finished_at"] < state["base_time"]
    for row in store.list_assets("mock", "audit"):
        assert row["created_at"] < state["base_time"]
    # 실데이터 모드에는 아무것도 생기지 않는다
    assert store.list_assets("real", "instance") == [] and store.list_assets("real", "audit") == []
    # 다시 불러도 그대로
    before = _mock_rows()
    assert history.ensure(time.monotonic() + 60) == "already-completed"
    assert _mock_rows() == before


def test_timeline_is_fixed_and_spans_twelve_weeks(fresh: Path) -> None:
    history.ensure(time.monotonic() + 60)
    base = history.read_state()["base_time"]
    times = history.timeline(base)
    assert times == history.timeline(base)
    first, last = min(times.values()), max(times.values())
    from datetime import datetime
    span = datetime.fromisoformat(base) - datetime.fromisoformat(first)
    assert 83 <= span.days <= 84 and last < base


def test_existing_mock_records_are_left_alone(fresh: Path) -> None:
    store.save("mock", "dataset", {"id": "user_dataset", "created_at": store.now()}, create=True)
    assert history.ensure(time.monotonic() + 60) == "skipped"
    assert history.read_state()["status"] == "skipped"
    assert _mock_rows() == 1
    assert history.ensure(time.monotonic() + 60) == "already-skipped"


def test_deleted_rows_are_not_restored_after_completion(fresh: Path) -> None:
    history.ensure(time.monotonic() + 60)
    store.delete_pipeline("mock", "namwon-living-monthly", "user")
    assert history.ensure(time.monotonic() + 60) == "already-completed"
    assert all(row["id"] != "namwon-living-monthly" for row in store.list_assets("mock", "pipeline"))


def _interrupt_after(monkeypatch: pytest.MonkeyPatch, steps: int) -> None:
    original, calls = history._step, {"n": 0}

    def stop_at(state: dict[str, Any], name: str, deadline: float, **kwargs: Any) -> bool:
        if name not in state["steps"]:
            calls["n"] += 1
            if calls["n"] > steps:
                raise history.Deadline(name)
        return original(state, name, deadline, **kwargs)

    monkeypatch.setattr(history, "_step", stop_at)


@pytest.mark.parametrize("steps", [0, 1, 4, 5, 9, 26, 35, 39])
def test_resume_after_interruption(fresh: Path, monkeypatch: pytest.MonkeyPatch, steps: int) -> None:
    """파일 생성 후·각 저장 후·활성화 후·DataOps 중간에서 끊겨도 같은 고정 ID로 이어 완성한다."""
    with monkeypatch.context() as patch:
        _interrupt_after(patch, steps)
        with pytest.raises(history.Deadline):
            history.ensure(FAR)
    assert history.read_state()["status"] == "started"
    assert history.ensure(time.monotonic() + 60) == "completed"
    assert history.read_state()["integrity"]["problems"] == []
    # 중복·고아 파일 없음: 학습데이터 4 + 버전 아티팩트 12
    files = [p for p in (fresh / "models" / "modelops" / "mock").rglob("*.json")]
    assert len(files) == 16
    assert len(store.list_assets("mock", "audit")) == 37


def test_interrupted_before_completion_marker_is_not_marked_complete(fresh: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(history, "verify", lambda base: {"problems": ["simulated"]})
        with pytest.raises(history.Conflict):
            history.ensure(FAR)
    assert history.read_state()["status"] == "started"
    assert history.ensure(time.monotonic() + 60) == "completed"


def test_pointer_saved_but_audit_missing_only_adds_audit(fresh: Path) -> None:
    history.ensure(FAR)
    ids = history.owned_ids()
    audit_id, step = ids["shinan-change-sim:run2:activated:audit"], "shinan-change-sim:run2:activate"
    history._delete_owned("audit", audit_id)
    state = history.read_state()
    history._write_state({**state, "status": "started", "steps": [s for s in state["steps"] if s != step]})
    instance_before = store.get("mock", "instance", ids["shinan-change-sim:instance"])
    assert history.ensure(FAR) == "completed"
    assert store.get("mock", "audit", audit_id)["action"] == "version_activated"
    assert store.get("mock", "instance", ids["shinan-change-sim:instance"]) == instance_before


def test_conflicting_user_record_stops_without_overwrite(fresh: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """재개 중 사용자가 같은 종류·지역·대상 모델을 따로 만들었으면 덮어쓰지 않고 멈춘다."""
    with monkeypatch.context() as patch:
        _interrupt_after(patch, 1)
        with pytest.raises(history.Deadline):
            history.ensure(FAR)
    dataset_id = history.owned_ids()["namwon-living-factor:dataset"]
    dataset = store.get("mock", "dataset", dataset_id)
    user = service.register_instance("mock", {"model_type": "factor_analysis", "dataset_id": dataset_id,
                                              "dataset_hash": dataset["content_hash"]}, "user")
    with pytest.raises(Exception):
        history.ensure(FAR)
    assert history.read_state()["status"] == "started"
    assert store.get("mock", "instance", user["id"])["active_version"] is None


@pytest.mark.parametrize("action", ["dataset_registered", "instance_registered", "pipeline_registered"])
def test_row_saved_but_registration_audit_missing_is_filled(fresh: Path, monkeypatch: pytest.MonkeyPatch, action: str) -> None:
    """등록 행 저장 직후·감사 저장 전에 끊겨도 재개 때 고정 ID 감사만 보충해 완료한다."""
    original = store.audit

    def drop_first(mode: str, act: str, *args: Any, **kwargs: Any) -> None:
        if act == action:
            raise RuntimeError("interrupted")
        original(mode, act, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(store, "audit", drop_first)
        with pytest.raises(RuntimeError):
            history.ensure(FAR)
    assert history.read_state()["status"] == "started"
    assert history.ensure(FAR) == "completed"
    assert history.read_state()["integrity"]["audit"] == history.EXPECTED_AUDIT
    assert len(store.list_assets("mock", "audit")) == 37


def test_deadline_during_training_writes_no_result(fresh: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """학습 도중 시간 상한을 넘기면 결과 파일·버전·감사·완료 상태를 쓰지 않고 멈춘다(작업은 재개 대상)."""
    expired = {"now": False}
    real_check, real_train = history._check_deadline, history.service._train

    def check(name: str, deadline: float) -> None:
        if expired["now"]:
            raise history.Deadline(name)
        real_check(name, deadline)

    def slow_train(*args: Any, **kwargs: Any) -> Any:
        result = real_train(*args, **kwargs)
        expired["now"] = True  # 학습 계산이 끝나는 사이 상한을 넘김
        return result

    with monkeypatch.context() as patch:
        patch.setattr(history, "_check_deadline", check)
        patch.setattr(history.service, "_train", slow_train)
        with pytest.raises(history.Deadline):
            history.ensure(FAR)
    ids = history.owned_ids()
    job = store.get("mock", "job", ids["namwon-living-factor:run0:job"])
    assert job["state"] == "running"
    assert store.list_assets("mock", "version") == []
    assert not any(row["action"] == "training_saved" for row in store.list_assets("mock", "audit"))
    assert not list((fresh / "models" / "modelops" / "mock").glob("modelops_*/*.json"))
    assert history.read_state()["status"] == "started"
    assert history.ensure(FAR) == "completed"


def test_deadline_before_completion_marker_keeps_started(fresh: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_check = history._check_deadline

    def check(name: str, deadline: float) -> None:
        if name == "complete":
            raise history.Deadline(name)
        real_check(name, deadline)

    with monkeypatch.context() as patch:
        patch.setattr(history, "_check_deadline", check)
        with pytest.raises(history.Deadline):
            history.ensure(FAR)
    assert history.read_state()["status"] == "started"
    assert history.ensure(FAR) == "completed"


def test_owned_row_deleted_before_completion_is_rebuilt(fresh: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """완료 표식 전에 지워진 소유 파이프라인은 재개 때 같은 정의로 다시 만들고 완료한다(완료 후 삭제는 미복원)."""
    with monkeypatch.context() as patch:
        patch.setattr(history, "verify", lambda base: {"problems": ["simulated"]})
        with pytest.raises(history.Conflict):
            history.ensure(FAR)
    store.delete_pipeline("mock", "shinan-population-change", "user")
    assert history.ensure(FAR) == "completed"
    pipeline = store.get("mock", "pipeline", "shinan-population-change")
    assert pipeline["instance_id"] == history.owned_ids()["shinan-change-sim:instance"]


def test_corrupted_owned_dataset_blocks_completion(fresh: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(history, "verify", lambda base: {"problems": ["simulated"]})
        with pytest.raises(history.Conflict):
            history.ensure(FAR)
    dataset = store.get("mock", "dataset", history.owned_ids()["namwon-change-sim:dataset"])
    Path(dataset["private_path"]).write_text("{}", encoding="utf-8")
    assert "dataset:namwon-change-sim" in history.verify(history.read_state()["base_time"])["problems"]
    with pytest.raises(Exception):  # 손상 확인(409) 또는 무결성 충돌 — 어느 쪽이든 덮어쓰지 않고 멈춘다
        history.ensure(FAR)
    assert history.read_state()["status"] == "started"


def test_user_source_sample_rows_answer_filter_sort_and_paging(fresh: Path) -> None:
    history.ensure(FAR)
    client = TestClient(app)
    token = client.post("/api/v3/dataops/token/ds_u01_namwon_living_factors", headers=MOCK).json()["access_token"]
    auth = {**MOCK, "Authorization": f"Bearer {token}"}
    body = client.get("/api/v3/dataops/ds_u01_namwon_living_factors",
                      params={"filter": "base_ym >= 202401", "sort": "base_ym", "page_size": 5}, headers=auth).json()
    assert body["source_kind"] == "mock"
    assert body["pagination"]["total"] == 12
    assert [row["base_ym"] for row in body["rows"]] == [202412, 202411, 202410, 202409, 202408]
    assert set(body["rows"][0]) == {"base_ym", "living_population", *history.factor_panel.FEATURES}
    # 샘플은 목록·단건 메타데이터 응답에 싣지 않는다
    assert "sample_rows" not in client.get("/api/v3/dataops/catalog/ds_u01_namwon_living_factors", headers=MOCK).json()


def test_deadline_between_saves_stops_remaining_writes(fresh: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """버전 저장 직후 상한을 넘기면 학습 감사·완료 상태를 쓰지 않는다(저장마다 확인). 재개는 감사만 채운다."""
    expired = {"now": False}
    real_check, real_save = history._check_deadline, store.save

    def check(name: str, deadline: float) -> None:
        if expired["now"]:
            raise history.Deadline(name)
        real_check(name, deadline)

    def save(mode: str, kind: str, body: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        result = real_save(mode, kind, body, **kwargs)
        if kind == "version":
            expired["now"] = True
        return result

    with monkeypatch.context() as patch:
        patch.setattr(history, "_check_deadline", check)
        patch.setattr(store, "save", save)
        with pytest.raises(history.Deadline):
            history.ensure(FAR)
    assert len(store.list_assets("mock", "version")) == 1
    assert not any(row["action"] == "training_saved" for row in store.list_assets("mock", "audit"))
    assert all(job["state"] != "saved" for job in store.list_assets("mock", "job"))
    assert history.ensure(FAR) == "completed"


def test_startup_deadline_writes_nothing_more(fresh: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """기동 경로에서 상한을 넘기면 오류 표식·미완료 작업 정리도 하지 않는다(다음 기동으로)."""
    monkeypatch.setattr(get_settings(), "demo_history", True)
    store.save("real", "job", {"id": "real-job", "instance_id": "r", "state": "running", "created_at": store.now()}, create=True)
    real_check = history._check_deadline

    def check(name: str, deadline: float) -> None:
        if name.endswith(":instance"):
            raise history.Deadline(name)
        real_check(name, deadline)

    monkeypatch.setattr(history, "_check_deadline", check)
    history.startup()
    state = history.read_state()
    assert state["status"] == "started" and "last_error" not in state
    assert store.get("real", "job", "real-job")["state"] == "running"  # 정리는 다음 기동에서


def test_hash_link_mismatch_blocks_completion(fresh: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(history, "verify", lambda base: {"problems": ["simulated"]})
        with pytest.raises(history.Conflict):
            history.ensure(FAR)
    version_id = history.owned_ids()["shinan-living-factor:run2:version"]
    version = store.get("mock", "version", version_id)
    store.save("mock", "version", {**version, "dataset_hash": "0" * 64})
    assert "refs:shinan-living-factor:2" in history.verify(history.read_state()["base_time"])["problems"]
    with pytest.raises(history.Conflict):
        history.ensure(FAR)
    assert history.read_state()["status"] == "started"


def test_redefined_owned_pipeline_is_a_conflict(fresh: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """완료 전 같은 ID·같은 학습데이터로 다른 정의(실행 정책)를 재등록했으면 덮어쓰지 않고 멈춘다."""
    with monkeypatch.context() as patch:
        patch.setattr(history, "verify", lambda base: {"problems": ["simulated"]})
        with pytest.raises(history.Conflict):
            history.ensure(FAR)
    pipeline = store.get("mock", "pipeline", "namwon-living-monthly")
    store.delete_pipeline("mock", "namwon-living-monthly", "user")
    service.register_pipeline("mock", {**{k: pipeline[k] for k in ("id", "name", "model_type", "dataset_id", "dataset_hash", "experiment")},
                                       "trigger_policy": "수동"}, "user")
    with pytest.raises(history.Conflict):
        history.ensure(FAR)
    assert store.get("mock", "pipeline", "namwon-living-monthly")["trigger_policy"] == "수동"


def test_sample_filter_keeps_every_condition(fresh: Path) -> None:
    history.ensure(FAR)
    client = TestClient(app)
    token = client.post("/api/v3/dataops/token/ds_u01_namwon_living_factors", headers=MOCK).json()["access_token"]
    auth = {**MOCK, "Authorization": f"Bearer {token}"}
    body = client.get("/api/v3/dataops/ds_u01_namwon_living_factors",
                      params={"filter": "base_ym >= 202410 AND base_ym >= 202401", "sort": "base_ym"}, headers=auth).json()
    assert [row["base_ym"] for row in body["rows"]] == [202412, 202411, 202410]
    assert body["pagination"]["total"] == 3


@pytest.mark.parametrize("field,value", [("region_id", "shinan"), ("target", "living_population"), ("model_type", "factor_analysis"),
                                         ("mode", "real"), ("source", "real")])
def test_artifact_link_mismatch_blocks_completion(fresh: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: str) -> None:
    """해시가 맞는 아티팩트라도 추론이 보는 연결 정보가 다르면 완료하지 않는다(추론과 같은 검사)."""
    with monkeypatch.context() as patch:
        patch.setattr(history, "verify", lambda base: {"problems": ["simulated"]})
        with pytest.raises(history.Conflict):
            history.ensure(FAR)
    version_id = history.owned_ids()["namwon-change-sim:run2:version"]
    version = store.get("mock", "version", version_id)
    path = Path(version["artifact_path"])
    artifact = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**artifact, field: value}, ensure_ascii=False), encoding="utf-8")
    store.save("mock", "version", {**version, "artifact_hash": hashlib.sha256(path.read_bytes()).hexdigest()})
    assert "artifact:namwon-change-sim:2" in history.verify(history.read_state()["base_time"])["problems"]
    with pytest.raises(history.Conflict):
        history.ensure(FAR)


@pytest.mark.parametrize("hit", ["version_activated", "version_rolled_back", "dataset_registered", "user_source"])
def test_any_write_after_deadline_is_refused_in_startup(fresh: Path, monkeypatch: pytest.MonkeyPatch, hit: str) -> None:
    """기동 경로의 저장 구간 검사 — 어느 저장에서 기한이 지나도 그 뒤 소유 쓰기·오류 표식·정리를 하지 않는다."""
    monkeypatch.setattr(get_settings(), "demo_history", True)
    store.save("real", "job", {"id": "real-job", "instance_id": "r", "state": "running", "created_at": store.now()}, create=True)
    clock = {"expired": False}
    real_monotonic = history.time.monotonic
    monkeypatch.setattr(history.time, "monotonic", lambda: real_monotonic() + (10**6 if clock["expired"] else 0))
    real_audit, real_add = store.audit, history.db.add_user_source

    def audit(mode: str, action: str, *args: Any, **kwargs: Any) -> None:
        real_audit(mode, action, *args, **kwargs)
        if action == hit:
            clock["expired"] = True

    def add_source(schema: dict[str, Any]) -> None:
        real_add(schema)
        if hit == "user_source":
            clock["expired"] = True

    monkeypatch.setattr(store, "audit", audit)
    monkeypatch.setattr(history.db, "add_user_source", add_source)
    history.startup()
    apis_after_expiry = len(history.db.list_built_apis())
    clock["expired"] = False
    state = history.read_state()
    assert state["status"] == "started" and "last_error" not in state
    assert store.get("real", "job", "real-job")["state"] == "running"
    assert apis_after_expiry == 0  # 기한 뒤 DataOps 발급 API까지 가지 않았다
    assert history.ensure(FAR) == "completed"  # 남은 기록은 다음 기동(여기서는 ensure)에서 이어 쓴다


def test_deadline_stops_further_writes(fresh: Path) -> None:
    """잠금 대기로 상한을 다 쓴 경우 — 시작 표식을 포함해 아무것도 쓰지 않는다."""
    with pytest.raises(history.Deadline):
        history.ensure(time.monotonic() - 1)
    assert history.read_state() is None
    assert _mock_rows() == 0


def test_concurrent_startups_build_once(fresh: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_settings(), "demo_history", True)
    errors: list[BaseException] = []

    def boot() -> None:
        try:
            history.startup()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            conn = getattr(db._local, "conn", None)
            if conn is not None:
                conn.close()

    threads = [threading.Thread(target=boot) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert errors == []
    assert history.read_state()["status"] == "completed"
    assert len(store.list_assets("mock", "audit")) == 37
    assert len(store.list_assets("mock", "version")) == 12


def test_two_processes_starting_together_build_once(fresh: Path) -> None:
    """같은 영속 경로를 쓰는 서버 프로세스 2개가 동시에 기동해도 파일 잠금으로 한 번만 만든다."""
    env = {**os.environ, "XOPS_DB_PATH": str(fresh / "xops.db"), "XOPS_MODEL_ARTIFACT_DIR": str(fresh / "models"),
           "XOPS_DEMO_HISTORY": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    code = "from main import app; from src.modelops import demo_history; demo_history.startup()"
    service_root = Path(__file__).resolve().parents[2]
    procs = [subprocess.Popen([sys.executable, "-B", "-c", code], cwd=service_root, env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE) for _ in range(2)]
    for proc in procs:
        _, err = proc.communicate(timeout=90)
        assert proc.returncode == 0, err.decode(errors="replace")[-500:]
    assert history.read_state()["status"] == "completed"
    assert len(store.list_assets("mock", "audit")) == 37
    assert len(store.list_assets("mock", "version")) == 12
    assert len(list((fresh / "models" / "modelops" / "mock").rglob("*.json"))) == 16


def test_startup_failure_keeps_app_running(fresh: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_settings(), "demo_history", True)
    monkeypatch.setattr(history.datasets, "available", lambda mode: (_ for _ in ()).throw(OSError("boom")))
    history.startup()
    state = history.read_state()
    assert state["status"] == "started" and state["last_error"]["kind"] == "OSError"


def test_recover_jobs_keeps_owned_jobs_for_resume(fresh: Path) -> None:
    store.save("mock", "job", {"id": "owned", "instance_id": "i", "state": "running", "created_at": store.now()}, create=True)
    store.save("mock", "job", {"id": "other", "instance_id": "i", "state": "running", "created_at": store.now()}, create=True)
    service.recover_jobs(skip_ids=frozenset({"owned"}))
    assert store.get("mock", "job", "owned")["state"] == "running"
    assert store.get("mock", "job", "other")["state"] == "failed"


def test_dataops_history_is_callable_in_mock_mode_only(fresh: Path) -> None:
    history.ensure(FAR)
    client = TestClient(app)
    token = client.post("/api/v3/dataops/token/ds_u01_namwon_living_factors", headers=MOCK).json()["access_token"]
    auth = {**MOCK, "Authorization": f"Bearer {token}"}
    apis = client.get("/api/v3/dataops/apis", headers=auth).json()
    assert {row["id"] for row in apis} == {api["id"] for api in history.BUILT_APIS}
    catalog = client.get("/api/v3/dataops/catalog", headers=MOCK).json()
    assert {"ds_u01_namwon_living_factors", "ds_u02_shinan_living_factors"} <= {row["id"] for row in catalog}
    for api in history.BUILT_APIS:
        params = {"sort": api["sort"], **({"filter": api["filter"]} if api["filter"] else {})}
        token = client.post(f"/api/v3/dataops/token/{api['source_id']}", headers=MOCK).json()["access_token"]
        response = client.get(f"/api/v3/dataops/{api['source_id']}", params=params,
                              headers={**MOCK, "Authorization": f"Bearer {token}"})
        assert response.status_code == 200, response.text
    real = {"X-Data-Mode": "real", "Authorization": f"Bearer {token}"}
    assert client.get("/api/v3/dataops/apis", headers=real).json() == []
    assert not {"ds_u01_namwon_living_factors"} & {row["id"] for row in client.get("/api/v3/dataops/catalog", headers={"X-Data-Mode": "real"}).json()}


def test_rollback_api_records_reason(fresh: Path) -> None:
    history.ensure(FAR)
    client = TestClient(app)
    token = client.post("/api/v3/dataops/token").json()["access_token"]
    auth = {**MOCK, "Authorization": f"Bearer {token}"}
    ids = history.owned_ids()
    instance_id, previous = ids["shinan-living-factor:instance"], ids["shinan-living-factor:run1:version"]
    response = client.post(f"/api/v3/modelops/instances/{instance_id}/versions/{previous}/rollback",
                           json={"reason": "점검 결과 이전 버전 유지"}, headers=auth)
    assert response.status_code == 200, response.text
    latest = store.list_assets("mock", "audit")[0]
    assert latest["action"] == "version_rolled_back" and latest["reason"] == "점검 결과 이전 버전 유지"
    # 본문 없이도 기존처럼 동작한다(사유 없음)
    current = store.get("mock", "instance", instance_id)
    response = client.post(f"/api/v3/modelops/instances/{instance_id}/versions/{current['previous_version']}/rollback", headers=auth)
    assert response.status_code == 200 and "reason" not in store.list_assets("mock", "audit")[0]
