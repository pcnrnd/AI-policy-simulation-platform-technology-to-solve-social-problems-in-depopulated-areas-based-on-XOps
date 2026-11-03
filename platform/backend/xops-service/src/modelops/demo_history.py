"""데모 표시(mock 모드)용 운영 이력 재구성 — 새 설치에서 한 번, 과거 12주에 걸친 운영 기록을 만든다.

설정 `XOPS_DEMO_HISTORY`(기본 끔)일 때만 기동 시 동기로 실행한다. 실제 학습·버전·활성화 함수를 같은 데이터로
그대로 호출하고, 시각·ID만 아래 명세(MANIFEST)에서 정한 값을 넘긴다. 판정 순서:
  파일 잠금 → 완료 표식이면 끝 → 시작 표식(미완료)이면 소유 행만 확인하며 이어서 → 표식 없이 mock 모델 기록이
  있으면 건너뜀(표식 남김) → 새 설치면 시작 표식·기준 시각을 첫 쓰기 전에 저장.
완료 뒤 사용자가 지운 소유 행은 되살리지 않는다. 시간 상한(잠금 대기 포함)을 넘기면 쓰기를 멈추고 다음 기동에 잇는다.
실패해도 앱 기동은 막지 않고 단계·사유 종류만 로그에 남긴다(경로·비밀값 제외).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from fastapi import HTTPException

from src.core import db
from src.core.logger import get_logger
from src.core.seed import get_seed
from src.core.settings import get_settings
from src.modelops import datasets, factor_panel, service, store

MODE = "mock"
ACTOR = "operator"
MANIFEST_VERSION = 1
KST = timezone(timedelta(hours=9))
ROLLBACK_REASON = "현장 검토 요청으로 이전 버전 유지"
_STATE_KEY = "state"

# 지역 × 원천 × 모델 종류. 학습 3회, 남원 생활인구만 3차 활성 뒤 2차로 복원 1회.
COMBOS = (
    {"key": "namwon-living-factor", "region": "namwon", "source": "mock_factor_namwon", "model_type": "factor_analysis",
     "pipeline": "namwon-living-monthly", "name": "남원 생활인구 요인분석 정기 학습", "rollback": True},
    {"key": "shinan-living-factor", "region": "shinan", "source": "mock_factor_shinan", "model_type": "factor_analysis",
     "pipeline": "shinan-living-monthly", "name": "신안 생활인구 요인분석 정기 학습", "rollback": False},
    {"key": "namwon-change-sim", "region": "namwon", "source": "mock_namwon", "model_type": "simulation",
     "pipeline": "namwon-population-change", "name": "남원 인구 변화율 시뮬레이션 정기 학습", "rollback": False},
    {"key": "shinan-change-sim", "region": "shinan", "source": "mock_shinan", "model_type": "simulation",
     "pipeline": "shinan-population-change", "name": "신안 인구 변화율 시뮬레이션 정기 학습", "rollback": False},
)
RUNS = 3
TRIGGER_POLICY = "정기 재학습(4주)"
# DataOps: 사용자 소스 2건(위 생활인구 원천의 실제 열 구성) + 그 소스로 호출하는 발급 API 3건.
USER_SOURCES = (("ds_u01_namwon_living_factors", "namwon"), ("ds_u02_shinan_living_factors", "shinan"))
BUILT_APIS = (
    {"id": "api_namwon_living_monthly", "source_id": "ds_u01_namwon_living_factors", "filter": "", "sort": "base_ym"},
    {"id": "api_shinan_living_monthly", "source_id": "ds_u02_shinan_living_factors", "filter": "", "sort": "base_ym"},
    {"id": "api_namwon_living_recent", "source_id": "ds_u01_namwon_living_factors", "filter": "base_ym >= 202401", "sort": "base_ym"},
)
_FEATURE_LABELS = {"youth_ratio": "청년 인구 비율", "aging_index": "고령화 지수", "working_age_pop": "생산연령 인구",
                   "elderly_pop": "고령 인구", "death_cancellations": "사망·말소", "edu_facility_density": "교육 시설 밀도",
                   "medical_facility_density": "의료 시설 밀도", "child_welfare_density": "아동복지 시설 밀도",
                   "facility_capacity": "시설 수용 규모"}
EXPECTED_AUDIT = {"dataset_registered": 4, "instance_registered": 4, "pipeline_registered": 4,
                  "training_saved": 12, "version_activated": 12, "version_rolled_back": 1}


class Conflict(RuntimeError):
    """소유하지 않은 기록과 겹쳐 덮어쓰지 않고 멈춘다."""


class Deadline(RuntimeError):
    """시간 상한 초과 — 쓰기를 멈추고 다음 기동 때 잇는다."""


def _oid(key: str) -> str:
    """명세 기반 고정 ID — 중단 뒤 다시 실행해도 같은 행을 가리킨다."""
    return "modelops_" + hashlib.sha256(f"ops-history-v{MANIFEST_VERSION}:{key}".encode()).hexdigest()[:32]


def manifest_hash() -> str:
    spec = {"version": MANIFEST_VERSION, "combos": COMBOS, "runs": RUNS, "reason": ROLLBACK_REASON,
            "sources": USER_SOURCES, "apis": BUILT_APIS, "audit": EXPECTED_AUDIT}
    return hashlib.sha256(json.dumps(spec, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# ── 시각표: 설치 시각 기준 과거 12주(Asia/Seoul 업무 시간), 한 번 정하면 바꾸지 않는다 ──
def _iso(at: datetime) -> str:
    return at.astimezone(timezone.utc).isoformat()


def timeline(base_time: str) -> dict[str, str]:
    """작업별 시각. created ≤ started ≤ finished ≤ activated, 모두 기준 시각보다 앞선다."""
    installed = datetime.fromisoformat(base_time).astimezone(KST)
    day0 = (installed - timedelta(days=84)).replace(hour=9, minute=30, second=0, microsecond=0)
    times: dict[str, str] = {}
    for index, combo in enumerate(COMBOS):
        key, offset = combo["key"], timedelta(minutes=25 * index)
        times[f"{key}:dataset"] = _iso(day0 + offset)
        times[f"{key}:instance"] = _iso(day0 + timedelta(days=1) + offset)
        times[f"{key}:pipeline"] = _iso(day0 + timedelta(days=1, minutes=6) + offset)
        for run in range(RUNS):
            created = day0 + timedelta(days=7 + 28 * run, hours=1) + timedelta(minutes=40 * index)
            started = created + timedelta(seconds=40)
            finished = started + timedelta(minutes=2, seconds=7 * index + 3 * run)
            times[f"{key}:run{run}:created"] = _iso(created)
            times[f"{key}:run{run}:started"] = _iso(started)
            times[f"{key}:run{run}:finished"] = _iso(finished)
            times[f"{key}:run{run}:activated"] = _iso(finished + timedelta(minutes=18))
        if combo["rollback"]:
            times[f"{key}:rollback"] = _iso(day0 + timedelta(days=70, hours=5, minutes=10))
    for index, (source_id, _) in enumerate(USER_SOURCES):
        times[f"source:{source_id}"] = _iso(day0 + timedelta(days=2, hours=1, minutes=15 * index))
    for index, api in enumerate(BUILT_APIS):
        times[f"api:{api['id']}"] = _iso(day0 + timedelta(days=3, hours=2, minutes=20 * index))
    return times


def owned_ids() -> dict[str, str]:
    """명세가 소유하는 modelops 행 ID 전체(역할 → ID)."""
    ids: dict[str, str] = {}
    for combo in COMBOS:
        key = combo["key"]
        for role in ("dataset", "instance", "dataset:audit", "instance:audit", "pipeline:audit"):
            ids[f"{key}:{role}"] = _oid(f"{key}:{role}")
        for run in range(RUNS):
            for role in ("job", "version", "saved:audit", "activated:audit"):
                ids[f"{key}:run{run}:{role}"] = _oid(f"{key}:run{run}:{role}")
        if combo["rollback"]:
            ids[f"{key}:rollback:audit"] = _oid(f"{key}:rollback:audit")
    return ids


# ── 상태 표식(같은 SQLite) ──
def _init_state() -> None:
    db._conn().execute("CREATE TABLE IF NOT EXISTS ops_history_state (key TEXT PRIMARY KEY, value_json TEXT NOT NULL)")
    db._conn().commit()


def read_state() -> dict[str, Any] | None:
    _init_state()
    row = db._conn().execute("SELECT value_json FROM ops_history_state WHERE key=?", (_STATE_KEY,)).fetchone()
    return json.loads(row[0]) if row else None


def _write_state(state: dict[str, Any]) -> None:
    db._conn().execute("INSERT INTO ops_history_state (key, value_json) VALUES (?, ?) "
                       "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json",
                       (_STATE_KEY, json.dumps(state, ensure_ascii=False)))
    db._conn().commit()


# ── 파일 잠금(같은 영속 경로를 쓰는 여러 프로세스 사이) ──
def lock_path() -> Path:
    return get_settings().db_path.parent / "ops-history.lock"


@contextmanager
def file_lock(deadline: float) -> Iterator[None]:
    path = lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+b")  # noqa: SIM115 — 잠금 수명 동안 열어 둔다
    locked = False
    try:
        while not locked:
            try:
                if os.name == "nt":
                    import msvcrt
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except OSError:
                if time.monotonic() >= deadline:
                    raise Deadline("lock") from None
                time.sleep(0.2)
        yield
    finally:
        if locked:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


# ── 실행 ──
def startup() -> None:
    """기동 훅. 설정이 꺼져 있으면 기존 동작(미완료 작업 정리)만 한다."""
    settings = get_settings()
    if not settings.demo_history:
        service.recover_jobs()
        return
    deadline = time.monotonic() + settings.demo_history_timeout_seconds
    try:
        with file_lock(deadline):
            try:
                result = ensure(deadline)
                get_logger().info(f"운영 이력 재구성: {result}")
            except Exception as exc:  # noqa: BLE001 — 앱 기동을 막지 않는다
                _record_error(exc)
            service.recover_jobs(skip_ids=frozenset(v for k, v in owned_ids().items() if k.endswith(":job")))
    except Deadline:
        # 다른 프로세스가 잠금을 쥐고 있다 — 그쪽이 재구성과 정리를 맡는다.
        get_logger().warning("운영 이력 재구성: 잠금 대기 시간 초과, 이번 기동은 건너뜀")


def _record_error(exc: Exception) -> None:
    stage = getattr(exc, "stage", None) or "unknown"
    kind = type(exc).__name__
    get_logger().warning(f"운영 이력 재구성 중단: 단계={stage} 사유={kind}")
    state = read_state()
    if state is not None and state.get("status") == "started":
        _write_state({**state, "last_error": {"stage": stage, "kind": kind, "at": store.now()}})


def ensure(deadline: float) -> str:
    """판정 순서대로 실행하고 결과(completed·skipped·already-completed·already-skipped)를 돌려준다. 잠금 안에서 부른다."""
    state = read_state()
    if state is not None and state["status"] == "completed":
        return "already-completed"
    if state is not None and state["status"] == "skipped":
        return "already-skipped"
    if state is None:
        if db._conn().execute("SELECT 1 FROM mo_assets WHERE mode=? LIMIT 1", (MODE,)).fetchone():
            _write_state({"status": "skipped", "reason": "existing-mock-records", "at": store.now()})
            return "skipped"
        installed = datetime.now(timezone.utc).replace(second=0, microsecond=0).isoformat()
        state = {"status": "started", "manifest_version": MANIFEST_VERSION, "manifest_hash": manifest_hash(),
                 "base_time": installed, "started_at": store.now(), "steps": []}
        _write_state(state)
    elif state.get("manifest_hash") != manifest_hash():
        raise Conflict("manifest")
    _run(state, deadline)
    integrity = verify(state["base_time"])
    if integrity["problems"]:
        raise Conflict("integrity")
    _write_state({**read_state(), "status": "completed", "completed_at": store.now(), "integrity": integrity})
    return "completed"


def _step(state: dict[str, Any], name: str, deadline: float) -> bool:
    """단계 표식 — 이미 끝난 단계면 False. 시간을 넘겼으면 멈춘다."""
    if name in state["steps"]:
        return False
    if time.monotonic() >= deadline:
        error = Deadline(name)
        error.stage = name  # type: ignore[attr-defined]
        raise error
    return True


def _done(state: dict[str, Any], name: str) -> None:
    state["steps"].append(name)
    _write_state(state)


def _run(state: dict[str, Any], deadline: float) -> None:
    times, ids = timeline(state["base_time"]), owned_ids()
    available = {row["id"]: row for row in datasets.available(MODE)}
    for combo in COMBOS:
        try:
            _run_combo(state, deadline, combo, times, ids, available)
        except (HTTPException, Conflict) as exc:
            exc.stage = combo["key"]  # type: ignore[attr-defined]
            raise
    _run_dataops(state, deadline, times)


def _exists(kind: str, asset_id: str) -> dict[str, Any] | None:
    try:
        return store.get(MODE, kind, asset_id)
    except HTTPException:
        return None


def _run_combo(state: dict[str, Any], deadline: float, combo: dict[str, Any], times: dict[str, str],
               ids: dict[str, str], available: dict[str, dict[str, Any]]) -> None:
    key = combo["key"]
    source = available[combo["source"]]
    dataset_id, instance_id = ids[f"{key}:dataset"], ids[f"{key}:instance"]
    if _step(state, f"{key}:dataset", deadline):
        if _exists("dataset", dataset_id) is None:
            datasets.register(MODE, {"name": source["name"], "source_dataset_id": source["id"], "source_hash": source["content_hash"]},
                              ACTOR, dataset_id=dataset_id, created_at=times[f"{key}:dataset"], audit_id=ids[f"{key}:dataset:audit"])
        _done(state, f"{key}:dataset")
    dataset = store.get(MODE, "dataset", dataset_id)
    body = {"model_type": combo["model_type"], "dataset_id": dataset_id, "dataset_hash": dataset["content_hash"],
            "name": combo["name"].replace(" 정기 학습", "")}
    if _step(state, f"{key}:instance", deadline):
        service.register_instance(MODE, body, ACTOR, instance_id=instance_id, created_at=times[f"{key}:instance"],
                                  audit_id=ids[f"{key}:instance:audit"])
        _done(state, f"{key}:instance")
    if _step(state, f"{key}:pipeline", deadline):
        pipeline = service.register_pipeline(MODE, {**body, "id": combo["pipeline"], "name": combo["name"],
                                                    "trigger_policy": TRIGGER_POLICY, "experiment": "", "id_prefix": "modelops_"},
                                             ACTOR, created_at=times[f"{key}:pipeline"], audit_id=ids[f"{key}:pipeline:audit"])
        if pipeline["instance_id"] != instance_id:
            raise Conflict("pipeline")
        _done(state, f"{key}:pipeline")
    for run in range(RUNS):
        _train(state, deadline, combo, run, times, ids)
        _activate(state, deadline, combo, run, times, ids)
    if combo["rollback"] and _step(state, f"{key}:rollback", deadline):
        _pointer_step(state, key, ids[f"{key}:run2:version"], ids[f"{key}:run1:version"],
                      ids[f"{key}:run1:version"], ids[f"{key}:run2:version"], f"{key}:rollback",
                      lambda: service.activate(MODE, instance_id, ids[f"{key}:run1:version"], ACTOR, rollback=True,
                                               reason=ROLLBACK_REASON, at=times[f"{key}:rollback"],
                                               audit_id=ids[f"{key}:rollback:audit"]),
                      audit=("version_rolled_back", ids[f"{key}:rollback:audit"], times[f"{key}:rollback"],
                             {"reason": ROLLBACK_REASON}))


def _train(state: dict[str, Any], deadline: float, combo: dict[str, Any], run: int,
           times: dict[str, str], ids: dict[str, str]) -> None:
    key, name = combo["key"], f"{combo['key']}:run{run}:train"
    if not _step(state, name, deadline):
        return
    job_id, version_id, audit_id = ids[f"{key}:run{run}:job"], ids[f"{key}:run{run}:version"], ids[f"{key}:run{run}:saved:audit"]
    job, version, audit = _exists("job", job_id), _exists("version", version_id), _exists("audit", audit_id)
    finished = times[f"{key}:run{run}:finished"]
    if version is not None and _artifact_intact(version):
        # 버전까지 저장된 뒤 끊겼다 — 빠진 감사·작업 상태만 채운다.
        if audit is None:
            store.audit(MODE, "training_saved", version["instance_id"], ACTOR, audit_id=audit_id, created_at=finished,
                        job_id=job_id, version_id=version_id, dataset_hash=version["dataset_hash"])
        if job is not None and job["state"] != "saved":
            store.save(MODE, "job", {**job, "state": "saved", "finished_at": finished, "artifact_path": version["artifact_path"]})
    else:
        if audit is not None:  # 소유 감사만 남고 버전이 없거나 손상 — 다시 학습하므로 감사도 다시 쓴다
            _delete_owned("audit", audit_id)
        job = service.train_sync(MODE, combo["pipeline"], ACTOR, job_id=job_id, version_id=version_id,
                                 created_at=times[f"{key}:run{run}:created"], started_at=times[f"{key}:run{run}:started"],
                                 finished_at=finished, audit_id=audit_id)
        if job["state"] != "saved":
            raise Conflict(f"train:{job.get('failed_stage')}")
    _done(state, name)


def _activate(state: dict[str, Any], deadline: float, combo: dict[str, Any], run: int,
              times: dict[str, str], ids: dict[str, str]) -> None:
    key, name = combo["key"], f"{combo['key']}:run{run}:activate"
    if not _step(state, name, deadline):
        return
    instance_id, version_id = ids[f"{key}:instance"], ids[f"{key}:run{run}:version"]
    before = ids[f"{key}:run{run - 1}:version"] if run else None
    before_previous = ids[f"{key}:run{run - 2}:version"] if run >= 2 else None
    at, audit_id = times[f"{key}:run{run}:activated"], ids[f"{key}:run{run}:activated:audit"]
    _pointer_step(state, key, before, before_previous, version_id, before, name,
                  lambda: service.activate(MODE, instance_id, version_id, ACTOR, at=at, audit_id=audit_id),
                  audit=("version_activated", audit_id, at, {}))


def _pointer_step(state: dict[str, Any], key: str, pre_active: str | None, pre_previous: str | None,
                  post_active: str, post_previous: str | None, name: str, apply: Any,
                  audit: tuple[str, str, str, dict[str, Any]]) -> None:
    """포인터 변경 한 단계. 이전 상태면 실행, 이미 바뀐 뒤 감사만 빠졌으면 감사만 채운다, 그 밖은 충돌."""
    instance = store.get(MODE, "instance", owned_ids()[f"{key}:instance"])
    current = (instance.get("active_version"), instance.get("previous_version"))
    action, audit_id, at, extra = audit
    if current == (pre_active, pre_previous):
        apply()
    elif current == (post_active, post_previous):
        if _exists("audit", audit_id) is None:
            store.audit(MODE, action, instance["id"], ACTOR, audit_id=audit_id, created_at=at,
                        version_id=post_active, previous_version=post_previous, **extra)
    else:
        raise Conflict(name)
    _done(state, name)


def _artifact_intact(version: dict[str, Any]) -> bool:
    path = Path(version.get("artifact_path", ""))
    return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == version.get("artifact_hash")


def _delete_owned(kind: str, asset_id: str) -> None:
    db._conn().execute("DELETE FROM mo_assets WHERE mode=? AND kind=? AND id=?", (MODE, kind, asset_id))
    db._conn().commit()


def _user_source(source_id: str, region_id: str, at: str) -> dict[str, Any]:
    region = next(row for row in get_seed()["regions"] if row["id"] == region_id)
    rows = factor_panel.generate(region)
    columns = [{"name": "base_ym", "type": "INTEGER", "description": "기준 연월 (YYYYMM)"}] + [
        {"name": name, "type": "DOUBLE PRECISION", "description": _FEATURE_LABELS[name]}
        for name in factor_panel.FEATURES] + [{"name": "living_population", "type": "DOUBLE PRECISION", "description": "생활인구"}]
    return {
        "id": source_id, "label": f"{region['name'].split()[-1]} 인구 구조·시설 지표", "source": "RDB · PostgreSQL",
        "object": f"tb_living_factors_{region_id}", "description": f"{region['name']} 월별 인구 구조·생활 시설 지표와 생활인구",
        "archive": {"tier": "Warm", "retention": "5년 보존", "loaded_at": at[:10], "rows": len(rows)},
        "lineage": {"origin": "지자체 인구·시설 통계 월별 집계", "version": "v1", "commit": hashlib.sha256(source_id.encode()).hexdigest()[:6]},
        "range": {"column": "base_ym", "from": str(rows[0]["base_ym"]), "to": str(rows[-1]["base_ym"])},
        "tags": ["생활인구", "정형"], "columns": columns,
        "user_registered": True, "mode": MODE, "is_seed": True, "registered_at": at,
    }


def _run_dataops(state: dict[str, Any], deadline: float, times: dict[str, str]) -> None:
    for source_id, region_id in USER_SOURCES:
        if _step(state, f"source:{source_id}", deadline):
            expected = _user_source(source_id, region_id, times[f"source:{source_id}"])
            existing = db.get_user_source(source_id)
            if existing is None:
                db.add_user_source(expected)
            elif existing != expected:
                raise Conflict(f"source:{source_id}")
            _done(state, f"source:{source_id}")
    for api in BUILT_APIS:
        if _step(state, f"api:{api['id']}", deadline):
            expected = _built_api(api, times[f"api:{api['id']}"])
            existing = next((row for row in db.list_built_apis() if row["id"] == api["id"]), None)
            if existing is None:
                db.upsert_built_api(expected)
            elif existing != expected:
                raise Conflict(f"api:{api['id']}")
            _done(state, f"api:{api['id']}")


def _built_api(api: dict[str, Any], at: str) -> dict[str, Any]:
    region_id = dict(USER_SOURCES)[api["source_id"]]
    region = next(row for row in get_seed()["regions"] if row["id"] == region_id)
    return {"id": api["id"], "method": "GET", "source_id": api["source_id"],
            "source_label": f"{region['name'].split()[-1]} 인구 구조·시설 지표", "filter": api["filter"], "sort": api["sort"],
            "page": 1, "page_size": 20, "auth_method": "JWT", "mode": MODE, "created_at": at}


# ── 무결성 확인(완료 표식 직전·배포 확인에서도 쓴다) ──
def verify(base_time: str) -> dict[str, Any]:
    """명세와 저장 상태 대조 — 포인터·소유 감사 건수·참조·해시·시각 순서. problems가 비어야 완료로 본다."""
    ids, times, problems = owned_ids(), timeline(base_time), []
    owned = set(ids.values())
    audits = [row for row in store.list_assets(MODE, "audit") if row["id"] in owned]
    counts = {action: sum(1 for row in audits if row["action"] == action) for action in EXPECTED_AUDIT}
    if counts != EXPECTED_AUDIT:
        problems.append("audit-counts")
    pointers = {}
    for combo in COMBOS:
        key = combo["key"]
        instance = _exists("instance", ids[f"{key}:instance"])
        versions = [ids[f"{key}:run{run}:version"] for run in range(RUNS)]
        expected = (versions[1], versions[2]) if combo["rollback"] else (versions[2], versions[1])
        actual = (instance or {}).get("active_version"), (instance or {}).get("previous_version")
        pointers[key] = "ok" if actual == expected else "mismatch"
        if actual != expected:
            problems.append(f"pointer:{key}")
        for run in range(RUNS):
            version, job = _exists("version", versions[run]), _exists("job", ids[f"{key}:run{run}:job"])
            if version is None or job is None or job["state"] != "saved" or not _artifact_intact(version):
                problems.append(f"version:{key}:{run}")
                continue
            order = [job["created_at"], job["started_at"], job["finished_at"], times[f"{key}:run{run}:activated"]]
            if order != sorted(order) or order[-1] >= base_time:
                problems.append(f"time:{key}:{run}")
    sources = [sid for sid, _ in USER_SOURCES if db.get_user_source(sid) is not None]
    apis = [api["id"] for api in BUILT_APIS if any(row["id"] == api["id"] for row in db.list_built_apis())]
    if len(sources) != len(USER_SOURCES) or len(apis) != len(BUILT_APIS):
        problems.append("dataops")
    return {"audit": counts, "audit_total": sum(counts.values()), "pointers": pointers,
            "user_sources": len(sources), "built_apis": len(apis), "problems": problems}
