"""공용 pytest 픽스처."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from typing import Callable

# 테스트는 격리된 임시 SQLite를 사용 — dev DB 오염 방지. app import 전에 설정해야 함.
_TEST_DB = os.path.join(tempfile.gettempdir(), "xops_test.db")
for _suffix in ("", "-wal", "-shm"):
    try:
        os.remove(_TEST_DB + _suffix)
    except FileNotFoundError:
        pass
os.environ["XOPS_DB_PATH"] = _TEST_DB

# 학습 아티팩트도 임시 경로로 격리 — 실제 data/models/ 를 건드리지 않는다.
_TEST_ARTIFACTS = os.path.join(tempfile.gettempdir(), "xops_test_models")
shutil.rmtree(_TEST_ARTIFACTS, ignore_errors=True)
os.environ["XOPS_MODEL_ARTIFACT_DIR"] = _TEST_ARTIFACTS

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402
from src.core import db  # noqa: E402


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


_RUN_TERMINAL_STATES = ("succeeded", "rejected", "rolled_back", "debounced", "failed")


@pytest.fixture()
def await_run() -> Callable[..., dict]:
    """오케스트레이션 접수(202) 응답을 받아 실행이 끝날 때까지 폴링한다.

    `POST /orchestration/events`·`/pipelines/{id}/run`이 백그라운드 스레드로 도는 뒤로는
    종결 상태를 보려면 `GET /orchestration/runs/{run_id}`를 다시 읽어야 한다.
    """

    def _await(client: TestClient, accepted: dict, timeout_s: float = 60.0) -> dict:
        run_id = accepted["run_id"]
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            run = client.get(f"/api/v3/orchestration/runs/{run_id}").json()
            if run.get("state") in _RUN_TERMINAL_STATES:
                return run
            time.sleep(0.05)
        raise AssertionError(f"실행이 제한 시간 안에 끝나지 않았습니다: {run_id}")

    return _await


@pytest.fixture()
def auth_headers(client: TestClient) -> dict[str, str]:
    token = client.post("/api/v3/dataops/token/ds_01_resident_registry").json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def reset_model() -> Callable[[str], None]:
    """모델의 승급 이력(버전 오버라이드·아티팩트)을 지워 '최초 재학습' 상태로 되돌린다.

    실측 학습에서는 승급이 보장되지 않는다: 한 번 승급하면 그 아티팩트의 실측 지표가
    다음 판정의 현행 기준이 되고, 같은 시드로 재학습하면 동점이라 반려된다(의도된 래칫).
    따라서 특정 버전·승급 결과를 단언하는 테스트는 다른 테스트의 승급 여부에 의존하지
    않도록 이 픽스처로 시작 상태를 고정한다.
    """

    def _reset(model_id: str) -> None:
        conn = db._conn()
        conn.execute("DELETE FROM model_versions WHERE model_id = ?", (model_id,))
        conn.execute("DELETE FROM model_artifacts WHERE model_id = ?", (model_id,))
        conn.commit()
        shutil.rmtree(os.path.join(_TEST_ARTIFACTS, model_id), ignore_errors=True)

    return _reset
