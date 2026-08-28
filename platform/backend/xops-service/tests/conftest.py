"""공용 pytest 픽스처."""

from __future__ import annotations

import os
import shutil
import tempfile
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
