"""공용 pytest 픽스처."""

from __future__ import annotations

import os
import tempfile

# 테스트는 격리된 임시 SQLite를 사용 — dev DB 오염 방지. app import 전에 설정해야 함.
_TEST_DB = os.path.join(tempfile.gettempdir(), "xops_test.db")
for _suffix in ("", "-wal", "-shm"):
    try:
        os.remove(_TEST_DB + _suffix)
    except FileNotFoundError:
        pass
os.environ["XOPS_DB_PATH"] = _TEST_DB

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture()
def auth_headers(client: TestClient) -> dict[str, str]:
    token = client.post("/api/v3/dataops/token/ds_01_resident_registry").json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
