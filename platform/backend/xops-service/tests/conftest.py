"""공용 pytest 픽스처."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture()
def auth_headers(client: TestClient) -> dict[str, str]:
    token = client.post("/api/v3/dataops/token/ds_01_resident_registry").json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
