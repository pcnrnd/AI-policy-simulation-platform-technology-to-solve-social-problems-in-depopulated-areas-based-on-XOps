"""SW 등록 단위 진입점(sw_main) — 라우터 경계와 import 독립성."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sw_main import create_sw01_app, create_sw02_app

_SERVICE_DIR = Path(__file__).resolve().parents[2]


def _api_prefixes(app) -> set[str]:
    return {r.path.split("/")[3] for r in app.routes if r.path.startswith("/api/v3/")}


def test_sw01_registers_only_dataops() -> None:
    app = create_sw01_app()
    assert _api_prefixes(app) == {"dataops"}
    client = TestClient(app)
    assert client.get("/").json()["program"] == "sw01-dataops"
    assert client.get("/api/v3/dataops/catalog").status_code == 200


def test_sw02_registers_only_monitoring_and_orchestration() -> None:
    app = create_sw02_app()
    assert _api_prefixes(app) == {"monitoring", "orchestration"}
    client = TestClient(app)
    assert client.get("/").json()["program"] == "sw02-mlops"
    assert client.get("/api/v3/orchestration/models").status_code == 200


# SW-02 → src.realdata.pg_reader → src.dataops.safety(식별자 검증 함수) 한 경로만 공유한다.
# 경계 문서(docs/sw-registration/README.md)와 이 허용 목록을 함께 고친다.
_SW02_SHARED = ("src.dataops", "src.dataops.safety")


@pytest.mark.parametrize(
    ("factory", "forbidden", "allowed"),
    [
        ("create_sw01_app", ("src.mlops", "src.realdata", "src.api.v3.monitoring"), ()),
        ("create_sw02_app", ("src.dataops", "src.api.v3.dataops"), _SW02_SHARED),
    ],
)
def test_entrypoint_does_not_import_other_sw(
    factory: str, forbidden: tuple[str, ...], allowed: tuple[str, ...]
) -> None:
    # 같은 프로세스에는 main.py가 이미 전체를 import해 두었으므로 새 인터프리터에서 확인한다.
    code = (
        "import sys, sw_main\n"
        f"sw_main.{factory}()\n"
        f"bad = sorted(m for m in sys.modules if m.startswith({forbidden!r})"
        f" and m not in {allowed!r})\n"
        "print(bad)\n"
        "sys.exit(1 if bad else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=_SERVICE_DIR,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
