"""compose.xops.yaml 구조 검증 — DSN·마운트·헬스체크가 코드 계약과 어긋나지 않는지.

Docker 없이도 확인 가능한 정적 검증이다. 실 컨테이너 기동은 별도 스모크로 남긴다.
pyyaml 은 선언된 테스트 의존성이 아니므로 없으면 건너뛴다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.core.settings import Settings

# tests/unit → tests → xops-service → backend → platform
_PLATFORM = Path(__file__).resolve().parents[4]
_COMPOSE = _PLATFORM / "compose.xops.yaml"


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    yaml = pytest.importorskip("yaml", reason="pyyaml 미설치 — compose 정적 검증 건너뜀")
    loaded: dict[str, Any] = yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))
    return loaded


def test_three_storage_services_declared(compose: dict[str, Any]) -> None:
    services = compose["services"]
    assert {"xops-service", "frontend", "pg", "timescale", "mongo"} <= set(services)
    assert services["pg"]["image"].startswith("postgis/postgis:")  # PG+PostGIS 겸용
    assert services["timescale"]["image"].startswith("timescale/timescaledb:")
    assert services["mongo"]["image"].startswith("mongo:")


def test_ports_avoid_default_collisions(compose: dict[str, Any]) -> None:
    services = compose["services"]
    assert services["pg"]["ports"] == ["5433:5432"]  # 로컬 5432 회피
    assert services["timescale"]["ports"] == ["5434:5432"]
    assert services["mongo"]["ports"] == ["27017:27017"]


def test_named_volumes_are_declared_not_bind_mounted(compose: dict[str, Any]) -> None:
    declared = set(compose["volumes"])
    assert {"pg-data", "timescale-data", "mongo-data"} <= declared
    for name, mount in (("pg", "pg-data"), ("timescale", "timescale-data"), ("mongo", "mongo-data")):
        data_mounts = [v for v in compose["services"][name]["volumes"] if v.startswith(f"{mount}:")]
        assert data_mounts, f"{name} 의 데이터 볼륨이 named volume 이어야 한다"


def test_initdb_scripts_are_mounted_read_only_and_exist(compose: dict[str, Any]) -> None:
    for service, folder in (("pg", "postgres"), ("timescale", "timescale"), ("mongo", "mongo")):
        mounts = compose["services"][service]["volumes"]
        expected = f"./initdb/{folder}:/docker-entrypoint-initdb.d:ro"
        assert expected in mounts, f"{service} initdb 마운트 누락"
        scripts = list((_PLATFORM / "initdb" / folder).glob("*"))
        assert scripts, f"initdb/{folder} 에 스크립트가 있어야 한다"


def test_healthchecks_and_dependency_gating(compose: dict[str, Any]) -> None:
    for name in ("pg", "timescale", "mongo"):
        assert compose["services"][name]["healthcheck"]["test"], f"{name} 헬스체크 필요"
    depends = compose["services"]["xops-service"]["depends_on"]
    for name in ("pg", "timescale", "mongo"):
        assert depends[name]["condition"] == "service_healthy"


def test_dsn_env_names_match_settings_fields(compose: dict[str, Any]) -> None:
    """compose 의 XOPS_* 환경변수가 Settings 필드와 실제로 대응해야 한다."""
    env = dict(item.split("=", 1) for item in compose["services"]["xops-service"]["environment"])
    prefix = Settings.model_config["env_prefix"]
    fields = set(Settings.model_fields)
    for key in ("XOPS_PG_DSN", "XOPS_TIMESCALE_DSN", "XOPS_MONGO_URI"):
        assert key in env, f"{key} 누락"
        assert key.removeprefix(prefix).lower() in fields, f"{key} 에 대응하는 Settings 필드가 없다"
    # 컨테이너 내부 통신이므로 호스트 매핑 포트(5433/5434)가 아니라 서비스명:내부포트를 써야 한다.
    assert "@pg:5432/" in env["XOPS_PG_DSN"]
    assert "@timescale:5432/" in env["XOPS_TIMESCALE_DSN"]
    assert "@mongo:27017/" in env["XOPS_MONGO_URI"]


def test_driver_requirements_are_installed_in_image() -> None:
    """DSN 이 있어도 이미지에 드라이버가 없으면 degrade 된다 — Dockerfile 이 설치해야 한다."""
    service_root = Path(__file__).resolve().parents[2]
    dockerfile = (service_root / "Dockerfile").read_text(encoding="utf-8")
    assert "requirements-db.txt" in dockerfile
    drivers = (service_root / "requirements-db.txt").read_text(encoding="utf-8")
    assert "psycopg" in drivers and "pymongo" in drivers
