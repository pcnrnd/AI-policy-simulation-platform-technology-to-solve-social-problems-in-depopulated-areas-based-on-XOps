"""Compose 구조 검증 — DSN·마운트·헬스체크가 코드 계약과 어긋나지 않는지.

앱(compose.xops.yaml, project=platform)과 저장소 3종(compose.db.yaml, project=population-db)은
분리되어 있다. 두 파일이 같은 외부망으로만 이어지는지까지 Docker 없이 정적으로 확인한다.
실 컨테이너 기동은 별도 스모크로 남긴다. pyyaml 은 선언된 테스트 의존성이 아니므로 없으면 건너뛴다.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from src.core.settings import Settings

# tests/unit → tests → xops-service → backend → platform
_PLATFORM = Path(__file__).resolve().parents[4]
_COMPOSE = _PLATFORM / "compose.xops.yaml"
_DB_COMPOSE = _PLATFORM / "compose.db.yaml"
_DB_EC2_COMPOSE = _PLATFORM / "compose.db.ec2.yaml"
_STORAGE = ("pg", "timescale", "mongo")
_SHARED_NETWORK = "population-db-data"


def _load(path: Path) -> dict[str, Any]:
    yaml = pytest.importorskip("yaml", reason="pyyaml 미설치 — compose 정적 검증 건너뜀")

    class _ComposeLoader(yaml.SafeLoader):
        """Compose 전용 태그(!override·!reset)는 값만 읽는다 — 병합 의미는 Compose가 처리한다."""

    for tag in ("!override", "!reset"):
        _ComposeLoader.add_constructor(
            tag, lambda loader, node: loader.construct_sequence(node) if isinstance(node, yaml.SequenceNode)
            else loader.construct_mapping(node) if isinstance(node, yaml.MappingNode) else loader.construct_scalar(node)
        )
    loaded: dict[str, Any] = yaml.load(path.read_text(encoding="utf-8"), Loader=_ComposeLoader)  # noqa: S506
    return loaded


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    return _load(_COMPOSE)


@pytest.fixture(scope="module")
def db_compose() -> dict[str, Any]:
    return _load(_DB_COMPOSE)


def test_app_compose_has_no_storage_services(compose: dict[str, Any]) -> None:
    """앱 project 에는 DB가 없다 — 앱 재생성이 DB를 다시 만들거나 새 빈 볼륨을 붙이지 않게 한다."""
    services = compose["services"]
    assert {"xops-service", "frontend"} <= set(services)
    assert not set(_STORAGE) & set(services)
    assert "depends_on" not in services["xops-service"]


def test_three_storage_services_declared(db_compose: dict[str, Any]) -> None:
    services = db_compose["services"]
    assert db_compose["name"] == "population-db"
    assert set(_STORAGE) <= set(services)
    assert services["pg"]["image"].startswith("postgis/postgis:")  # PG+PostGIS 겸용
    assert services["timescale"]["image"].startswith("timescale/timescaledb:")
    assert services["mongo"]["image"].startswith("mongo:")


def test_app_and_db_share_only_external_network(compose: dict[str, Any], db_compose: dict[str, Any]) -> None:
    for spec in (compose, db_compose):
        network = spec["networks"]["db-data"]
        assert network == {"external": True, "name": _SHARED_NETWORK}
    assert "db-data" in compose["services"]["xops-service"]["networks"]
    assert "db-data" not in compose["services"]["frontend"].get("networks", [])
    for name in _STORAGE:
        assert db_compose["services"][name]["networks"] == ["db-data"]


def test_ports_avoid_default_collisions() -> None:
    """운영 덮어쓰기는 DB 포트를 서버 안(127.0.0.1)에만 열고 기본 포트 5432를 피한다."""
    services = _load(_DB_EC2_COMPOSE)["services"]
    assert services["pg"]["ports"] == ["127.0.0.1:5433:5432"]
    assert services["timescale"]["ports"] == ["127.0.0.1:5434:5432"]
    assert services["mongo"]["ports"] == ["127.0.0.1:27017:27017"]


def test_named_volumes_are_external_not_bind_mounted(db_compose: dict[str, Any]) -> None:
    """데이터 볼륨은 외부 소유 이름 볼륨이다 — 이름 누락 시 새 빈 볼륨으로 '성공'하지 않게 한다."""
    declared = db_compose["volumes"]
    for mount in ("pg-data", "timescale-data", "mongo-data", "mongo-config"):
        assert declared[mount]["external"] is True
        assert re.fullmatch(r"\$\{POP_DB_[A-Z_]+_VOLUME:\?.+\}", declared[mount]["name"])
    for name, mount in (("pg", "pg-data"), ("timescale", "timescale-data"), ("mongo", "mongo-data")):
        data_mounts = [v for v in db_compose["services"][name]["volumes"] if v.startswith(f"{mount}:")]
        assert data_mounts, f"{name} 의 데이터 볼륨이 named volume 이어야 한다"


def test_initdb_scripts_are_mounted_read_only_and_exist(db_compose: dict[str, Any]) -> None:
    for service, folder in (("pg", "postgres"), ("timescale", "timescale"), ("mongo", "mongo")):
        mounts = db_compose["services"][service]["volumes"]
        expected = f"./initdb/{folder}:/docker-entrypoint-initdb.d:ro"
        assert expected in mounts, f"{service} initdb 마운트 누락"
        scripts = list((_PLATFORM / "initdb" / folder).glob("*"))
        assert scripts, f"initdb/{folder} 에 스크립트가 있어야 한다"


def test_storage_healthchecks_declared(db_compose: dict[str, Any]) -> None:
    """DB는 별도 project 라 앱 depends_on 으로 기다리지 않는다 — 각 DB 헬스체크와 runbook 준비 검사로 확인한다."""
    for name in _STORAGE:
        assert db_compose["services"][name]["healthcheck"]["test"], f"{name} 헬스체크 필요"
    assert (_PLATFORM / "scripts" / "check_db_readiness.py").is_file()


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


def test_xops_service_liveness_healthcheck(compose: dict[str, Any]) -> None:
    """생존 확인은 `GET /`를 python urllib으로 부른다(slim 이미지에 curl이 없다). DB 준비 판정이 아니다."""
    check = compose["services"]["xops-service"]["healthcheck"]
    command = " ".join(check["test"])
    assert check["test"][0] == "CMD"
    assert "urllib.request" in command and "http://127.0.0.1:8000/" in command


def test_readonly_client_env_is_passed_through(compose: dict[str, Any]) -> None:
    """조회 전용 자격증명은 저장소에 두지 않고 기동 환경(.env)에서 주입한다 — 미설정이면 빈 값."""
    env = dict(item.split("=", 1) for item in compose["services"]["xops-service"]["environment"])
    prefix = Settings.model_config["env_prefix"]
    for key in ("XOPS_READONLY_CLIENT_ID", "XOPS_READONLY_CLIENT_SECRET"):
        assert env.get(key) == "${%s:-}" % key, f"{key} 전달 누락"
        assert key.removeprefix(prefix).lower() in Settings.model_fields


def test_nginx_serves_openapi_and_keeps_api_block() -> None:
    """같은 도메인 연동 — `/api/v3/openapi.json` 정확 일치를 두고 기존 `/api/` 블록은 그대로 둔다."""
    conf = (_PLATFORM / "frontend" / "nginx.conf").read_text(encoding="utf-8")
    assert re.search(
        r"location = /api/v3/openapi\.json \{[^}]*proxy_pass http://xops-service:8000/openapi\.json;", conf
    )
    api_block = re.search(r"location /api/ \{([^}]*)\}", conf)
    assert api_block is not None
    # URI 없는 proxy_pass — `/api/v3/...` 경로가 그대로 전달된다(슬래시를 붙이면 경로가 잘린다).
    assert "proxy_pass http://xops-service:8000;" in api_block.group(1)
    # 타 부서 경로는 env 생성 설정을 include하고 하드코딩 활성 location을 두지 않는다.
    assert re.search(r"^include /etc/nginx/integration/maps\.conf;", conf, re.MULTILINE)
    assert re.search(r"^\s*include /etc/nginx/integration/routes\.conf;", conf, re.MULTILINE)
    assert not re.search(r"^\s*location /department/", conf, re.MULTILINE)


def test_driver_requirements_are_installed_in_image() -> None:
    """DSN 이 있어도 이미지에 드라이버가 없으면 degrade 된다 — Dockerfile 이 설치해야 한다."""
    service_root = Path(__file__).resolve().parents[2]
    dockerfile = (service_root / "Dockerfile").read_text(encoding="utf-8")
    assert "requirements-db.txt" in dockerfile
    drivers = (service_root / "requirements-db.txt").read_text(encoding="utf-8")
    assert "psycopg" in drivers and "pymongo" in drivers
