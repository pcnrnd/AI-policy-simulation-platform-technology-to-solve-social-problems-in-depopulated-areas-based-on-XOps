"""토큰 발급 클라이언트 게이트 단위 테스트 (T4/①) — demo-open + prod 게이트."""

from __future__ import annotations

import pytest

from src.api import dependencies
from src.core.exceptions import AuthError
from src.core.settings import Settings


def _use_settings(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    monkeypatch.setattr(dependencies, "get_settings", lambda: settings)


def test_dev_open(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_settings(monkeypatch, Settings(environment="dev"))
    dependencies.require_client(None, None)  # 개방 — 예외 없음


def test_prod_without_configured_creds_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    # prod에서 자격증명 미설정은 "개방"이 아니라 거절이다. 정상 운영이면 기동 단계에서 이미 막힌다.
    _use_settings(monkeypatch, Settings(environment="prod"))
    with pytest.raises(AuthError):
        dependencies.require_client(None, None)


def test_prod_without_configured_creds_refuses_startup() -> None:
    with pytest.raises(RuntimeError):
        Settings(environment="prod", jwt_secret="a-real-secret").validate_runtime()
    # 둘 다 설정하면 기동 가능
    Settings(
        environment="prod", jwt_secret="a-real-secret", client_id="cid", client_secret="csec"
    ).validate_runtime()


def test_prod_requires_valid_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_settings(monkeypatch, Settings(environment="prod", client_id="cid", client_secret="csec"))
    with pytest.raises(AuthError):
        dependencies.require_client(None, None)
    with pytest.raises(AuthError):
        dependencies.require_client("cid", "wrong")
    dependencies.require_client("cid", "csec")  # 일치 — 통과


def test_issued_scope_by_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    # 주 자격증명은 읽기·쓰기, 조회 전용 자격증명은 data:read만 받는다(prod·dev 공통).
    for env in ("prod", "dev"):
        _use_settings(
            monkeypatch,
            Settings(
                environment=env,
                client_id="cid",
                client_secret="csec",
                readonly_client_id="rid",
                readonly_client_secret="rsec",
            ),
        )
        assert dependencies.require_client("cid", "csec") == "data:read data:write"
        assert dependencies.require_client("rid", "rsec") == "data:read"
        with pytest.raises(AuthError):
            dependencies.require_client("rid", "wrong")


def test_readonly_creds_must_be_set_together() -> None:
    base = {"environment": "prod", "jwt_secret": "a-real-secret", "client_id": "cid", "client_secret": "csec"}
    with pytest.raises(RuntimeError):
        Settings(**base, readonly_client_id="rid").validate_runtime()
    with pytest.raises(RuntimeError):
        Settings(**base, readonly_client_secret="rsec").validate_runtime()
    Settings(**base, readonly_client_id="rid", readonly_client_secret="rsec").validate_runtime()
