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


def test_prod_without_configured_creds_open(monkeypatch: pytest.MonkeyPatch) -> None:
    # prod이지만 client_id 미설정이면 게이트 비활성(개방)
    _use_settings(monkeypatch, Settings(environment="prod"))
    dependencies.require_client(None, None)


def test_prod_requires_valid_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_settings(monkeypatch, Settings(environment="prod", client_id="cid", client_secret="csec"))
    with pytest.raises(AuthError):
        dependencies.require_client(None, None)
    with pytest.raises(AuthError):
        dependencies.require_client("cid", "wrong")
    dependencies.require_client("cid", "csec")  # 일치 — 통과
