"""jwt 단위 테스트 — 발급/검증/스코프."""

from __future__ import annotations

import time

import pytest

from src.auth.jwt import decode_jwt, issue_jwt, issue_oauth2, require_scope
from src.core.exceptions import AuthError


def test_issue_and_decode_roundtrip() -> None:
    token = issue_jwt("ds_01")
    payload = decode_jwt(token)
    assert payload["source"] == "ds_01"
    assert payload["scope"] == "data:read data:write"


def test_tampered_signature_rejected() -> None:
    token = issue_jwt("ds_01")
    header, body, _sig = token.split(".")
    with pytest.raises(AuthError):
        decode_jwt(f"{header}.{body}.deadbeef")


def test_malformed_token_rejected() -> None:
    with pytest.raises(AuthError):
        decode_jwt("not-a-jwt")


def test_expired_token_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    token = issue_jwt("ds_01")
    now = time.time()  # 실제 시각 캡처 후 미래로 점프 (self-recursion 방지)
    monkeypatch.setattr(time, "time", lambda: now + 7200)
    with pytest.raises(AuthError):
        decode_jwt(token)


def test_require_scope() -> None:
    payload = decode_jwt(issue_jwt("ds_01"))
    require_scope(payload, "data:read")
    with pytest.raises(AuthError):
        require_scope(payload, "data:admin")


def test_oauth2_grant_shape() -> None:
    grant = issue_oauth2("ds_02")
    assert grant["grant_type"] == "authorization_code"
    assert grant["token_type"] == "Bearer"
    assert decode_jwt(grant["access_token"])["source"] == "ds_02"
