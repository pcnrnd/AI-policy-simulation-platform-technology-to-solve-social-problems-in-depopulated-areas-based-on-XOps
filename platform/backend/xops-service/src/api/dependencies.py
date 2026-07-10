"""공용 FastAPI 의존성 — JWT 인증/스코프 게이팅."""

from __future__ import annotations

from typing import Any, Callable

from fastapi import Depends, Header
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.auth.jwt import decode_jwt, require_scope
from src.core.exceptions import AuthError
from src.core.settings import get_settings

# auto_error=False: 미인증 시 403 대신 우리 AuthError(401, buildUnauthorized 계약)로 처리
_bearer = HTTPBearer(auto_error=False)


def require_auth(scope: str) -> Callable[..., dict[str, Any]]:
    """지정 scope를 요구하는 인증 의존성 팩토리."""

    def _dependency(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> dict[str, Any]:
        if creds is None:
            raise AuthError("JWT 토큰이 필요합니다. 토큰 발급 후 Authorization: Bearer <token>로 요청하세요.")
        payload = decode_jwt(creds.credentials)
        require_scope(payload, scope)
        return payload

    return _dependency


def require_client(
    x_client_id: str | None = Header(None),
    x_client_secret: str | None = Header(None),
) -> None:
    """토큰 발급 게이트 — prod에서 client 자격증명이 설정된 경우에만 검증(dev는 개방)."""
    settings = get_settings()
    if settings.environment == "prod" and settings.client_id:
        if x_client_id != settings.client_id or x_client_secret != settings.client_secret:
            raise AuthError("클라이언트 자격증명이 유효하지 않습니다 (X-Client-Id / X-Client-Secret).")
