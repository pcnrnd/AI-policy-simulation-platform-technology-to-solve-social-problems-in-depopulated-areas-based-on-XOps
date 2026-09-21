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


def optional_auth(scope: str) -> Callable[..., dict[str, Any] | None]:
    """토큰이 있을 때만 검증하는 의존성 팩토리 — 없으면 None(401을 내지 않는다).

    공개 GET(`/monitoring/*`·`/orchestration/*`)이 기존 계약을 그대로 유지하면서
    실데이터(rd_*)는 인증된 호출에만 내려주기 위한 게이트다. 토큰이 **틀린** 경우는
    require_auth와 똑같이 AuthError로 거절한다 — "없음"과 "틀림"을 섞지 않는다.
    """

    def _dependency(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> dict[str, Any] | None:
        if creds is None:
            return None
        payload = decode_jwt(creds.credentials)
        require_scope(payload, scope)
        return payload

    return _dependency


def require_client(
    x_client_id: str | None = Header(None),
    x_client_secret: str | None = Header(None),
) -> None:
    """토큰 발급 게이트 — prod는 항상 검증(dev는 개방).

    자격증명 미설정은 게이트 해제가 아니다. prod에서 미설정이면 `Settings.validate_runtime`이
    기동을 거부하므로, 여기까지 온 prod 요청은 반드시 설정된 값과 대조된다.
    """
    settings = get_settings()
    if settings.environment == "prod":
        if x_client_id != settings.client_id or x_client_secret != settings.client_secret:
            raise AuthError("클라이언트 자격증명이 유효하지 않습니다 (X-Client-Id / X-Client-Secret).")
