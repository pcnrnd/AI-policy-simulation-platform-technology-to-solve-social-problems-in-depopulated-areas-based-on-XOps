"""HS256 JWT 발급/검증 — stdlib만 사용(외부 pyjwt 불필요).

프론트 dataopsApi.js의 issueMockJwt/issueMockOAuth2/decodeJwtPayload 계약과 호환:
payload = {sub, scope:"data:read data:write", source, iat, exp}. 단 서명은 실제 HS256.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any

from src.core.exceptions import AuthError
from src.core.settings import get_settings

_CLIENT_SUB = "rnd-dataops-client"


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(seg: str) -> bytes:
    pad = "=" * (-len(seg) % 4)
    return base64.urlsafe_b64decode(seg + pad)


def _sign(signing_input: bytes) -> str:
    settings = get_settings()
    sig = hmac.new(settings.jwt_secret.encode(), signing_input, hashlib.sha256).digest()
    return _b64url_encode(sig)


def issue_jwt(source_id: str) -> str:
    """소스 접근용 HS256 토큰 발급."""
    settings = get_settings()
    header = {"alg": settings.jwt_algorithm, "typ": "JWT"}
    now = int(time.time())
    payload = {
        "sub": _CLIENT_SUB,
        "scope": settings.jwt_scope,
        "source": source_id,
        "iat": now,
        "exp": now + settings.jwt_expiry_seconds,
    }
    segments = [
        _b64url_encode(json.dumps(header, separators=(",", ":")).encode()),
        _b64url_encode(json.dumps(payload, separators=(",", ":")).encode()),
    ]
    signing_input = ".".join(segments).encode()
    segments.append(_sign(signing_input))
    return ".".join(segments)


def issue_oauth2(source_id: str) -> dict[str, Any]:
    """OAuth2 Authorization Code Grant 흐름(access_token은 JWT 형식)."""
    settings = get_settings()
    now = int(time.time())
    code = _b64url_encode(json.dumps({"c": source_id, "t": now}).encode())[:16]
    return {
        "grant_type": "authorization_code",
        "authorization_code": code,
        "token_type": "Bearer",
        "expires_in": settings.jwt_expiry_seconds,
        "scope": settings.jwt_scope,
        "access_token": issue_jwt(source_id),
    }


def decode_jwt(token: str) -> dict[str, Any]:
    """서명·만료를 검증하고 payload 반환. 실패 시 AuthError."""
    try:
        header_seg, payload_seg, sig_seg = token.split(".")
    except ValueError as exc:
        raise AuthError("잘못된 토큰 형식입니다.") from exc

    signing_input = f"{header_seg}.{payload_seg}".encode()
    if not hmac.compare_digest(sig_seg, _sign(signing_input)):
        raise AuthError("토큰 서명이 유효하지 않습니다.")

    try:
        payload: dict[str, Any] = json.loads(_b64url_decode(payload_seg))
    except (ValueError, json.JSONDecodeError) as exc:
        raise AuthError("토큰 payload를 해석할 수 없습니다.") from exc

    if int(payload.get("exp", 0)) < int(time.time()):
        raise AuthError("토큰이 만료되었습니다.")
    return payload


def require_scope(payload: dict[str, Any], scope: str) -> None:
    """payload에 지정 scope가 포함됐는지 확인."""
    granted = str(payload.get("scope", "")).split()
    if scope not in granted:
        raise AuthError(f"scope '{scope}' 권한이 없습니다.")
