"""도메인 예외 계층 + FastAPI 핸들러 등록."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class XopsError(Exception):
    """서비스 기본 예외."""

    status_code = 400

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class SourceNotFoundError(XopsError):
    """요청한 데이터 소스가 카탈로그에 없음."""

    status_code = 404


class UnsafeQueryError(XopsError):
    """SQL 인젝션 가드가 위험 요소를 감지."""

    status_code = 400


class AuthError(XopsError):
    """인증/인가 실패."""

    status_code = 401


def register_exception_handlers(app: FastAPI) -> None:
    """도메인 예외를 일관된 JSON 응답으로 변환."""

    @app.exception_handler(XopsError)
    async def _handle(_: Request, exc: XopsError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"status": exc.status_code, "error": type(exc).__name__, "message": exc.message},
        )
