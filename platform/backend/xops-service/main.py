"""xops-service 진입점 — DataOps + MLOps 운영 평면.

실행: cd platform/backend/xops-service && python -m uvicorn main:app --reload
(임포트가 `from src...` 형태이므로 cwd는 반드시 xops-service/ 여야 함)
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.v3.router import api_router
from src.core.exceptions import register_exception_handlers
from src.core.settings import get_settings


def create_app() -> FastAPI:
    """FastAPI 앱 구성 — CORS + /api/v3 라우터 + 예외 핸들러."""
    settings = get_settings()
    app = FastAPI(title="xops-service", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_exception_handlers(app)
    app.include_router(api_router, prefix=settings.api_prefix)

    @app.get("/")
    def read_root() -> dict[str, str]:
        """헬스체크 (스캐폴드 계약 유지)."""
        return {"xops": "connected"}

    return app


app = create_app()
