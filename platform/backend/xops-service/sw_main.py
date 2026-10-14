"""SW 등록 단위별 단독 실행 진입점 — SW-01(DataOps)·SW-02(MLOps).

`main.py`는 전체 라우터를 한 프로세스로 띄운다. 여기서는 SW별 라우터만 등록해
각 프로그램이 서로를 import하지 않고 기동됨을 보인다. 팩토리라서 uvicorn `--factory`로 띄운다.

    python -m uvicorn sw_main:create_sw01_app --factory --port 8001   # SW-01
    python -m uvicorn sw_main:create_sw02_app --factory --port 8002   # SW-02

라우터 import는 각 팩토리 안에서만 한다 — 모듈 최상단에서 둘 다 import하면 독립성이 깨진다.
"""

from __future__ import annotations

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.db import init_db
from src.core.exceptions import register_exception_handlers
from src.core.logger import get_logger
from src.core.settings import get_settings


# ponytail: main.create_app과 같은 구성을 복제한다. main.py를 import하면 전체 라우터가 딸려 오기 때문.
def _build(title: str, *routers: APIRouter) -> FastAPI:
    settings = get_settings()
    settings.validate_runtime()
    init_db()
    get_logger().info(f"{title} 기동 env={settings.environment} prefix={settings.api_prefix}")
    app = FastAPI(title=title, version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_exception_handlers(app)
    for router in routers:
        app.include_router(router, prefix=settings.api_prefix)

    @app.get("/")
    def read_root() -> dict[str, str]:
        """헬스체크."""
        return {"xops": "connected", "program": title}

    return app


def create_sw01_app() -> FastAPI:
    """SW-01 메타데이터 기반 다기종 데이터 아카이빙 관리 — /api/v3/dataops만 등록."""
    from src.api.v3 import dataops

    return _build("sw01-dataops", dataops.router)


def create_sw02_app() -> FastAPI:
    """SW-02 AI 모델 성능 모니터링·재학습 관리 — /api/v3/monitoring·/orchestration만 등록."""
    from src.api.v3 import monitoring, orchestration

    return _build("sw02-mlops", monitoring.router, orchestration.router)
