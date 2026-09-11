"""/api/v3 라우터 집약 — dataops · monitoring · orchestration · overview · realdata."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.v3 import dataops, monitoring, orchestration, overview, realdata

api_router = APIRouter()
api_router.include_router(dataops.router)
api_router.include_router(monitoring.router)
api_router.include_router(orchestration.router)
api_router.include_router(overview.router)
api_router.include_router(realdata.router)
