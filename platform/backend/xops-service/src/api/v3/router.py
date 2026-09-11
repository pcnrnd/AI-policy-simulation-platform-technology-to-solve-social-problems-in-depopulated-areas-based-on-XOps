"""/api/v3 라우터 집약 — dataops · monitoring · orchestration · overview · realdata · realdata_datasets · realdata_runs · realdata_analysis."""
from fastapi import APIRouter

from src.api.v3 import (
    dataops,
    monitoring,
    orchestration,
    overview,
    realdata,
    realdata_analysis,
    realdata_datasets,
    realdata_runs,
)

api_router = APIRouter()
api_router.include_router(dataops.router)
api_router.include_router(monitoring.router)
api_router.include_router(orchestration.router)
api_router.include_router(overview.router)
api_router.include_router(realdata.router)
api_router.include_router(realdata_datasets.router)
api_router.include_router(realdata_runs.router)
api_router.include_router(realdata_analysis.router)
