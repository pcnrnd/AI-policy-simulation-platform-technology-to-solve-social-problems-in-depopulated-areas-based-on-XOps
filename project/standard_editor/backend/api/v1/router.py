"""
API 라우터 통합
모든 API 엔드포인트를 통합합니다.
"""
from fastapi import APIRouter
from api.v1 import repository, github, minio

api_router = APIRouter()

# Repository 라우터 등록
api_router.include_router(repository.router)

# GitHub 라우터 등록
api_router.include_router(github.router)

# MinIO 라우터 등록
api_router.include_router(minio.router)

