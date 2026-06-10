"""
FastAPI 애플리케이션 진입점
메인 애플리케이션을 설정하고 실행합니다.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config.settings import settings
from api.v1.router import api_router
from core.logger import logger
from core.system_check import check_system_requirements

# FastAPI 앱 생성
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="DataOps를 위한 표준 명세 편집기/해석기 API"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 등록
app.include_router(api_router, prefix=settings.api_prefix)


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "DataOps Standard Editor API",
        "version": settings.api_version,
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    system_status = check_system_requirements()
    status = "healthy" if system_status["all_requirements_met"] else "degraded"
    
    return {
        "status": status,
        "system_requirements": system_status
    }


if __name__ == "__main__":
    import uvicorn
    import sys
    from pathlib import Path
    
    # 현재 파일의 디렉토리를 Python 경로에 추가
    backend_dir = Path(__file__).parent
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    
    logger.info("Starting FastAPI server...")
    logger.info(f"Working directory: {backend_dir}")
    
    # 시스템 요구사항 확인
    system_status = check_system_requirements()
    if not system_status["all_requirements_met"]:
        logger.warning("Some system requirements are not met:")
        if not system_status["git"]["installed"]:
            logger.warning("  - Git is not installed")
        if not system_status["dvc"]["installed"]:
            logger.warning("  - DVC is not installed")
    else:
        logger.info("All system requirements are met")
        if system_status["git"]["version"]:
            logger.info(f"  Git: {system_status['git']['version']}")
        if system_status["dvc"]["version"]:
            logger.info(f"  DVC: {system_status['dvc']['version']}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(backend_dir)]
    )
