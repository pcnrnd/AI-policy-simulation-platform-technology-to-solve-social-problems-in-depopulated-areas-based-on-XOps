"""
Repository API 엔드포인트
Repository 관련 API를 정의합니다.
"""
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict

from models.repository import (
    RepositoryInitRequest,
    CheckoutRequest,
    CommitRequest,
    AddRequest,
    UpdateRequest,
    MergeRequest,
    BranchRequest,
    StatusRequest,
    RepositoryResponse
)
from services.repository_service import RepositoryService
from api.dependencies import get_repository_service
from core.exceptions import RepositoryError, BranchError
from core.logger import logger

router = APIRouter(prefix="/repository", tags=["repository"])


@router.post("/init", response_model=RepositoryResponse)
async def init_repository(
    request: RepositoryInitRequest,
    service: RepositoryService = Depends(get_repository_service)
) -> RepositoryResponse:
    """
    Repository 초기화
    
    - **path**: 저장소 경로
    - **remote_url**: 원격 저장소 URL (선택)
    """
    try:
        result = service.init_repository(
            path=request.path,
            remote_url=request.remote_url
        )
        return RepositoryResponse(**result)
    except RepositoryError as e:
        logger.error(f"Repository init error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/checkout", response_model=RepositoryResponse)
async def checkout_repository(
    request: CheckoutRequest,
    service: RepositoryService = Depends(get_repository_service)
) -> RepositoryResponse:
    """
    브랜치 체크아웃
    
    - **branch**: 브랜치 이름
    - **path**: 저장소 경로
    - **create**: 새 브랜치 생성 여부
    """
    try:
        result = service.checkout(
            branch=request.branch,
            path=request.path,
            create=request.create
        )
        return RepositoryResponse(**result)
    except BranchError as e:
        logger.error(f"Checkout error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/commit", response_model=RepositoryResponse)
async def commit_changes(
    request: CommitRequest,
    service: RepositoryService = Depends(get_repository_service)
) -> RepositoryResponse:
    """
    변경사항 커밋
    
    - **message**: 커밋 메시지
    - **path**: 저장소 경로
    - **push**: 원격 저장소로 푸시 여부
    """
    try:
        result = service.commit(
            message=request.message,
            path=request.path,
            push=request.push
        )
        return RepositoryResponse(**result)
    except RepositoryError as e:
        logger.error(f"Commit error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/add", response_model=RepositoryResponse)
async def add_files(
    request: AddRequest,
    service: RepositoryService = Depends(get_repository_service)
) -> RepositoryResponse:
    """
    파일 추가
    
    - **files**: 추가할 파일 경로 리스트
    - **path**: 저장소 경로
    """
    try:
        result = service.add_files(
            files=request.files,
            path=request.path
        )
        return RepositoryResponse(**result)
    except RepositoryError as e:
        logger.error(f"Add files error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/update", response_model=RepositoryResponse)
async def update_repository(
    request: UpdateRequest,
    service: RepositoryService = Depends(get_repository_service)
) -> RepositoryResponse:
    """
    원격 저장소에서 업데이트 (Pull)
    
    - **path**: 저장소 경로
    """
    try:
        result = service.update(path=request.path)
        return RepositoryResponse(**result)
    except RepositoryError as e:
        logger.error(f"Update error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/merge", response_model=RepositoryResponse)
async def merge_branches(
    request: MergeRequest,
    service: RepositoryService = Depends(get_repository_service)
) -> RepositoryResponse:
    """
    브랜치 병합
    
    - **source_branch**: 병합할 소스 브랜치
    - **target_branch**: 타겟 브랜치
    - **path**: 저장소 경로
    """
    try:
        result = service.merge(
            source_branch=request.source_branch,
            target_branch=request.target_branch,
            path=request.path
        )
        return RepositoryResponse(**result)
    except BranchError as e:
        logger.error(f"Merge error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/branches", response_model=RepositoryResponse)
async def get_branches(
    path: str,
    service: RepositoryService = Depends(get_repository_service)
) -> RepositoryResponse:
    """
    브랜치 목록 조회
    
    - **path**: 저장소 경로
    """
    try:
        result = service.get_branches(path=path)
        return RepositoryResponse(**result)
    except BranchError as e:
        logger.error(f"Get branches error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/status", response_model=RepositoryResponse)
async def get_status(
    path: str,
    service: RepositoryService = Depends(get_repository_service)
) -> RepositoryResponse:
    """
    저장소 상태 조회
    
    - **path**: 저장소 경로
    """
    try:
        result = service.get_status(path=path)
        return RepositoryResponse(**result)
    except RepositoryError as e:
        logger.error(f"Get status error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

