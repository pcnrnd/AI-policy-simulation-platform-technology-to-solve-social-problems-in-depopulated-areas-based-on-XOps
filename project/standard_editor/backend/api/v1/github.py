"""
GitHub API 엔드포인트
GitHub 원격 저장소 관련 API를 정의합니다.
"""
from fastapi import APIRouter, Depends, HTTPException

from models.github import (
    GitHubRemoteRequest,
    GitHubPushRequest,
    GitHubRemoteResponse
)
from services.repository_service import RepositoryService
from api.dependencies import get_repository_service
from core.exceptions import RepositoryError, ValidationError
from core.logger import logger
from core.error_messages import get_user_friendly_message

router = APIRouter(prefix="/github", tags=["github"])


@router.post("/remote", response_model=GitHubRemoteResponse)
async def set_github_remote(
    request: GitHubRemoteRequest,
    service: RepositoryService = Depends(get_repository_service)
) -> GitHubRemoteResponse:
    """
    GitHub 원격 저장소 설정
    
    - **path**: 저장소 경로
    - **remote_name**: 원격 저장소 이름 (기본값: origin)
    - **remote_url**: GitHub 저장소 URL
    """
    try:
        result = service.set_github_remote(
            path=request.path,
            remote_name=request.remote_name,
            remote_url=request.remote_url
        )
        return GitHubRemoteResponse(**result)
    except (RepositoryError, ValidationError) as e:
        error_msg = get_user_friendly_message(str(e))
        logger.error(f"Set GitHub remote error: {str(e)}")
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        error_msg = get_user_friendly_message(str(e))
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/remote", response_model=GitHubRemoteResponse)
async def get_github_remotes(
    path: str,
    service: RepositoryService = Depends(get_repository_service)
) -> GitHubRemoteResponse:
    """
    GitHub 원격 저장소 목록 조회
    
    - **path**: 저장소 경로
    """
    try:
        result = service.get_github_remotes(path=path)
        return GitHubRemoteResponse(**result)
    except (RepositoryError, ValidationError) as e:
        error_msg = get_user_friendly_message(str(e))
        logger.error(f"Get GitHub remotes error: {str(e)}")
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        error_msg = get_user_friendly_message(str(e))
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=error_msg)


@router.post("/push", response_model=GitHubRemoteResponse)
async def push_to_github(
    request: GitHubPushRequest,
    service: RepositoryService = Depends(get_repository_service)
) -> GitHubRemoteResponse:
    """
    GitHub에 푸시
    
    - **path**: 저장소 경로
    - **remote_name**: 원격 저장소 이름 (기본값: origin)
    - **branch**: 푸시할 브랜치 (선택)
    """
    try:
        result = service.push_to_github(
            path=request.path,
            remote_name=request.remote_name,
            branch=request.branch
        )
        return GitHubRemoteResponse(**result)
    except (RepositoryError, ValidationError) as e:
        error_msg = get_user_friendly_message(str(e))
        logger.error(f"Push to GitHub error: {str(e)}")
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        error_msg = get_user_friendly_message(str(e))
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=error_msg)

