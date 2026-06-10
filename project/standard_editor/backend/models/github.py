"""
GitHub 관련 Pydantic 모델
GitHub API 요청/응답 스키마를 정의합니다.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict


class GitHubRemoteRequest(BaseModel):
    """GitHub Remote 설정 요청 모델"""
    path: str = Field(..., description="저장소 경로")
    remote_name: str = Field("origin", description="원격 저장소 이름")
    remote_url: str = Field(..., description="GitHub 저장소 URL (예: https://github.com/user/repo.git)")


class GitHubPushRequest(BaseModel):
    """GitHub Push 요청 모델"""
    path: str = Field(..., description="저장소 경로")
    remote_name: str = Field("origin", description="원격 저장소 이름")
    branch: Optional[str] = Field(None, description="푸시할 브랜치 (기본값: 현재 브랜치)")


class GitHubRemoteResponse(BaseModel):
    """GitHub Remote 응답 모델"""
    success: bool
    message: Optional[str] = None
    remotes: Optional[Dict[str, str]] = None
    data: Optional[dict] = None

