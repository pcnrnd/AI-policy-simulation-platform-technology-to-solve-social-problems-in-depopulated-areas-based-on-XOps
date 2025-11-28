"""
Repository 관련 Pydantic 모델
API 요청/응답 스키마를 정의합니다.
"""
from pydantic import BaseModel, Field
from typing import Optional, List


class RepositoryInitRequest(BaseModel):
    """Repository 초기화 요청 모델"""
    path: str = Field(..., description="저장소 경로")
    remote_url: Optional[str] = Field(None, description="원격 저장소 URL")


class CheckoutRequest(BaseModel):
    """Checkout 요청 모델"""
    branch: str = Field(..., description="브랜치 이름")
    path: str = Field(..., description="저장소 경로")
    create: bool = Field(False, description="새 브랜치 생성 여부")


class CommitRequest(BaseModel):
    """Commit 요청 모델"""
    message: str = Field(..., description="커밋 메시지")
    path: str = Field(..., description="저장소 경로")
    push: bool = Field(False, description="원격 저장소로 푸시 여부")


class AddRequest(BaseModel):
    """Add 요청 모델"""
    files: List[str] = Field(..., description="추가할 파일 경로 리스트")
    path: str = Field(..., description="저장소 경로")


class UpdateRequest(BaseModel):
    """Update 요청 모델"""
    path: str = Field(..., description="저장소 경로")


class MergeRequest(BaseModel):
    """Merge 요청 모델"""
    source_branch: str = Field(..., description="병합할 소스 브랜치")
    target_branch: str = Field(..., description="타겟 브랜치")
    path: str = Field(..., description="저장소 경로")


class BranchRequest(BaseModel):
    """Branch 생성 요청 모델"""
    branch_name: str = Field(..., description="브랜치 이름")
    path: str = Field(..., description="저장소 경로")


class StatusRequest(BaseModel):
    """Status 요청 모델"""
    path: str = Field(..., description="저장소 경로")


class RepositoryResponse(BaseModel):
    """Repository 응답 모델"""
    success: bool
    message: Optional[str] = None
    data: Optional[dict] = None
    results: Optional[dict] = None

