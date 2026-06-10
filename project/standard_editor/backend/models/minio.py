"""
MinIO 관련 Pydantic 모델
MinIO API 요청/응답 스키마를 정의합니다.
"""
from pydantic import BaseModel, Field
from typing import Optional, List


class MinIOUploadRequest(BaseModel):
    """MinIO 파일 업로드 요청 모델"""
    bucket_name: str = Field(..., description="버킷 이름")
    object_name: str = Field(..., description="객체 이름 (파일 경로)")
    file_path: str = Field(..., description="로컬 파일 경로")
    content_type: Optional[str] = Field(None, description="콘텐츠 타입")


class MinIODownloadRequest(BaseModel):
    """MinIO 파일 다운로드 요청 모델"""
    bucket_name: str = Field(..., description="버킷 이름")
    object_name: str = Field(..., description="객체 이름")


class MinIODeleteRequest(BaseModel):
    """MinIO 파일 삭제 요청 모델"""
    bucket_name: str = Field(..., description="버킷 이름")
    object_name: str = Field(..., description="객체 이름")


class MinIOListRequest(BaseModel):
    """MinIO 객체 목록 조회 요청 모델"""
    bucket_name: str = Field(..., description="버킷 이름")
    prefix: Optional[str] = Field(None, description="객체 이름 접두사")


class MinIOResponse(BaseModel):
    """MinIO 응답 모델"""
    success: bool
    message: Optional[str] = None
    data: Optional[dict] = None
    buckets: Optional[List[dict]] = None
    objects: Optional[List[dict]] = None
    count: Optional[int] = None

