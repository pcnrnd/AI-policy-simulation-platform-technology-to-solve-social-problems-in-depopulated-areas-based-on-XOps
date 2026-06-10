"""
MinIO API 엔드포인트
MinIO 직접 업로드 관련 API를 정의합니다.
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from typing import Optional

from models.minio import (
    MinIOUploadRequest,
    MinIODownloadRequest,
    MinIODeleteRequest,
    MinIOListRequest,
    MinIOResponse
)
from services.minio_service import MinIOService
from core.exceptions import RepositoryError
from core.logger import logger
from core.error_messages import get_user_friendly_message

router = APIRouter(prefix="/minio", tags=["minio"])


def get_minio_service() -> MinIOService:
    """MinIO 서비스 의존성 주입"""
    try:
        return MinIOService()
    except ImportError as e:
        logger.error(f"MinIO service initialization failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail="MinIO 서비스를 사용할 수 없습니다. minio 패키지가 설치되어 있는지 확인하세요."
        )


@router.get("/buckets", response_model=MinIOResponse)
async def list_buckets(
    service: MinIOService = Depends(get_minio_service)
) -> MinIOResponse:
    """
    버킷 목록 조회
    """
    try:
        result = service.list_buckets()
        return MinIOResponse(**result)
    except RepositoryError as e:
        error_msg = get_user_friendly_message(str(e))
        logger.error(f"List buckets error: {str(e)}")
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        error_msg = get_user_friendly_message(str(e))
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=error_msg)


@router.post("/buckets", response_model=MinIOResponse)
async def create_bucket(
    bucket_name: str,
    service: MinIOService = Depends(get_minio_service)
) -> MinIOResponse:
    """
    버킷 생성
    
    - **bucket_name**: 버킷 이름
    """
    try:
        result = service.create_bucket(bucket_name)
        return MinIOResponse(**result)
    except RepositoryError as e:
        error_msg = get_user_friendly_message(str(e))
        logger.error(f"Create bucket error: {str(e)}")
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        error_msg = get_user_friendly_message(str(e))
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=error_msg)


@router.post("/upload", response_model=MinIOResponse)
async def upload_file(
    bucket_name: str,
    object_name: str,
    file: UploadFile = File(...),
    service: MinIOService = Depends(get_minio_service)
) -> MinIOResponse:
    """
    파일 업로드 (Multipart Form)
    
    - **bucket_name**: 버킷 이름
    - **object_name**: 객체 이름 (파일 경로)
    - **file**: 업로드할 파일
    """
    try:
        file_data = await file.read()
        result = service.upload_file(
            bucket_name=bucket_name,
            object_name=object_name,
            file_data=file_data,
            content_type=file.content_type
        )
        return MinIOResponse(**result)
    except RepositoryError as e:
        error_msg = get_user_friendly_message(str(e))
        logger.error(f"Upload file error: {str(e)}")
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        error_msg = get_user_friendly_message(str(e))
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=error_msg)


@router.post("/upload-path", response_model=MinIOResponse)
async def upload_file_from_path(
    request: MinIOUploadRequest,
    service: MinIOService = Depends(get_minio_service)
) -> MinIOResponse:
    """
    로컬 파일 경로로부터 업로드
    
    - **bucket_name**: 버킷 이름
    - **object_name**: 객체 이름 (파일 경로)
    - **file_path**: 로컬 파일 경로
    - **content_type**: 콘텐츠 타입 (선택)
    """
    try:
        result = service.upload_file(
            bucket_name=request.bucket_name,
            object_name=request.object_name,
            file_path=request.file_path,
            content_type=request.content_type
        )
        return MinIOResponse(**result)
    except RepositoryError as e:
        error_msg = get_user_friendly_message(str(e))
        logger.error(f"Upload file from path error: {str(e)}")
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        error_msg = get_user_friendly_message(str(e))
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/objects", response_model=MinIOResponse)
async def list_objects(
    bucket_name: str,
    prefix: Optional[str] = None,
    service: MinIOService = Depends(get_minio_service)
) -> MinIOResponse:
    """
    버킷 내 객체 목록 조회
    
    - **bucket_name**: 버킷 이름
    - **prefix**: 객체 이름 접두사 (선택)
    """
    try:
        result = service.list_objects(bucket_name, prefix=prefix)
        return MinIOResponse(**result)
    except RepositoryError as e:
        error_msg = get_user_friendly_message(str(e))
        logger.error(f"List objects error: {str(e)}")
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        error_msg = get_user_friendly_message(str(e))
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=error_msg)


@router.post("/download", response_model=MinIOResponse)
async def download_file(
    request: MinIODownloadRequest,
    service: MinIOService = Depends(get_minio_service)
) -> MinIOResponse:
    """
    파일 다운로드
    
    - **bucket_name**: 버킷 이름
    - **object_name**: 객체 이름
    """
    try:
        result = service.download_file(
            bucket_name=request.bucket_name,
            object_name=request.object_name
        )
        # 파일 데이터는 base64로 인코딩하여 반환
        import base64
        result["data"] = {
            "content": base64.b64encode(result["data"]).decode("utf-8"),
            "size": result["size"],
            "object_name": result["object_name"]
        }
        return MinIOResponse(**result)
    except RepositoryError as e:
        error_msg = get_user_friendly_message(str(e))
        logger.error(f"Download file error: {str(e)}")
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        error_msg = get_user_friendly_message(str(e))
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=error_msg)


@router.delete("/delete", response_model=MinIOResponse)
async def delete_file(
    request: MinIODeleteRequest,
    service: MinIOService = Depends(get_minio_service)
) -> MinIOResponse:
    """
    파일 삭제
    
    - **bucket_name**: 버킷 이름
    - **object_name**: 객체 이름
    """
    try:
        result = service.delete_file(
            bucket_name=request.bucket_name,
            object_name=request.object_name
        )
        return MinIOResponse(**result)
    except RepositoryError as e:
        error_msg = get_user_friendly_message(str(e))
        logger.error(f"Delete file error: {str(e)}")
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        error_msg = get_user_friendly_message(str(e))
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=error_msg)

