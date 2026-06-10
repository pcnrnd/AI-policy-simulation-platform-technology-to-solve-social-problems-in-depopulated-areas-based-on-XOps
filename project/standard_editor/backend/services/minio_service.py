"""
MinIO 서비스
MinIO/S3 호환 저장소와의 직접 통신을 담당하는 서비스 클래스입니다.
"""
from typing import Dict, List, Optional, BinaryIO
from pathlib import Path
from io import BytesIO

from core.exceptions import RepositoryError
from core.logger import logger
from config.settings import settings

try:
    from minio import Minio
    from minio.error import S3Error
except ImportError:
    Minio = None
    S3Error = None
    logger.warning("minio 패키지가 설치되지 않았습니다. pip install minio를 실행하세요.")


class MinIOService:
    """MinIO/S3 직접 업로드 서비스"""
    
    def __init__(self):
        """
        초기화
        """
        if Minio is None:
            raise ImportError("minio 패키지가 설치되지 않았습니다. pip install minio를 실행하세요.")
        
        self.client = Minio(
            settings.minio_endpoint.replace("http://", "").replace("https://", ""),
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_endpoint.startswith("https://")
        )
    
    def list_buckets(self) -> Dict:
        """
        버킷 목록 조회
        
        Returns:
            버킷 목록
        """
        try:
            buckets = self.client.list_buckets()
            bucket_list = [{"name": bucket.name, "created": bucket.creation_date.isoformat()} for bucket in buckets]
            
            return {
                "success": True,
                "buckets": bucket_list
            }
        except Exception as e:
            logger.error(f"MinIO list buckets failed: {str(e)}")
            raise RepositoryError(f"버킷 목록 조회 실패: {str(e)}")
    
    def create_bucket(self, bucket_name: str) -> Dict:
        """
        버킷 생성
        
        Args:
            bucket_name: 버킷 이름
            
        Returns:
            생성 결과
        """
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                logger.info(f"Bucket created: {bucket_name}")
                return {
                    "success": True,
                    "message": f"버킷 '{bucket_name}'이 생성되었습니다."
                }
            else:
                return {
                    "success": True,
                    "message": f"버킷 '{bucket_name}'이 이미 존재합니다.",
                    "skipped": True
                }
        except Exception as e:
            logger.error(f"MinIO create bucket failed: {str(e)}")
            raise RepositoryError(f"버킷 생성 실패: {str(e)}")
    
    def upload_file(
        self,
        bucket_name: str,
        object_name: str,
        file_path: Optional[str] = None,
        file_data: Optional[bytes] = None,
        content_type: Optional[str] = None
    ) -> Dict:
        """
        파일 업로드
        
        Args:
            bucket_name: 버킷 이름
            object_name: 객체 이름 (파일 경로)
            file_path: 로컬 파일 경로
            file_data: 파일 데이터 (bytes)
            content_type: 콘텐츠 타입
            
        Returns:
            업로드 결과
        """
        try:
            # 버킷 존재 확인 및 생성
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                logger.info(f"Bucket created: {bucket_name}")
            
            # 파일 데이터 준비
            if file_path:
                file_path_obj = Path(file_path)
                if not file_path_obj.exists():
                    raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
                
                file_size = file_path_obj.stat().st_size
                with open(file_path_obj, 'rb') as file_data_stream:
                    self.client.put_object(
                        bucket_name,
                        object_name,
                        file_data_stream,
                        length=file_size,
                        content_type=content_type or self._guess_content_type(file_path)
                    )
            elif file_data:
                file_data_stream = BytesIO(file_data)
                self.client.put_object(
                    bucket_name,
                    object_name,
                    file_data_stream,
                    length=len(file_data),
                    content_type=content_type or "application/octet-stream"
                )
            else:
                raise ValueError("file_path 또는 file_data 중 하나는 필수입니다.")
            
            logger.info(f"File uploaded: {bucket_name}/{object_name}")
            return {
                "success": True,
                "message": f"파일이 업로드되었습니다: {bucket_name}/{object_name}",
                "bucket": bucket_name,
                "object": object_name
            }
        except Exception as e:
            logger.error(f"MinIO upload failed: {str(e)}")
            raise RepositoryError(f"파일 업로드 실패: {str(e)}")
    
    def list_objects(self, bucket_name: str, prefix: Optional[str] = None) -> Dict:
        """
        버킷 내 객체 목록 조회
        
        Args:
            bucket_name: 버킷 이름
            prefix: 객체 이름 접두사 (선택)
            
        Returns:
            객체 목록
        """
        try:
            objects = self.client.list_objects(bucket_name, prefix=prefix, recursive=True)
            object_list = []
            
            for obj in objects:
                object_list.append({
                    "name": obj.object_name,
                    "size": obj.size,
                    "last_modified": obj.last_modified.isoformat() if obj.last_modified else None,
                    "etag": obj.etag
                })
            
            return {
                "success": True,
                "objects": object_list,
                "count": len(object_list)
            }
        except Exception as e:
            logger.error(f"MinIO list objects failed: {str(e)}")
            raise RepositoryError(f"객체 목록 조회 실패: {str(e)}")
    
    def download_file(self, bucket_name: str, object_name: str) -> Dict:
        """
        파일 다운로드
        
        Args:
            bucket_name: 버킷 이름
            object_name: 객체 이름
            
        Returns:
            파일 데이터 및 메타데이터
        """
        try:
            response = self.client.get_object(bucket_name, object_name)
            file_data = response.read()
            response.close()
            response.release_conn()
            
            return {
                "success": True,
                "data": file_data,
                "size": len(file_data),
                "object_name": object_name
            }
        except Exception as e:
            logger.error(f"MinIO download failed: {str(e)}")
            raise RepositoryError(f"파일 다운로드 실패: {str(e)}")
    
    def delete_file(self, bucket_name: str, object_name: str) -> Dict:
        """
        파일 삭제
        
        Args:
            bucket_name: 버킷 이름
            object_name: 객체 이름
            
        Returns:
            삭제 결과
        """
        try:
            self.client.remove_object(bucket_name, object_name)
            logger.info(f"File deleted: {bucket_name}/{object_name}")
            return {
                "success": True,
                "message": f"파일이 삭제되었습니다: {bucket_name}/{object_name}"
            }
        except Exception as e:
            logger.error(f"MinIO delete failed: {str(e)}")
            raise RepositoryError(f"파일 삭제 실패: {str(e)}")
    
    def _guess_content_type(self, file_path: str) -> str:
        """
        파일 확장자로부터 콘텐츠 타입 추측
        
        Args:
            file_path: 파일 경로
            
        Returns:
            콘텐츠 타입
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        
        content_types = {
            '.csv': 'text/csv',
            '.json': 'application/json',
            '.parquet': 'application/octet-stream',
            '.pkl': 'application/octet-stream',
            '.py': 'text/x-python',
            '.txt': 'text/plain',
            '.md': 'text/markdown',
            '.yaml': 'text/yaml',
            '.yml': 'text/yaml',
        }
        
        return content_types.get(extension, 'application/octet-stream')

