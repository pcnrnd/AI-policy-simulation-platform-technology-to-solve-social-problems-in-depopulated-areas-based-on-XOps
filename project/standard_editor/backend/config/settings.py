"""
설정 관리 모듈
환경 변수 및 애플리케이션 설정을 관리합니다.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List


class Settings(BaseSettings):
    """애플리케이션 설정"""
    
    # Pydantic v2 설정
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )
    
    # API 설정
    api_title: str = "DataOps Standard Editor API"
    api_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    
    # CORS 설정
    cors_origins: List[str] = ["http://localhost:5173", "http://localhost:3000"]
    
    # 명령어 실행 설정
    command_timeout: int = 300  # 5분
    default_working_directory: Optional[str] = None
    
    # DVC 설정
    dvc_remote_name: str = "minio_dvc"
    dvc_remote_url: str = "s3://dvc"
    minio_endpoint: str = "http://minio:9000"
    minio_access_key: str = "minio"
    minio_secret_key: str = "minio123"


settings = Settings()

