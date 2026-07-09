"""서비스 설정 — 임계치·경로·인증을 전부 환경변수(.env)로 노출. 하드코딩 금지."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# xops-service/ 루트 (이 파일: src/core/settings.py → parents[2])
SERVICE_ROOT = Path(__file__).resolve().parents[2]
# 프론트 mock 데이터(카탈로그·시계열 시드)를 백엔드가 그대로 로드
_DEFAULT_MOCK = SERVICE_ROOT.parent.parent / "frontend" / "src" / "assets" / "mock_data.json"
_DEFAULT_JWT_SECRET = "dev-only-secret-change-me"


class Settings(BaseSettings):
    """모든 임계치/경로/CORS/토큰 설정의 단일 출처."""

    model_config = SettingsConfigDict(env_prefix="XOPS_", env_file=".env", extra="ignore")

    # 실행 환경 — "prod"에서는 기본 시크릿 사용 시 기동 거부
    environment: str = "dev"

    # API
    api_prefix: str = "/api/v3"
    dataops_version: str = "3.0.0-R3"
    cors_origins: list[str] = ["http://localhost:8000", "http://localhost:5173"]

    # 데이터 소스 (메타데이터 카탈로그 시드)
    mock_data_path: Path = _DEFAULT_MOCK

    # 인증 (HS256) — 시크릿은 반드시 환경변수로 주입
    jwt_secret: str = _DEFAULT_JWT_SECRET
    jwt_algorithm: str = "HS256"
    jwt_expiry_seconds: int = 3600
    jwt_scope: str = "data:read data:write"

    # DataOps 기본 페이징
    default_page_size: int = 20

    # MLOps 드리프트/이상치 임계 (Notion 명세)
    psi_threshold: float = 0.2
    kl_threshold: float = 0.1
    zscore_threshold: float = 3.0
    iqr_multiplier: float = 1.5

    # 오케스트레이션 — 자동 롤백 임계 지연(ms)
    rollback_latency_ms: float = 200.0
    # 동일 model_id 재학습 최소 간격(분) — manual 이벤트는 무시
    retrain_min_interval_minutes: float = 30.0

    def validate_runtime(self) -> None:
        """기동 시 정합성 검증 — prod에서 기본 JWT 시크릿이면 거부."""
        if self.environment == "prod" and self.jwt_secret == _DEFAULT_JWT_SECRET:
            raise RuntimeError(
                "XOPS_ENVIRONMENT=prod에서는 XOPS_JWT_SECRET을 반드시 설정해야 합니다 (기본 시크릿 사용 불가)."
            )


@lru_cache
def get_settings() -> Settings:
    """설정 싱글톤."""
    return Settings()
