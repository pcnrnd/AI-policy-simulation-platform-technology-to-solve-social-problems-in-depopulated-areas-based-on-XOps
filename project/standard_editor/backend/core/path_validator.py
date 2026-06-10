"""
경로 검증 유틸리티
경로 관련 검증 및 정규화를 담당합니다.
"""
import os
from pathlib import Path
from typing import Optional
from core.exceptions import ValidationError
from core.logger import logger


def validate_path(path: str, must_exist: bool = False, must_be_dir: bool = False) -> Path:
    """
    경로 검증 및 정규화
    
    Args:
        path: 검증할 경로
        must_exist: 경로가 존재해야 하는지 여부
        must_be_dir: 경로가 디렉토리여야 하는지 여부
        
    Returns:
        정규화된 Path 객체
        
    Raises:
        ValidationError: 경로가 유효하지 않은 경우
    """
    if not path or not isinstance(path, str):
        raise ValidationError("경로가 제공되지 않았습니다.")
    
    # 경로 정규화
    try:
        normalized_path = Path(path).resolve()
    except (OSError, ValueError) as e:
        raise ValidationError(f"경로를 정규화할 수 없습니다: {str(e)}")
    
    # 경로 존재 여부 확인
    if must_exist and not normalized_path.exists():
        raise ValidationError(f"경로가 존재하지 않습니다: {normalized_path}")
    
    # 디렉토리 여부 확인
    if must_exist and must_be_dir and not normalized_path.is_dir():
        raise ValidationError(f"경로가 디렉토리가 아닙니다: {normalized_path}")
    
    if must_exist and not must_be_dir and normalized_path.is_dir():
        raise ValidationError(f"경로가 파일이 아닙니다: {normalized_path}")
    
    return normalized_path


def ensure_directory_exists(path: str) -> Path:
    """
    디렉토리가 존재하는지 확인하고, 없으면 생성
    
    Args:
        path: 디렉토리 경로
        
    Returns:
        Path 객체
        
    Raises:
        ValidationError: 디렉토리를 생성할 수 없는 경우
    """
    try:
        dir_path = Path(path).resolve()
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    except (OSError, PermissionError) as e:
        raise ValidationError(f"디렉토리를 생성할 수 없습니다: {str(e)}")


def check_repository_initialized(path: str) -> dict:
    """
    저장소가 이미 초기화되어 있는지 확인
    
    Args:
        path: 저장소 경로
        
    Returns:
        초기화 상태 딕셔너리 (git_initialized, dvc_initialized)
    """
    repo_path = Path(path)
    
    git_initialized = (repo_path / ".git").exists()
    dvc_initialized = (repo_path / ".dvc").exists()
    
    return {
        "git_initialized": git_initialized,
        "dvc_initialized": dvc_initialized,
        "path": str(repo_path)
    }

