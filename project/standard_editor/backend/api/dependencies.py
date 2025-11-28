"""
의존성 주입
FastAPI 의존성을 정의합니다.
"""
from repositories.command_executor import SubprocessCommandExecutor
from services.dvc_service import DVCService
from services.git_service import GitService
from services.repository_service import RepositoryService


# 전역 인스턴스 (싱글톤 패턴)
_command_executor = None
_dvc_service = None
_git_service = None
_repository_service = None


def get_command_executor() -> SubprocessCommandExecutor:
    """명령어 실행기 인스턴스 반환"""
    global _command_executor
    if _command_executor is None:
        _command_executor = SubprocessCommandExecutor()
    return _command_executor


def get_dvc_service() -> DVCService:
    """DVC 서비스 인스턴스 반환"""
    global _dvc_service
    if _dvc_service is None:
        executor = get_command_executor()
        _dvc_service = DVCService(executor)
    return _dvc_service


def get_git_service() -> GitService:
    """Git 서비스 인스턴스 반환"""
    global _git_service
    if _git_service is None:
        executor = get_command_executor()
        _git_service = GitService(executor)
    return _git_service


def get_repository_service() -> RepositoryService:
    """Repository 서비스 인스턴스 반환"""
    global _repository_service
    if _repository_service is None:
        dvc_service = get_dvc_service()
        git_service = get_git_service()
        _repository_service = RepositoryService(dvc_service, git_service)
    return _repository_service

