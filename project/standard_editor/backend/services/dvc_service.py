"""
DVC 서비스
DVC 명령어 실행을 담당하는 서비스 클래스입니다.
"""
from typing import Dict, List, Optional
from pathlib import Path

from repositories.command_executor import CommandExecutor
from core.exceptions import CommandExecutionError, RepositoryError
from core.logger import logger
from config.settings import settings


class DVCService:
    """DVC 명령어 실행 서비스"""
    
    def __init__(self, executor: CommandExecutor):
        """
        초기화
        
        Args:
            executor: 명령어 실행기
        """
        self.executor = executor
    
    def status(self, path: Optional[str] = None) -> Dict:
        """
        DVC 상태 확인
        
        Args:
            path: 작업 디렉토리 경로
            
        Returns:
            DVC 상태 정보
        """
        try:
            result = self.executor.execute(["dvc", "status"], cwd=path)
            return result
        except CommandExecutionError as e:
            logger.error(f"DVC status failed: {e.message}")
            raise RepositoryError(f"DVC status failed: {e.message}")
    
    def add(self, files: List[str], path: Optional[str] = None, force: bool = False) -> Dict:
        """
        파일을 DVC에 추가
        
        Args:
            files: 추가할 파일 경로 리스트
            path: 작업 디렉토리 경로
            force: 강제 추가 여부
            
        Returns:
            추가 결과
        """
        try:
            command = ["dvc", "add"]
            if force:
                command.append("-f")
            command.extend(files)
            
            result = self.executor.execute(command, cwd=path, json_output=False)
            return result
        except CommandExecutionError as e:
            logger.error(f"DVC add failed: {e.message}")
            raise RepositoryError(f"DVC add failed: {e.message}")
    
    def commit(self, message: str, path: Optional[str] = None) -> Dict:
        """
        변경사항 커밋
        
        Args:
            message: 커밋 메시지
            path: 작업 디렉토리 경로
            
        Returns:
            커밋 결과
        """
        try:
            result = self.executor.execute(
                ["dvc", "commit", "-m", message],
                cwd=path,
                json_output=False
            )
            return result
        except CommandExecutionError as e:
            logger.error(f"DVC commit failed: {e.message}")
            raise RepositoryError(f"DVC commit failed: {e.message}")
    
    def push(self, path: Optional[str] = None, remote: Optional[str] = None) -> Dict:
        """
        원격 저장소로 푸시
        
        Args:
            path: 작업 디렉토리 경로
            remote: 원격 저장소 이름
            
        Returns:
            푸시 결과
        """
        try:
            command = ["dvc", "push"]
            if remote:
                command.extend(["-r", remote])
            
            result = self.executor.execute(command, cwd=path, json_output=False)
            return result
        except CommandExecutionError as e:
            logger.error(f"DVC push failed: {e.message}")
            raise RepositoryError(f"DVC push failed: {e.message}")
    
    def pull(self, path: Optional[str] = None, remote: Optional[str] = None) -> Dict:
        """
        원격 저장소에서 풀
        
        Args:
            path: 작업 디렉토리 경로
            remote: 원격 저장소 이름
            
        Returns:
            풀 결과
        """
        try:
            command = ["dvc", "pull"]
            if remote:
                command.extend(["-r", remote])
            
            result = self.executor.execute(command, cwd=path, json_output=False)
            return result
        except CommandExecutionError as e:
            logger.error(f"DVC pull failed: {e.message}")
            raise RepositoryError(f"DVC pull failed: {e.message}")
    
    def init(self, path: Optional[str] = None) -> Dict:
        """
        DVC 저장소 초기화
        
        Args:
            path: 작업 디렉토리 경로
            
        Returns:
            초기화 결과
        """
        try:
            result = self.executor.execute(["dvc", "init"], cwd=path, json_output=False)
            return result
        except CommandExecutionError as e:
            logger.error(f"DVC init failed: {e.message}")
            raise RepositoryError(f"DVC init failed: {e.message}")
    
    def remote_add(
        self, 
        name: str, 
        url: str, 
        path: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None
    ) -> Dict:
        """
        원격 저장소 추가
        
        Args:
            name: 원격 저장소 이름
            url: 원격 저장소 URL
            path: 작업 디렉토리 경로
            endpoint_url: 엔드포인트 URL (S3/MinIO)
            access_key_id: 액세스 키
            secret_access_key: 시크릿 키
            
        Returns:
            추가 결과
        """
        try:
            # 원격 저장소 추가
            result = self.executor.execute(
                ["dvc", "remote", "add", name, url],
                cwd=path,
                json_output=False
            )
            
            # S3/MinIO 설정
            if endpoint_url:
                self.executor.execute(
                    ["dvc", "remote", "modify", name, "endpointurl", endpoint_url],
                    cwd=path,
                    json_output=False
                )
            
            if access_key_id:
                self.executor.execute(
                    ["dvc", "remote", "modify", name, "access_key_id", access_key_id],
                    cwd=path,
                    json_output=False
                )
            
            if secret_access_key:
                self.executor.execute(
                    ["dvc", "remote", "modify", name, "secret_access_key", secret_access_key],
                    cwd=path,
                    json_output=False
                )
            
            return result
        except CommandExecutionError as e:
            logger.error(f"DVC remote add failed: {e.message}")
            raise RepositoryError(f"DVC remote add failed: {e.message}")

