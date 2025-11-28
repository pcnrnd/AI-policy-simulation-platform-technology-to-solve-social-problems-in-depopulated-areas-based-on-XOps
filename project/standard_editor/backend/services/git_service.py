"""
Git 서비스
Git 명령어 실행을 담당하는 서비스 클래스입니다.
"""
from typing import Dict, List, Optional

from repositories.command_executor import CommandExecutor
from core.exceptions import CommandExecutionError, RepositoryError, BranchError
from core.logger import logger


class GitService:
    """Git 명령어 실행 서비스"""
    
    def __init__(self, executor: CommandExecutor):
        """
        초기화
        
        Args:
            executor: 명령어 실행기
        """
        self.executor = executor
    
    def init(self, path: Optional[str] = None) -> Dict:
        """
        Git 저장소 초기화
        
        Args:
            path: 작업 디렉토리 경로
            
        Returns:
            초기화 결과
        """
        try:
            result = self.executor.execute(["git", "init"], cwd=path, json_output=False)
            return result
        except CommandExecutionError as e:
            logger.error(f"Git init failed: {e.message}")
            raise RepositoryError(f"Git init failed: {e.message}")
    
    def checkout(self, branch: str, path: Optional[str] = None, create: bool = False) -> Dict:
        """
        브랜치 체크아웃
        
        Args:
            branch: 브랜치 이름
            path: 작업 디렉토리 경로
            create: 새 브랜치 생성 여부
            
        Returns:
            체크아웃 결과
        """
        try:
            command = ["git", "checkout"]
            if create:
                command.append("-b")
            command.append(branch)
            
            result = self.executor.execute(command, cwd=path, json_output=False)
            return result
        except CommandExecutionError as e:
            logger.error(f"Git checkout failed: {e.message}")
            raise BranchError(f"Git checkout failed: {e.message}")
    
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
            # 먼저 add all
            self.executor.execute(["git", "add", "."], cwd=path, json_output=False)
            
            # 커밋
            result = self.executor.execute(
                ["git", "commit", "-m", message],
                cwd=path,
                json_output=False
            )
            return result
        except CommandExecutionError as e:
            logger.error(f"Git commit failed: {e.message}")
            raise RepositoryError(f"Git commit failed: {e.message}")
    
    def add(self, files: List[str], path: Optional[str] = None) -> Dict:
        """
        파일을 스테이징 영역에 추가
        
        Args:
            files: 추가할 파일 경로 리스트
            path: 작업 디렉토리 경로
            
        Returns:
            추가 결과
        """
        try:
            command = ["git", "add"] + files
            result = self.executor.execute(command, cwd=path, json_output=False)
            return result
        except CommandExecutionError as e:
            logger.error(f"Git add failed: {e.message}")
            raise RepositoryError(f"Git add failed: {e.message}")
    
    def branch_list(self, path: Optional[str] = None) -> Dict:
        """
        브랜치 목록 조회
        
        Args:
            path: 작업 디렉토리 경로
            
        Returns:
            브랜치 목록
        """
        try:
            result = self.executor.execute(
                ["git", "branch", "-a"],
                cwd=path,
                json_output=False
            )
            return result
        except CommandExecutionError as e:
            logger.error(f"Git branch list failed: {e.message}")
            raise BranchError(f"Git branch list failed: {e.message}")
    
    def merge(self, source_branch: str, path: Optional[str] = None) -> Dict:
        """
        브랜치 병합
        
        Args:
            source_branch: 병합할 소스 브랜치
            path: 작업 디렉토리 경로
            
        Returns:
            병합 결과
        """
        try:
            result = self.executor.execute(
                ["git", "merge", source_branch],
                cwd=path,
                json_output=False
            )
            return result
        except CommandExecutionError as e:
            logger.error(f"Git merge failed: {e.message}")
            raise BranchError(f"Git merge failed: {e.message}")
    
    def push(self, remote: str = "origin", branch: Optional[str] = None, path: Optional[str] = None) -> Dict:
        """
        원격 저장소로 푸시
        
        Args:
            remote: 원격 저장소 이름
            branch: 푸시할 브랜치
            path: 작업 디렉토리 경로
            
        Returns:
            푸시 결과
        """
        try:
            command = ["git", "push", remote]
            if branch:
                command.append(branch)
            
            result = self.executor.execute(command, cwd=path, json_output=False)
            return result
        except CommandExecutionError as e:
            logger.error(f"Git push failed: {e.message}")
            raise RepositoryError(f"Git push failed: {e.message}")
    
    def pull(self, remote: str = "origin", branch: Optional[str] = None, path: Optional[str] = None) -> Dict:
        """
        원격 저장소에서 풀
        
        Args:
            remote: 원격 저장소 이름
            branch: 풀할 브랜치
            path: 작업 디렉토리 경로
            
        Returns:
            풀 결과
        """
        try:
            command = ["git", "pull", remote]
            if branch:
                command.append(branch)
            
            result = self.executor.execute(command, cwd=path, json_output=False)
            return result
        except CommandExecutionError as e:
            logger.error(f"Git pull failed: {e.message}")
            raise RepositoryError(f"Git pull failed: {e.message}")

