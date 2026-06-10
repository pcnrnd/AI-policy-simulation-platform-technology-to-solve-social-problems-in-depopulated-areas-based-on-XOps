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
    
    def has_changes(self, path: Optional[str] = None) -> bool:
        """
        커밋할 변경사항이 있는지 확인
        
        Args:
            path: 작업 디렉토리 경로
            
        Returns:
            변경사항 존재 여부
        """
        try:
            # git status로 변경사항 확인
            result = self.executor.execute(
                ["git", "status", "--porcelain"],
                cwd=path,
                json_output=False
            )
            # --porcelain 옵션은 변경사항이 있으면 출력이 있고, 없으면 빈 문자열
            stdout = result.get("stdout", "").strip()
            return len(stdout) > 0
        except Exception as e:
            logger.warning(f"Failed to check changes: {str(e)}")
            # 에러 발생 시 변경사항이 있다고 가정 (안전한 선택)
            return True
    
    def commit(self, message: str, path: Optional[str] = None, allow_empty: bool = False) -> Dict:
        """
        변경사항 커밋
        
        Args:
            message: 커밋 메시지
            path: 작업 디렉토리 경로
            allow_empty: 빈 커밋 허용 여부
            
        Returns:
            커밋 결과
        """
        try:
            # 먼저 add all
            self.executor.execute(["git", "add", "."], cwd=path, json_output=False)
            
            # 변경사항 확인
            if not allow_empty and not self.has_changes(path):
                raise RepositoryError("커밋할 변경사항이 없습니다.")
            
            # 커밋
            command = ["git", "commit", "-m", message]
            if allow_empty:
                command.append("--allow-empty")
            
            result = self.executor.execute(command, cwd=path, json_output=False)
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
    
    def remote_add(self, name: str, url: str, path: Optional[str] = None) -> Dict:
        """
        원격 저장소 추가
        
        Args:
            name: 원격 저장소 이름 (예: origin)
            url: 원격 저장소 URL (예: https://github.com/user/repo.git)
            path: 작업 디렉토리 경로
            
        Returns:
            추가 결과
        """
        try:
            result = self.executor.execute(
                ["git", "remote", "add", name, url],
                cwd=path,
                json_output=False
            )
            return result
        except CommandExecutionError as e:
            # 이미 존재하는 경우 무시
            if "already exists" in e.message.lower():
                logger.info(f"Remote {name} already exists")
                return {"message": f"Remote {name} already exists", "skipped": True}
            logger.error(f"Git remote add failed: {e.message}")
            raise RepositoryError(f"Git remote add failed: {e.message}")
    
    def remote_list(self, path: Optional[str] = None) -> Dict:
        """
        원격 저장소 목록 조회
        
        Args:
            path: 작업 디렉토리 경로
            
        Returns:
            원격 저장소 목록
        """
        try:
            result = self.executor.execute(
                ["git", "remote", "-v"],
                cwd=path,
                json_output=False
            )
            # 결과 파싱
            remotes = {}
            stdout = result.get("stdout", "")
            for line in stdout.strip().split('\n'):
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        remote_name = parts[0]
                        remote_url = parts[1]
                        remotes[remote_name] = remote_url
            
            return {
                "remotes": remotes,
                "stdout": stdout
            }
        except CommandExecutionError as e:
            logger.error(f"Git remote list failed: {e.message}")
            raise RepositoryError(f"Git remote list failed: {e.message}")
    
    def remote_remove(self, name: str, path: Optional[str] = None) -> Dict:
        """
        원격 저장소 제거
        
        Args:
            name: 원격 저장소 이름
            path: 작업 디렉토리 경로
            
        Returns:
            제거 결과
        """
        try:
            result = self.executor.execute(
                ["git", "remote", "remove", name],
                cwd=path,
                json_output=False
            )
            return result
        except CommandExecutionError as e:
            logger.error(f"Git remote remove failed: {e.message}")
            raise RepositoryError(f"Git remote remove failed: {e.message}")
    
    def remote_set_url(self, name: str, url: str, path: Optional[str] = None) -> Dict:
        """
        원격 저장소 URL 변경
        
        Args:
            name: 원격 저장소 이름
            url: 새로운 URL
            path: 작업 디렉토리 경로
            
        Returns:
            변경 결과
        """
        try:
            result = self.executor.execute(
                ["git", "remote", "set-url", name, url],
                cwd=path,
                json_output=False
            )
            return result
        except CommandExecutionError as e:
            logger.error(f"Git remote set-url failed: {e.message}")
            raise RepositoryError(f"Git remote set-url failed: {e.message}")

