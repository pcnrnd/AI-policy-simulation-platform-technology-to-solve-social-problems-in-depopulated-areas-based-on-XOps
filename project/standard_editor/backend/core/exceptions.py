"""
커스텀 예외 클래스
애플리케이션 전반에서 사용되는 예외를 정의합니다.
"""


class CommandExecutionError(Exception):
    """명령어 실행 실패 예외"""
    
    def __init__(self, message: str, command: str = None, return_code: int = None):
        self.message = message
        self.command = command
        self.return_code = return_code
        super().__init__(self.message)


class RepositoryError(Exception):
    """Repository 관련 오류 예외"""
    pass


class BranchError(Exception):
    """Branch 관련 오류 예외"""
    pass


class CommitError(Exception):
    """Commit 관련 오류 예외"""
    pass


class ValidationError(Exception):
    """입력 검증 오류 예외"""
    pass

