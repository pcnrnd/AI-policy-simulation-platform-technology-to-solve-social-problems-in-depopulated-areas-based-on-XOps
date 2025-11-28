"""
명령어 실행 추상화 계층
subprocess를 사용한 명령어 실행을 추상화합니다.
"""
import subprocess
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from pathlib import Path

from core.exceptions import CommandExecutionError
from core.logger import logger
from config.settings import settings


class CommandExecutor(ABC):
    """명령어 실행 추상 클래스"""
    
    @abstractmethod
    def execute(self, command: List[str], cwd: Optional[str] = None) -> Dict:
        """
        명령어 실행
        
        Args:
            command: 실행할 명령어 리스트
            cwd: 작업 디렉토리
            
        Returns:
            실행 결과 딕셔너리
        """
        pass
    
    @abstractmethod
    def validate_command(self, command: List[str]) -> bool:
        """
        명령어 유효성 검증
        
        Args:
            command: 검증할 명령어 리스트
            
        Returns:
            유효성 여부
        """
        pass


class SubprocessCommandExecutor(CommandExecutor):
    """subprocess를 사용한 명령어 실행 구현"""
    
    def __init__(self, timeout: int = None):
        """
        초기화
        
        Args:
            timeout: 명령어 실행 타임아웃 (초)
        """
        self.timeout = timeout or settings.command_timeout
    
    def validate_command(self, command: List[str]) -> bool:
        """명령어 유효성 검증"""
        if not command or not isinstance(command, list):
            return False
        if not all(isinstance(arg, str) for arg in command):
            return False
        return True
    
    def execute(
        self, 
        command: List[str], 
        cwd: Optional[str] = None,
        json_output: bool = True
    ) -> Dict:
        """
        명령어 실행
        
        Args:
            command: 실행할 명령어 리스트
            cwd: 작업 디렉토리
            json_output: JSON 출력 여부
            
        Returns:
            실행 결과 딕셔너리
        """
        if not self.validate_command(command):
            raise ValueError(f"Invalid command: {command}")
        
        # 작업 디렉토리 설정
        working_dir = Path(cwd) if cwd else Path(settings.default_working_directory) if settings.default_working_directory else None
        
        if working_dir and not working_dir.exists():
            raise FileNotFoundError(f"Working directory not found: {working_dir}")
        
        # JSON 출력 옵션 추가
        if json_output and "--show-json" not in command:
            command = command + ["--show-json"]
        
        logger.info(f"Executing command: {' '.join(command)} in {working_dir or 'current directory'}")
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(working_dir) if working_dir else None,
                check=False
            )
            
            if result.returncode == 0:
                # JSON 출력 파싱 시도
                if json_output and result.stdout:
                    try:
                        data = json.loads(result.stdout)
                        return {
                            "success": True,
                            "data": data,
                            "stdout": result.stdout,
                            "stderr": result.stderr
                        }
                    except json.JSONDecodeError:
                        # JSON 파싱 실패 시 일반 출력 반환
                        return {
                            "success": True,
                            "data": result.stdout,
                            "stdout": result.stdout,
                            "stderr": result.stderr
                        }
                else:
                    return {
                        "success": True,
                        "data": result.stdout,
                        "stdout": result.stdout,
                        "stderr": result.stderr
                    }
            else:
                error_msg = result.stderr or result.stdout or "Unknown error"
                logger.error(f"Command failed: {' '.join(command)}, Error: {error_msg}")
                raise CommandExecutionError(
                    message=error_msg,
                    command=' '.join(command),
                    return_code=result.returncode
                )
                
        except subprocess.TimeoutExpired:
            error_msg = f"Command timeout after {self.timeout} seconds"
            logger.error(error_msg)
            raise CommandExecutionError(message=error_msg, command=' '.join(command))
        except Exception as e:
            logger.error(f"Unexpected error executing command: {str(e)}")
            raise CommandExecutionError(message=str(e), command=' '.join(command))

