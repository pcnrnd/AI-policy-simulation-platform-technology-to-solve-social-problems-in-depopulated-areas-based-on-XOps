"""
시스템 요구사항 확인
Git, DVC 등 필수 도구의 설치 여부를 확인합니다.
"""
import shutil
from typing import Dict, Optional
from core.logger import logger


def check_command_installed(command: str) -> tuple[bool, Optional[str]]:
    """
    명령어가 설치되어 있는지 확인
    
    Args:
        command: 확인할 명령어 이름
        
    Returns:
        (설치 여부, 버전 정보)
    """
    try:
        path = shutil.which(command)
        if path is None:
            return False, None
        
        # 버전 정보 가져오기 시도
        import subprocess
        try:
            result = subprocess.run(
                [command, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            version = result.stdout.strip().split('\n')[0] if result.returncode == 0 else None
            return True, version
        except Exception:
            return True, None
    except Exception as e:
        logger.error(f"Error checking command {command}: {str(e)}")
        return False, None


def check_system_requirements() -> Dict:
    """
    시스템 요구사항 확인
    
    Returns:
        시스템 요구사항 상태 딕셔너리
    """
    git_installed, git_version = check_command_installed("git")
    dvc_installed, dvc_version = check_command_installed("dvc")
    
    all_installed = git_installed and dvc_installed
    
    return {
        "git": {
            "installed": git_installed,
            "version": git_version
        },
        "dvc": {
            "installed": dvc_installed,
            "version": dvc_version
        },
        "all_requirements_met": all_installed
    }

