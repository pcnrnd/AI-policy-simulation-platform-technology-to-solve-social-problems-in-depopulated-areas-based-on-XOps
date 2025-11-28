"""
로깅 설정 모듈
애플리케이션 전반에서 사용되는 로거를 설정합니다.
"""
import logging
import sys
from pathlib import Path


def setup_logger(name: str = "standard_editor") -> logging.Logger:
    """
    로거 설정
    
    Args:
        name: 로거 이름
        
    Returns:
        설정된 로거 인스턴스
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 포맷터
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    return logger


# 전역 로거 인스턴스
logger = setup_logger()

