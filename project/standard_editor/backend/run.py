"""
서버 실행 스크립트
백엔드 디렉토리에서 실행할 때 사용합니다.
"""
import sys
import os
from pathlib import Path

# 백엔드 디렉토리를 Python 경로에 추가
backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

if __name__ == "__main__":
    import uvicorn
    from core.logger import logger
    
    # 환경 변수에서 포트와 호스트 가져오기 (기본값 설정)
    host = os.getenv("HOST", "127.0.0.1")  # 개발 환경: localhost
    port = int(os.getenv("PORT", 8001))    # 기본 포트: 8001
    
    logger.info("Starting FastAPI server...")
    logger.info(f"Backend directory: {backend_dir}")
    logger.info(f"Server will run on http://{host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        reload_dirs=[str(backend_dir)]
    )

