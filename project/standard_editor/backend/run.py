"""
서버 실행 스크립트
백엔드 디렉토리에서 실행할 때 사용합니다.
"""
import sys
from pathlib import Path

# 백엔드 디렉토리를 Python 경로에 추가
backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

if __name__ == "__main__":
    import uvicorn
    from core.logger import logger
    
    logger.info("Starting FastAPI server...")
    logger.info(f"Backend directory: {backend_dir}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(backend_dir)]
    )

