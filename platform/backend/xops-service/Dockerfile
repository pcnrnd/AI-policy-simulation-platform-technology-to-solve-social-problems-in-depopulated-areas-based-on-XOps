FROM python:3.11-slim

WORKDIR /app

# 의존성 먼저 설치 (레이어 캐시)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 소스 (임포트가 `from src...`이므로 cwd=/app 유지)
COPY main.py ./
COPY src ./src

# SQLite 영속화 볼륨 마운트 지점
VOLUME ["/app/data"]
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
