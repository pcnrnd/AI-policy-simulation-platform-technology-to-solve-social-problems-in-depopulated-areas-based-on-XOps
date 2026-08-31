FROM python:3.11-slim

WORKDIR /app

# 의존성 먼저 설치 (레이어 캐시)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 실 저장소 드라이버 — compose 의 XOPS_*_DSN 을 실제로 쓰려면 이미지에 있어야 한다.
# 별 레이어로 두어 런타임 의존성 캐시를 깨지 않는다. 빼고 빌드해도 InMemory 로 degrade한다.
COPY requirements-db.txt .
RUN pip install --no-cache-dir -r requirements-db.txt

# 애플리케이션 소스 (임포트가 `from src...`이므로 cwd=/app 유지)
COPY main.py ./
COPY src ./src

# SQLite 영속화 볼륨 마운트 지점
VOLUME ["/app/data"]
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
