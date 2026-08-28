FROM python:3.10-slim

# uv 바이너리만 복사 (pip 미사용)
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /bin/

WORKDIR /app

# venv 는 /app 밖에 둔다 — compose 가 .:/app 을 마운트해도 가려지지 않도록
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=180 \
    PATH="/opt/venv/bin:$PATH"

# 의존성 레이어 — 소스가 바뀌어도 uv.lock 이 같으면 재설치하지 않음
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-cache

COPY . .

# 한 이미지로 두 앱을 기동한다. APP / PORT 로 선택:
#   chart_speed (응답속도): docker run -p 8505:8505 <image>                                  (기본값)
#   chart_type  (유형)    : docker run -e APP=chart_type -e PORT=8506 -p 8506:8506 <image>
ENV APP=chart_speed \
    PORT=8505

EXPOSE 8505 8506

CMD ["sh", "-c", "exec streamlit run ${APP}.py --server.port ${PORT} --server.address 0.0.0.0"]
