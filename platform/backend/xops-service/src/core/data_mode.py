from contextvars import ContextVar
from urllib.parse import parse_qsl, urlencode

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

_mode = ContextVar("data_mode", default=None)


def current_mode() -> str | None:
    return _mode.get()


def source_mode(schema: dict) -> str:
    return schema.get("mode") or ("mock" if schema.get("is_seed") else "real")


class DataModeMiddleware(BaseHTTPMiddleware):
    """요청의 데이터 모드를 정한다. 헤더가 없으면 실데이터(real)로 본다 — 데모 기록은 `X-Data-Mode: mock`을
    명시한 요청에만 보인다(외부 연동·헤더 없는 호출로 데모 기록이 새지 않게). `include_seed`는 mock 안에서만 의미가 있다."""

    async def dispatch(self, request, call_next):
        explicit = request.headers.get("X-Data-Mode")
        if explicit is not None and explicit not in ("mock", "real"):
            return JSONResponse({"detail": "데이터 모드를 확인하세요."}, status_code=400)
        mode = explicit or "real"
        token = _mode.set(mode)
        try:
            path = request.url.path
            if mode == "mock" and "/realdata/" in path:
                return JSONResponse({"detail": "실데이터 모드에서 데이터를 확인하세요."}, status_code=404)
            if any(section in path for section in ("/monitoring/", "/orchestration/", "/overview/", "/dataops/catalog")):
                query = [(key, value) for key, value in parse_qsl(request.scope["query_string"].decode()) if key != "include_seed"]
                query.append(("include_seed", "true" if mode == "mock" else "false"))
                request.scope["query_string"] = urlencode(query).encode()
            response = await call_next(request)
            if explicit is not None:
                response.headers["X-Data-Mode"] = explicit
            return response
        finally:
            _mode.reset(token)
