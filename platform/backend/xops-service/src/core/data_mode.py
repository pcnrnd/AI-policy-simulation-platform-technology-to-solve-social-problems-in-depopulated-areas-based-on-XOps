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
    async def dispatch(self, request, call_next):
        mode = request.headers.get("X-Data-Mode")
        if mode is not None and mode not in ("mock", "real"):
            return JSONResponse({"detail": "데이터 모드를 확인하세요."}, status_code=400)
        token = _mode.set(mode)
        try:
            if mode is not None:
                path = request.url.path
                if mode == "mock" and "/realdata/" in path:
                    return JSONResponse({"detail": "실데이터 모드에서 데이터를 확인하세요."}, status_code=404)
                if any(section in path for section in ("/monitoring/", "/orchestration/", "/overview/", "/dataops/catalog")):
                    query = [(key, value) for key, value in parse_qsl(request.scope["query_string"].decode()) if key != "include_seed"]
                    query.append(("include_seed", "true" if mode == "mock" else "false"))
                    request.scope["query_string"] = urlencode(query).encode()
            response = await call_next(request)
            if mode is not None:
                response.headers["X-Data-Mode"] = mode
            return response
        finally:
            _mode.reset(token)
