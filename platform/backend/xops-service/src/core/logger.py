"""구조화 JSON 로깅 — print() 금지."""

from __future__ import annotations

import io
import json
import logging
import sys
from datetime import datetime, timezone
from typing import TextIO


class JsonFormatter(logging.Formatter):
    """로그 레코드를 한 줄 JSON으로 직렬화."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _utf8_stream() -> TextIO:
    """콘솔 코드페이지와 무관하게 UTF-8로 쓰는 스트림.

    Windows 기본 콘솔(cp949)에서는 `—`·일부 한글이 인코딩되지 않아 로그 **레코드가 통째로
    유실**된다(UnicodeEncodeError → logging 내부 오류). 운영 로그와 degrade 경고를 잃지
    않도록 UTF-8 + backslashreplace 로 감싼다. 콘솔 표시가 깨지는 것은 감수하되 기록은 남는다.
    """
    try:
        return io.TextIOWrapper(
            open(sys.stdout.fileno(), "wb", closefd=False),
            encoding="utf-8",
            errors="backslashreplace",
            line_buffering=True,
        )
    except (OSError, ValueError):  # fileno 를 쓸 수 없는 환경(pytest capture 등)
        return sys.stdout


def get_logger(name: str = "xops") -> logging.Logger:
    """JSON 핸들러가 붙은 로거 반환 (중복 핸들러 방지)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(_utf8_stream())
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
