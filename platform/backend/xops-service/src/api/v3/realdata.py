"""실데이터 연계 라우터 골격(M0) — `health`·`dong-map` 2개 라우트만 등록한다.

datasets·training-runs·candidates·evaluation·explain·drift·analysis 라우트는 이후
작성자(A·B·C)가 추가한다(G0 §7 소유·경계). 이 파일은 그 전까지 봉투 계약(R6)의
최소 실동 예시로만 쓴다.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends

from src.api.dependencies import require_auth
from src.core.settings import get_settings
from src.realdata.pg_reader import ALLOWED_TABLES, RealdataUnavailable, fetch_table_existence
from src.schemas.realdata import Envelope, Provenance

router = APIRouter(prefix="/realdata", tags=["realdata"])


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _envelope(status: str, *, message: str | None = None, data: Any = None, **provenance: Any) -> dict[str, Any]:
    return Envelope[Any](
        status=status,  # type: ignore[arg-type]
        message=message,
        data=data,
        provenance=Provenance(computed_at=_now(), **provenance),
    ).model_dump(mode="json")


@router.get("/health")
def health() -> dict[str, Any]:
    """PG DSN 유무·허용 테이블 전체의 존재 여부 — 인증 없음(조회 전용 상태 확인)."""
    settings = get_settings()
    if not settings.pg_dsn:
        return _envelope("error", message="XOPS_PG_DSN이 설정되지 않았습니다.")
    try:
        tables = fetch_table_existence(sorted(ALLOWED_TABLES))
    except RealdataUnavailable as exc:
        return _envelope("error", message=str(exc))
    return _envelope("ok", data={"pg_dsn_configured": True, "table_status": tables})


@router.get("/dong-map")
def dong_map(_: dict[str, Any] = Depends(require_auth("data:read"))) -> dict[str, Any]:
    """남원 행정동 대응표(KT dong_code ↔ dong_name, 23건) — `data:read`."""
    path = get_settings().realdata_dong_map_path
    if not path.exists():
        return _envelope("error", message=f"행정동 대응표 파일이 없습니다: {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    return _envelope("ok", data=doc, rules_version=doc.get("version"))
