"""DataOps 엔드포인트 — Data API Builder(CRUD/필터/정렬/페이징) + JWT 인증.

프론트 계약: GET /api/v3/dataops/{source_id} 등. 저장소 접근은 메타데이터로 추상화.
정적 경로(token·oauth2·catalog)를 {source_id} 동적 경로보다 먼저 선언해 충돌 방지.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from src.api.dependencies import require_auth
from src.auth.jwt import issue_jwt, issue_oauth2
from src.dataops.catalog import get_catalog
from src.dataops.service import DataService
from src.schemas.dataops import SourceSummary, TokenResponse, WriteBody

router = APIRouter(prefix="/dataops", tags=["dataops"])
_service = DataService()


# ── 인증 발급 ──────────────────────────────────────────────
@router.post("/token/{source_id}", response_model=TokenResponse)
def issue_token(source_id: str) -> TokenResponse:
    """소스 접근용 JWT 발급 (HS256, scope data:read data:write)."""
    from src.core.settings import get_settings

    return TokenResponse(access_token=issue_jwt(source_id), scope=get_settings().jwt_scope)


@router.post("/oauth2/{source_id}")
def issue_oauth2_token(source_id: str) -> dict[str, Any]:
    """OAuth2 Authorization Code Grant 흐름 발급."""
    return issue_oauth2(source_id)


# ── 카탈로그 ───────────────────────────────────────────────
@router.get("/catalog", response_model=list[SourceSummary])
def list_catalog(q: str = Query("", description="소스명·태그·설명·객체명 부분 일치 검색")) -> list[dict[str, Any]]:
    """메타데이터 카탈로그 목록/검색."""
    return get_catalog().search(q)


@router.get("/catalog/{source_id}", response_model=SourceSummary)
def get_source(source_id: str) -> dict[str, Any]:
    """단일 소스 메타데이터 조회."""
    return get_catalog().get(source_id)


# ── CRUD (가상화 API) ──────────────────────────────────────
@router.get("/{source_id}")
def read(
    source_id: str,
    filter: str | None = Query(None, description="단일 조건 `col op value`"),
    sort: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int | None = Query(None, ge=1, le=200),
    payload: dict[str, Any] = Depends(require_auth("data:read")),
) -> dict[str, Any]:
    """GET — 필터/정렬/페이징으로 조회."""
    schema = get_catalog().get(source_id)
    return _service.execute(
        method="GET", schema=schema, payload=payload, filter_expr=filter, sort=sort, page=page, page_size=page_size
    )


@router.post("/{source_id}")
def create(
    source_id: str,
    body: WriteBody | None = None,
    payload: dict[str, Any] = Depends(require_auth("data:write")),
) -> dict[str, Any]:
    """POST — 신규 행 생성."""
    return _service.execute(method="POST", schema=get_catalog().get(source_id), payload=payload)


@router.put("/{source_id}")
def replace(
    source_id: str,
    body: WriteBody | None = None,
    filter: str | None = Query(None),
    payload: dict[str, Any] = Depends(require_auth("data:write")),
) -> dict[str, Any]:
    """PUT — 조건에 맞는 행 치환."""
    return _service.execute(method="PUT", schema=get_catalog().get(source_id), payload=payload, filter_expr=filter)


@router.patch("/{source_id}")
def modify(
    source_id: str,
    body: WriteBody | None = None,
    filter: str | None = Query(None),
    payload: dict[str, Any] = Depends(require_auth("data:write")),
) -> dict[str, Any]:
    """PATCH — 부분 수정."""
    return _service.execute(method="PATCH", schema=get_catalog().get(source_id), payload=payload, filter_expr=filter)


@router.delete("/{source_id}")
def remove(
    source_id: str,
    filter: str | None = Query(None),
    payload: dict[str, Any] = Depends(require_auth("data:write")),
) -> dict[str, Any]:
    """DELETE — 조건에 맞는 행 삭제."""
    return _service.execute(method="DELETE", schema=get_catalog().get(source_id), payload=payload, filter_expr=filter)
