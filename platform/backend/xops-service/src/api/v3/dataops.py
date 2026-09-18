"""DataOps 엔드포인트 — Data API Builder(CRUD/필터/정렬/페이징) + JWT 인증.

프론트 계약: GET /api/v3/dataops/{source_id} 등. 저장소 접근은 메타데이터로 추상화.
정적 경로(token·oauth2·catalog)를 {source_id} 동적 경로보다 먼저 선언해 충돌 방지.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from src.api.dependencies import require_auth, require_client
from src.auth.jwt import issue_jwt, issue_oauth2
from src.core import db
from src.core.exceptions import SourceNotFoundError
from src.core.settings import get_settings
from src.dataops.catalog import get_catalog
from src.dataops.liveness import annotate
from src.dataops.service import DataService
from src.schemas.dataops import (
    ArchiveRegisterRequest,
    BuiltApiRequest,
    BuiltApiSummary,
    SourceSummary,
    TokenResponse,
    WriteBody,
)

router = APIRouter(prefix="/dataops", tags=["dataops"])
_service = DataService()


# ── 인증 발급 ──────────────────────────────────────────────
@router.post("/token", response_model=TokenResponse)
def issue_catalog_token(_: None = Depends(require_client)) -> TokenResponse:
    """소스에 매이지 않은 JWT 발급.

    `/token/{source_id}` 는 미존재 소스에 유효 토큰을 내주지 않도록 카탈로그 존재를 검사한다(P-D2).
    그런데 카탈로그 등록은 "아직 없는 소스"를 만드는 요청이라 그 검사와 태생적으로 충돌해,
    소스를 고르지 않은 상태에서는 등록용 토큰을 받을 수 없었다. 권한은 scope 로만 판정하므로
    소스 표기가 없는 토큰을 따로 내준다 — 기존 경로와 검증 규칙은 그대로다.
    """
    return TokenResponse(access_token=issue_jwt(), scope=get_settings().jwt_scope)


@router.post("/token/{source_id}", response_model=TokenResponse)
def issue_token(source_id: str, _: None = Depends(require_client)) -> TokenResponse:
    """소스 접근용 JWT 발급 (HS256, scope data:read data:write)."""
    get_catalog().get(source_id)  # 카탈로그에 없으면 SourceNotFoundError(404) — 미존재 소스에 유효 토큰 발급 방지
    return TokenResponse(access_token=issue_jwt(source_id), scope=get_settings().jwt_scope)


@router.post("/oauth2")
def issue_catalog_oauth2(_: None = Depends(require_client)) -> dict[str, Any]:
    """소스에 매이지 않은 OAuth2 발급 — 위 `/token` 과 같은 이유."""
    return issue_oauth2()


@router.post("/oauth2/{source_id}")
def issue_oauth2_token(source_id: str, _: None = Depends(require_client)) -> dict[str, Any]:
    """OAuth2 Authorization Code Grant 흐름 발급."""
    # 위 /token 경로와 같은 검증 — 두 경로가 같은 access_token을 내주므로 카탈로그 존재 검사도
    # 같아야 한다. 이 가드가 없으면 미존재 source_id로도 유효 토큰이 발급됐다(P-D2).
    get_catalog().get(source_id)  # 카탈로그에 없으면 SourceNotFoundError(404)
    return issue_oauth2(source_id)


# ── 카탈로그 ───────────────────────────────────────────────
@router.get("/catalog", response_model=list[SourceSummary])
def list_catalog(
    q: str = Query("", description="소스명·태그·설명·객체명 부분 일치 검색"),
    live: bool = Query(False, description="각 소스의 실적재 행수(live_rows)를 함께 조회"),
    include_seed: bool = Query(False, description="데모 시드 소스(is_seed)까지 포함 — 데모 표시 ON 전용"),
) -> list[dict[str, Any]]:
    """메타데이터 카탈로그 목록/검색.

    `live=true` 면 소스별로 저장소를 읽어 `live_rows` 를 덧붙인다. 등록만 되어 있고 실제로는
    적재되지 않은 소스를 가려내기 위한 것이다. 기본값이 false 이므로 기존 응답은 그대로다.
    확인하지 못한 소스(DSN 부재·드라이버 미설치·연결 실패)는 0이 아니라 null 이 된다.

    `include_seed` 의 기본값은 false — 목록의 기본은 실데이터(+사용자 등록분)다. 데모 표시 ON
    화면만 `include_seed=true` 로 시드까지 받는다. 단건 조회·토큰 발급·CRUD 는 이 필터와
    무관하다(데모 ON에서 고른 시드 소스가 곧바로 404가 되면 안 된다).
    """
    sources = get_catalog().search(q, include_seed=include_seed)
    return annotate(sources) if live else sources


@router.get("/catalog/{source_id}", response_model=SourceSummary)
def get_source(source_id: str) -> dict[str, Any]:
    """단일 소스 메타데이터 조회."""
    return get_catalog().get(source_id)


# 카탈로그 쓰기는 `data:write` 요구 — 등록된 object·컬럼명·range 가 생성 쿼리에 조립되므로
# 조회(위 GET 2개)와 달리 공개로 둘 수 없다. payload를 쓰지 않으므로 발급 게이트와 같이 `_`.
@router.post("/catalog", response_model=SourceSummary, status_code=201)
def register_source(
    body: ArchiveRegisterRequest,
    _: dict[str, Any] = Depends(require_auth("data:write")),
) -> dict[str, Any]:
    """신규 아카이브(사용자 소스) 등록 → 카탈로그 병합. 등록 즉시 가상화 API 대상이 됨."""
    return get_catalog().add(body.to_schema())


@router.delete("/catalog/{source_id}")
def delete_source(
    source_id: str,
    _: dict[str, Any] = Depends(require_auth("data:write")),
) -> dict[str, str]:
    """사용자 등록 소스 삭제 (기본 시드 소스는 보호)."""
    get_catalog().remove(source_id)
    return {"deleted": source_id}


# ── 발급 API 목록 (API 빌드·등록 결과) ─────────────────────
# `/{source_id}` 동적 경로보다 먼저 선언한다 — 뒤에 두면 GET /apis 가 소스 조회로 잡힌다.
def _to_summary(api: dict[str, Any]) -> dict[str, Any]:
    return {**api, "endpoint": f"{get_settings().api_prefix}/dataops/{api['source_id']}"}


@router.get("/apis", response_model=list[BuiltApiSummary])
def list_built_apis() -> list[dict[str, Any]]:
    """빌드·등록된 API 목록 (최근 순). 조회는 카탈로그 GET과 같이 공개."""
    return [_to_summary(api) for api in db.list_built_apis()]


@router.post("/apis", response_model=BuiltApiSummary, status_code=201)
def register_built_api(
    body: BuiltApiRequest,
    _: dict[str, Any] = Depends(require_auth("data:write")),
) -> dict[str, Any]:
    """API 구성 등록 — 같은 id 재등록은 덮어쓴다. 대상 소스가 없으면 404."""
    get_catalog().get(body.source_id)
    return _to_summary(db.upsert_built_api(body.model_dump()))


@router.delete("/apis/{api_id}")
def delete_built_api(
    api_id: str,
    _: dict[str, Any] = Depends(require_auth("data:write")),
) -> dict[str, str]:
    """발급 API 제거."""
    if not db.delete_built_api(api_id):
        raise SourceNotFoundError(f"발급 API를 찾을 수 없습니다: {api_id}")
    return {"deleted": api_id}


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
    """POST — 신규 행 생성. body.data 가 있으면 실 저장소에 파라미터 바인딩으로 INSERT 된다."""
    return _service.execute(
        method="POST", schema=get_catalog().get(source_id), payload=payload, values=body.data if body else None
    )


@router.put("/{source_id}")
def replace(
    source_id: str,
    body: WriteBody | None = None,
    filter: str | None = Query(None),
    payload: dict[str, Any] = Depends(require_auth("data:write")),
) -> dict[str, Any]:
    """PUT — 조건에 맞는 행 치환. filter + body.data 가 있으면 실 저장소에 UPDATE 된다."""
    return _service.execute(
        method="PUT",
        schema=get_catalog().get(source_id),
        payload=payload,
        filter_expr=filter,
        values=body.data if body else None,
    )


@router.patch("/{source_id}")
def modify(
    source_id: str,
    body: WriteBody | None = None,
    filter: str | None = Query(None),
    payload: dict[str, Any] = Depends(require_auth("data:write")),
) -> dict[str, Any]:
    """PATCH — 부분 수정. filter + body.data 가 있으면 실 저장소에 UPDATE 된다."""
    return _service.execute(
        method="PATCH",
        schema=get_catalog().get(source_id),
        payload=payload,
        filter_expr=filter,
        values=body.data if body else None,
    )


@router.delete("/{source_id}")
def remove(
    source_id: str,
    filter: str | None = Query(None),
    payload: dict[str, Any] = Depends(require_auth("data:write")),
) -> dict[str, Any]:
    """DELETE — 조건에 맞는 행 삭제."""
    return _service.execute(method="DELETE", schema=get_catalog().get(source_id), payload=payload, filter_expr=filter)
