"""공표 통계 조회 라우터 — 정책 추천 그래프의 관광지점 월별 입장객·생활인구 분기 발표 값.

공개 기관이 공표한 통계라 데모·실데이터 표시 모두 같은 값을 쓴다. 그래서 데모(mock) 요청을 막는
`/realdata/` 경로(협약 실데이터) 밖에 둔다. 테이블은 `load_external_data.py`가 적재한 PG `ext_*`
(허용 목록 `pg_reader.ALLOWED_TABLES`)이고, 출처는 DataOps 카탈로그 ds_14·ds_15에만 적는다.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Query

from src.api.dependencies import require_auth
from src.realdata.pg_reader import RealdataUnavailable, fetch_all
from src.schemas.realdata import Envelope, Provenance

router = APIRouter(prefix="/public-stats", tags=["public-stats"])
_log = logging.getLogger(__name__)

SPOT_TABLE = "ext_kto_spot_monthly_visitors"
LIVING_TABLE = "ext_mois_living_population_quarterly"
SPOT_DATASET = "ds_14_kto_spot_monthly_visitors"  # DataOps 카탈로그 항목
LIVING_DATASET = "ds_15_mois_living_population_quarterly"
MAX_SPOTS = 10  # 한 화면의 사례 선 수 상한 — 그 이상은 요청 오류


def _envelope(status: str, *, message: str | None = None, data: Any = None, **provenance: Any) -> dict[str, Any]:
    return Envelope[Any](
        status=status,  # type: ignore[arg-type]
        message=message,
        data=data,
        provenance=Provenance(computed_at=datetime.now(timezone.utc), **provenance),
    ).model_dump(mode="json")


def _rows(table: str, columns: list[str], where: dict[str, Any], order_by: list[str]) -> list[dict[str, Any]]:
    """테이블 조회. 연결 실패·테이블 미적재 같은 DB 오류는 원문을 로그에만 남기고 RealdataUnavailable로 바꾼다."""
    try:
        return fetch_all(table, columns, where=where, order_by=order_by)
    except RealdataUnavailable:
        raise
    except Exception as exc:  # psycopg 오류 — 드라이버가 선택 의존성이라 넓게 받는다
        _log.warning("공표 통계 조회 실패 table=%s: %s", table, exc)
        raise RealdataUnavailable("공표 통계를 읽지 못했습니다. 적재 상태를 확인하세요.") from exc


@router.get("/spot-visitors")
def spot_visitors(
    spot: list[str] = Query(..., min_length=1, max_length=MAX_SPOTS, description="관광지점 이름(통계표 그대로)"),
    _: dict[str, Any] = Depends(require_auth("data:read")),
) -> dict[str, Any]:
    """관광지점 월별 입장객 합계 — 지점마다 월 순서 계열. 공표된 달만 있다(빈 달은 행 없음)."""
    names = list(dict.fromkeys(name.strip() for name in spot if name.strip()))
    try:
        series = [
            {
                "spot_name": name,
                "points": [
                    {"ym": f"{row['base_ym'] // 100}-{row['base_ym'] % 100:02d}", "value": int(row["visitors_total"])}
                    for row in _rows(SPOT_TABLE, ["base_ym", "visitors_total"], {"spot_name": name}, ["base_ym"])
                ],
            }
            for name in names
        ]
    except RealdataUnavailable as exc:
        return _envelope("error", message=str(exc))
    return _envelope("ok", data={"spots": series}, dataset_id=SPOT_DATASET)


@router.get("/living-population")
def living_population(
    region: str = Query(..., min_length=1, max_length=40, description="지역 이름(예: 전북특별자치도 남원시)"),
    _: dict[str, Any] = Depends(require_auth("data:read")),
) -> dict[str, Any]:
    """지역의 생활인구 분기 발표 값 — 등록인구·체류인구(발표 단위 그대로), 분기 순서."""
    name = region.strip()
    columns = ["base_year", "quarter", "registered_population", "stay_population", "unit"]
    try:
        rows = _rows(LIVING_TABLE, columns, {"sigungu_name": name}, ["base_year", "quarter"])
    except RealdataUnavailable as exc:
        return _envelope("error", message=str(exc))
    units = {row["unit"] for row in rows}
    if len(units) > 1:
        return _envelope("error", message=f"한 지역의 단위가 섞여 있습니다: {sorted(units)}")
    points = [
        {
            "year": int(row["base_year"]),
            "quarter": int(row["quarter"]),
            "registered": float(row["registered_population"]),
            "stay": float(row["stay_population"]),
        }
        for row in rows
    ]
    data = {"region": name, "unit": units.pop() if units else None, "points": points}
    return _envelope("ok", data=data, dataset_id=LIVING_DATASET)
