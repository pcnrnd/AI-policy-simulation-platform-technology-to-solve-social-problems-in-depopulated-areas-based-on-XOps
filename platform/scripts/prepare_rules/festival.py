"""nowon_kt_festival: 시트별 헤더 행 오프셋(5~8행)이 다르다. 상수표로 고정한다."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from . import common
from .common import Column, SheetResult

DATASET = "nowon_kt_festival"

_DATE_RE = re.compile(r"^\d{8}$")


def _to_iso_date(value: str) -> str:
    v = value.strip()
    if not _DATE_RE.fullmatch(v):
        return v
    return f"{v[0:4]}-{v[4:6]}-{v[6:8]}"


# slug -> (header_row_in_source(1-indexed), key_columns, out_columns)
# out_columns: (out_name, source_name, type, description)
_SHEETS: dict[str, tuple[int, list[str], list[tuple[str, str, str, str]]]] = {
    "02_daily_visitors": (
        5,
        ["date"],
        [
            ("date", "일별", "date", "YYYY-MM-DD"),
            ("spot_name", "관광지명", "string", ""),
            ("local_visitors", "현지인 방문객", "number", "반올림 없음"),
            ("nonlocal_visitors", "외지인 방문객", "number", "반올림 없음"),
            ("foreign_visitors", "외국인 방문객", "number", "반올림 없음"),
            ("total_visitors", "현지인+외지인+외국인", "number", "반올림 없음"),
            ("nonlocal_foreign_visitors", "외지인+외국인", "number", "반올림 없음"),
        ],
    ),
    "03_visitors_by_sex_age": (
        6,
        ["date", "sex_code", "age_code"],
        [
            ("date", "일별", "date", "YYYY-MM-DD"),
            ("spot_name", "관광지명", "string", ""),
            ("sex_code", "성별 코드", "string", "F/M"),
            ("age_code", "연령 코드", "string", "예: A2029"),
            ("local_visitors", "현지인 방문객", "number", "반올림 없음"),
            ("nonlocal_visitors", "외지인 방문객", "number", "반올림 없음"),
            ("local_nonlocal_visitors", "현지인+외지인", "number", "반올림 없음"),
        ],
    ),
    "04_hourly_inflow": (
        8,
        ["date", "hour"],
        [
            ("date", "일별", "date", "YYYY-MM-DD"),
            ("spot_name", "관광지명", "string", ""),
            ("hour", "시간대별", "string", "'00'~'23' 문자열 유지"),
            ("local_visitors", "현지인 방문객", "number", "반올림 없음"),
            ("nonlocal_visitors", "외지인 방문객", "number", "반올림 없음"),
            ("foreign_visitors", "외국인 방문객", "number", "반올림 없음"),
            ("total_visitors", "현지인+외지인+외국인", "number", "반올림 없음"),
            ("nonlocal_foreign_visitors", "외지인+외국인", "number", "반올림 없음"),
        ],
    ),
    "05_hourly_presence": (
        8,
        ["date", "hour"],
        [
            ("date", "일별", "date", "YYYY-MM-DD"),
            ("spot_name", "관광지명", "string", ""),
            ("hour", "시간대별", "string", "'00'~'23' 문자열 유지"),
            ("local_visitors", "현지인 방문객", "number", "반올림 없음"),
            ("nonlocal_visitors", "외지인 방문객", "number", "반올림 없음"),
            ("foreign_visitors", "외국인 방문객", "number", "반올림 없음"),
            ("total_visitors", "현지인+외지인+외국인", "number", "반올림 없음"),
            ("nonlocal_foreign_visitors", "외지인+외국인", "number", "반올림 없음"),
        ],
    ),
    "06_visitors_by_nationality": (
        7,
        ["date", "nationality_code"],
        [
            ("date", "일별", "date", "YYYY-MM-DD"),
            ("spot_name", "관광지명", "string", ""),
            ("nationality_name", "국적명", "string", ""),
            ("nationality_code", "국적코드", "string", ""),
            ("foreign_visitors", "외국인 방문객", "number", "반올림 없음"),
        ],
    ),
    "07_hourly_by_nationality": (
        8,
        ["date", "hour", "nationality_code"],
        [
            ("date", "일별", "date", "YYYY-MM-DD"),
            ("spot_name", "관광지명", "string", ""),
            ("hour", "시간대별", "string", "'00'~'23' 문자열 유지"),
            ("nationality_name", "국적명", "string", ""),
            ("nationality_code", "국적코드", "string", ""),
            ("foreign_visitors", "외국인 방문객", "number", "반올림 없음"),
        ],
    ),
    "08_residence_ratio": (
        7,
        ["date", "residence_sigungu_code"],
        [
            ("date", "일별", "date", "YYYY-MM-DD"),
            ("spot_name", "관광지명", "string", ""),
            ("residence_sido_name", "거주지역_시도명", "string", ""),
            ("residence_sido_code", "거주지역_시도코드", "string", ""),
            ("residence_sigungu_name", "거주지역_시군구명", "string", ""),
            ("residence_sigungu_code", "거주지역_시군구코드", "string", ""),
            ("nonlocal_visitors_ratio", "외지인 방문객(%)", "number", "0~1 소수 원값"),
        ],
    ),
    "09_move_after_24h_ratio": (
        7,
        ["date", "move_sigungu_code"],
        [
            ("date", "일별", "date", "YYYY-MM-DD"),
            ("spot_name", "관광지명", "string", ""),
            ("move_sido_name", "이동지역_시도명", "string", ""),
            ("move_sido_code", "이동지역_시도코드", "string", ""),
            ("move_sigungu_name", "이동지역_시군구명", "string", ""),
            ("move_sigungu_code", "이동지역_시군구코드", "string", ""),
            ("nonlocal_visitors_ratio", "외지인 방문객(%)", "number", "0~1 소수 원값"),
            ("foreign_visitors_ratio", "외국인 방문객(%)", "number", "0~1 소수 원값"),
        ],
    ),
    "10_outflow_after_2h_ratio": (
        7,
        ["date", "outflow_dong_code"],
        [
            ("date", "일별", "date", "YYYY-MM-DD"),
            ("spot_name", "관광지명", "string", ""),
            ("outflow_sido_name", "유출지역_시도명", "string", ""),
            ("outflow_sido_code", "유출지역_시도코드", "string", ""),
            ("outflow_sigungu_name", "유출지역_시군구명", "string", ""),
            ("outflow_sigungu_code", "유출지역_시군구코드", "string", ""),
            ("outflow_dong_name", "유출지역_행정동명", "string", ""),
            ("outflow_dong_code", "유출지역_행정동코드", "string", ""),
            ("local_visitors_ratio", "현지인 방문객(%)", "number", "0~1 소수 원값"),
            ("nonlocal_visitors_ratio", "외지인 방문객(%)", "number", "0~1 소수 원값"),
            ("foreign_visitors_ratio", "외국인 방문객(%)", "number", "0~1 소수 원값"),
        ],
    ),
    "11_avg_stay_hours": (
        5,
        ["date"],
        [
            ("date", "일별", "date", "YYYY-MM-DD"),
            ("spot_name", "관광지명", "string", ""),
            ("local_stay_hours", "현지인 체류시간", "number", ""),
            ("nonlocal_stay_hours", "외지인 체류시간", "number", ""),
            ("foreign_stay_hours", "외국인 체류시간", "number", ""),
        ],
    ),
}

# 02_daily_visitors 헤더 우측(J~M열, 0-indexed 9~12)의 축제 전/중/후 요약 부속표.
# 본표에서 제외하고(재계산 가능) 존재만 기록한다.
_SIDE_TABLE_COLS = 7  # 본표는 0~6, 7열부터 제외


def build(extracted_dir: Path) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []

    meta_src = extracted_dir / "01_overview.csv"
    meta_rows = common.read_rows(meta_src)
    manifest.append(
        common.write_meta(
            DATASET,
            f"extracted/{DATASET}/01_overview.csv",
            f"{DATASET}/01_overview.csv",
            len(meta_rows),
        )
    )

    for slug, (header_row, key_cols, out_cols) in _SHEETS.items():
        manifest.append(_build_sheet(extracted_dir, slug, header_row, key_cols, out_cols))
    return manifest


def _build_sheet(
    extracted_dir: Path,
    slug: str,
    header_row: int,
    key_columns: list[str],
    out_columns: list[tuple[str, str, str, str]],
) -> dict[str, Any]:
    src = extracted_dir / f"{slug}.csv"
    rows = common.read_rows(src)
    header_idx = header_row - 1
    data_rows = rows[header_idx + 1 :]
    rows_in = len(data_rows)

    ncols = len(out_columns)
    date_idx = next((i for i, c in enumerate(out_columns) if c[2] == "date"), None)

    notes = []
    is_daily_visitors = slug == "02_daily_visitors"
    if is_daily_visitors:
        notes.append(
            "J~M열(0-idx 9~12) 축제 전/중/후 요약 부속표는 본표에서 제외"
            "(현지인+외지인+외국인 컬럼에서 재계산 가능)"
        )

    out_rows = []
    for row in data_rows:
        trimmed = [c.strip() for c in row[:ncols]]
        while len(trimmed) < ncols:
            trimmed.append("")
        if date_idx is not None:
            trimmed[date_idx] = _to_iso_date(trimmed[date_idx])
        out_rows.append(trimmed)

    header = [c[0] for c in out_columns]
    columns = [Column(c[0], c[1], c[2], c[3]) for c in out_columns]

    result = SheetResult(
        dataset=DATASET,
        source_csv=f"extracted/{DATASET}/{slug}.csv",
        out_csv=f"{DATASET}/{slug}.csv",
        header_row_in_source=header_row,
        rows_in=rows_in,
        header=header,
        rows=out_rows,
        columns=columns,
        key_columns=key_columns,
        dropped_rows={"title": 0, "total": 0, "blank": 0, "side_table": 0},
        transforms=["column_rename", "type_fix", "date_format"]
        + (["side_table_excluded"] if is_daily_visitors else []),
        notes=notes,
    )
    return common.write_sheet(result)
