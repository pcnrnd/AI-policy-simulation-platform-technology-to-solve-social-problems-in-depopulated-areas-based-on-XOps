"""namwon_kt_visitors: 두 시트를 같은 이름·같은 형으로 정렬(조인 아님, 규격 통일)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import common
from .common import Column, SheetResult

DATASET = "namwon_kt_visitors"

_COMMON_COLUMNS = [
    Column("base_ym", "월별", "yearmonth", "기준년월(YYYYMM)"),
    Column("sido_name", "시도명", "string", ""),
    Column("sido_code", "시도코드", "string", "선행 0 보존을 위해 문자열"),
    Column("sigungu_name", "시군구명", "string", ""),
    Column("sigungu_code", "시군구코드", "string", "선행 0 보존을 위해 문자열"),
    Column("dong_name", "행정동명", "string", ""),
    Column("dong_code", "행정동코드", "string", "선행 0 보존을 위해 문자열"),
]


def build(extracted_dir: Path) -> list[dict[str, Any]]:
    return [
        _build_monthly_dong_visitors(extracted_dir),
        _build_visitors_by_sex_age(extracted_dir),
    ]


def _build_monthly_dong_visitors(extracted_dir: Path) -> dict[str, Any]:
    src = extracted_dir / "01_monthly_dong_visitors.csv"
    rows = common.read_rows(src)
    rows_in = len(rows) - 1
    data_rows = [[c.strip() for c in row] for row in rows[1:]]

    header = [c.name for c in _COMMON_COLUMNS] + [
        "local_visitors",
        "nonlocal_visitors",
        "foreign_visitors",
    ]
    columns = list(_COMMON_COLUMNS) + [
        Column("local_visitors", "현지인 방문객", "number", "반올림 없음"),
        Column("nonlocal_visitors", "외지인 방문객", "number", "반올림 없음"),
        Column("foreign_visitors", "외국인 방문객", "number", "반올림 없음"),
    ]

    result = SheetResult(
        dataset=DATASET,
        source_csv=f"extracted/{DATASET}/01_monthly_dong_visitors.csv",
        out_csv=f"{DATASET}/01_monthly_dong_visitors.csv",
        header_row_in_source=1,
        rows_in=rows_in,
        header=header,
        rows=data_rows,
        columns=columns,
        key_columns=["base_ym", "dong_code"],
        dropped_rows={"title": 0, "total": 0, "blank": 0, "side_table": 0},
        transforms=["column_rename", "type_fix"],
        notes=["02_visitors_by_sex_age.csv 와 공통 컬럼명·형 통일(파일 간 조인 아님)"],
    )
    return common.write_sheet(result)


def _build_visitors_by_sex_age(extracted_dir: Path) -> dict[str, Any]:
    src = extracted_dir / "02_visitors_by_sex_age.csv"
    rows = common.read_rows(src)
    rows_in = len(rows) - 1
    data_rows = [[c.strip() for c in row] for row in rows[1:]]

    header = [c.name for c in _COMMON_COLUMNS] + [
        "sex_code",
        "age_code",
        "local_visitors",
        "nonlocal_visitors",
    ]
    columns = list(_COMMON_COLUMNS) + [
        Column("sex_code", "성별코드", "string", "F/M"),
        Column("age_code", "연령코드", "string", "예: A2029"),
        Column("local_visitors", "현지인 방문객", "number", "반올림 없음"),
        Column("nonlocal_visitors", "외지인 방문객", "number", "반올림 없음"),
    ]

    result = SheetResult(
        dataset=DATASET,
        source_csv=f"extracted/{DATASET}/02_visitors_by_sex_age.csv",
        out_csv=f"{DATASET}/02_visitors_by_sex_age.csv",
        header_row_in_source=1,
        rows_in=rows_in,
        header=header,
        rows=data_rows,
        columns=columns,
        key_columns=["base_ym", "dong_code", "sex_code", "age_code"],
        dropped_rows={"title": 0, "total": 0, "blank": 0, "side_table": 0},
        transforms=["column_rename", "type_fix"],
        notes=[
            "01_monthly_dong_visitors.csv 와 공통 컬럼명·형 통일(파일 간 조인 아님)",
            "격자 결손 2행: 기대 조합(58개월×23행정동×2성별×8연령=21344) 대비 실제 21342행"
            " — 채우지 않음",
        ],
    )
    return common.write_sheet(result)
