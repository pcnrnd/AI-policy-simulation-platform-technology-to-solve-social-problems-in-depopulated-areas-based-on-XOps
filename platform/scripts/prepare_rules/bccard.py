"""namwon_bccard_consumption: 설명 시트 1 + 데이터 시트 1, 값 변경 없음."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import common
from .common import Column, SheetResult

DATASET = "namwon_bccard_consumption"


def build(extracted_dir: Path) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []

    meta_src = extracted_dir / "01_data_layout.csv"
    meta_rows = common.read_rows(meta_src)
    manifest.append(
        common.write_meta(
            DATASET,
            f"extracted/{DATASET}/01_data_layout.csv",
            f"{DATASET}/01_data_layout.csv",
            len(meta_rows),
        )
    )

    manifest.append(_build_dong_industry_sales(extracted_dir))
    return manifest


def _build_dong_industry_sales(extracted_dir: Path) -> dict[str, Any]:
    src = extracted_dir / "02_dong_industry_sales.csv"
    rows = common.read_rows(src)
    rows_in = len(rows) - 1
    data_rows = rows[1:]

    header = [
        "base_ym",
        "customer_type",
        "dong_name",
        "industry_6",
        "sex",
        "age_group",
        "weekday",
        "sales_est_krw",
    ]
    out_rows = [[c.strip() for c in row] for row in data_rows]

    columns = [
        Column("base_ym", "기준년월", "yearmonth", "기준년월(YYYYMM)"),
        Column("customer_type", "고객구분", "string", "외지인/현지인"),
        Column("dong_name", "행정동명", "string", "행정동명(총 23개)"),
        Column("industry_6", "6대 업종명", "string", "식음료/쇼핑소매/문화레져/숙박/유흥/교통"),
        Column("sex", "성별", "string", "남/여"),
        Column("age_group", "연령", "string", "20대~60대이상"),
        Column("weekday", "요일", "string", "월~일"),
        Column("sales_est_krw", "매출추정액(원)", "integer", "매출추정액(원), 반올림 없음"),
    ]

    result = SheetResult(
        dataset=DATASET,
        source_csv=f"extracted/{DATASET}/02_dong_industry_sales.csv",
        out_csv=f"{DATASET}/02_dong_industry_sales.csv",
        header_row_in_source=1,
        rows_in=rows_in,
        header=header,
        rows=out_rows,
        columns=columns,
        key_columns=[
            "base_ym",
            "customer_type",
            "dong_name",
            "industry_6",
            "sex",
            "age_group",
            "weekday",
        ],
        dropped_rows={"title": 0, "total": 0, "blank": 0, "side_table": 0},
        transforms=["column_rename", "type_fix"],
        notes=["값 변경 없음: 원본 셀 문자열 그대로(공백 제거만 적용)"],
    )
    return common.write_sheet(result)
