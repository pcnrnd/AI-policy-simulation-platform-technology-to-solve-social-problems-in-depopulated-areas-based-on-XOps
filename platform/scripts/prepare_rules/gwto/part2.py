"""gwto 시트 10~18."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import common
from ..common import Column
from . import blocks
from .blocks import MeltStats
from .part1 import build_year_month_sheet


def build(extracted_dir: Path) -> list[dict[str, Any]]:
    return [
        _build_10(extracted_dir),
        build_year_month_sheet(
            extracted_dir,
            slug="11_s2_2_consumption_age",
            max_col=13,
            category_name="age_group",
            category_source="연령",
            value_name="consumption_krw",
            value_source="관광소비(원)",
        ),
        build_year_month_sheet(
            extracted_dir,
            slug="12_s2_3_consumption_sex",
            max_col=13,
            category_name="sex",
            category_source="성별",
            value_name="consumption_krw",
            value_source="관광소비(원)",
        ),
        build_year_month_sheet(
            extracted_dir,
            slug="13_s2_4_consumption_industry",
            max_col=13,
            category_name="industry",
            category_source="업종(음식/숙박/레저/교통)",
            value_name="consumption_krw",
            value_source="관광소비(원)",
        ),
        build_year_month_sheet(
            extracted_dir,
            slug="14_s2_5_consumption_hour",
            max_col=13,
            category_name="hour_range",
            category_source="시간대",
            value_name="consumption_krw",
            value_source="관광소비(원)",
        ),
        build_year_month_sheet(
            extracted_dir,
            slug="15_s2_6_tourist_spending_power",
            max_col=13,
            category_name="metric",
            category_source="구분(관광객/관광소비/관광소비력)",
            value_name="value",
            value_source="값",
        ),
        _build_16(extracted_dir),
        _build_labeled_county_sheet(
            extracted_dir,
            slug="17_s3_2_city_age",
            category_name="age_group",
            category_source="구분(연령)",
            value_name="tourist_count",
        ),
        _build_labeled_county_sheet(
            extracted_dir,
            slug="18_s3_3_city_sex",
            category_name="sex",
            category_source="구분(성별: 남자/여자)",
            value_name="tourist_count",
        ),
    ]


def _build_10(extracted_dir: Path) -> dict[str, Any]:
    slug = "10_s2_1_consumption_monthly"
    rows = common.read_rows(extracted_dir / f"{slug}.csv")
    header = rows[0]

    year_cols = {c: blocks.parse_year_only(header[c]) for c in range(1, 7)}
    latest_year = max(y for y in year_cols.values() if y)

    stats = MeltStats()
    out: list[tuple[int, str, str]] = []
    for r in range(1, 13):
        row = rows[r]
        month = blocks.parse_month_only(row[0])
        if month is None:
            stats.blank_rows += 1
            continue
        for c, year in year_cols.items():
            value = row[c].strip() if c < len(row) else ""
            if not value or year is None:
                continue
            stats.value_cells += 1
            out.append((year * 100 + month, "consumption_krw", value))
        for c, metric in ((7, "yoy_ratio"), (8, "mom_ratio")):
            value = row[c].strip() if c < len(row) else ""
            if not value:
                continue
            stats.value_cells += 1
            out.append((latest_year * 100 + month, metric, value))

    stats.total_rows += 2  # 누계(13), 평균(14)

    header_out = ["base_ym", "metric", "value"]
    out_rows = [[str(b), m, v] for b, m, v in out]
    columns = [
        Column("base_ym", "구분(월)+연도 헤더 결합", "yearmonth", ""),
        Column(
            "metric",
            "연도/전년대비/전월대비",
            "string",
            "consumption_krw|yoy_ratio|mom_ratio",
        ),
        Column("value", "값", "number", "yoy_ratio/mom_ratio 는 0~1 소수(부호 포함)"),
    ]
    dropped = stats.dropped_rows_dict()
    dropped["side_table"] = 19
    result = common.SheetResult(
        dataset="gwto_kt_tourism_indicators",
        source_csv=f"extracted/gwto_kt_tourism_indicators/{slug}.csv",
        out_csv=f"gwto_kt_tourism_indicators/{slug}.csv",
        header_row_in_source=1,
        rows_in=stats.value_cells,
        header=header_out,
        rows=out_rows,
        columns=columns,
        key_columns=["base_ym", "metric"],
        dropped_rows=dropped,
        transforms=["unpivot", "wide_metric_to_long"],
        notes=[
            "'20XX년 평균' 열은 재계산 가능해 제외",
            "rows15~33 '한글변환용' 표는 본표 값/1억으로 반올림한 재계산 가능 표라 제외"
            "(예: 1월 191401939803.326/1e8≈1914.02 → 1914 일치 확인)",
        ],
    )
    return common.write_sheet(result)


def _build_16(extracted_dir: Path) -> dict[str, Any]:
    slug = "16_s3_1_city_tourists"
    rows = common.read_rows(extracted_dir / f"{slug}.csv")
    out, stats = blocks.melt_grouped_wide_blocks(rows)

    header = ["base_ym", "sigungu_name", "traveler_group", "tourist_count"]
    out_rows = [[f"{y}{m:02d}", county, cat, v] for county, y, m, cat, v in out]
    columns = [
        Column("base_ym", "슈퍼헤더(연-월)", "yearmonth", ""),
        Column("sigungu_name", "시군구명", "string", ""),
        Column("traveler_group", "서브헤더(외지인/외국인/전체)", "string", ""),
        Column("tourist_count", "값", "number", "반올림 없음"),
    ]
    result = common.SheetResult(
        dataset="gwto_kt_tourism_indicators",
        source_csv=f"extracted/gwto_kt_tourism_indicators/{slug}.csv",
        out_csv=f"gwto_kt_tourism_indicators/{slug}.csv",
        header_row_in_source=1,
        rows_in=stats.value_cells + stats.duplicate_cells,
        header=header,
        rows=out_rows,
        columns=columns,
        key_columns=["base_ym", "sigungu_name", "traveler_group"],
        dropped_rows=stats.dropped_rows_dict(),
        transforms=["unpivot", "슈퍼서브헤더_세로블록반복"],
        notes=[
            f"블록 간 중복 (entity, base_ym, group) 셀 {stats.duplicate_cells}개 — 뒤 블록"
            " 값으로 덮어씀(예: 2025년 두 블록 중 뒤쪽이 외국인 수치를 데이터랩 기준으로 정정,"
            " row105 주석 '*외국인 skt데이터' 확인)"
            if stats.duplicate_cells
            else "",
            "rows129~149 기준월 비교표·rows151 이후 순위/증감표는 헤더 형식이 달라(col0 공백"
            " 또는 비교표 라벨) 자동 제외됨 — uncertain: 순위표는 재계산 가능하다고 보아 별도"
            " 보존하지 않음",
        ],
    )
    result.notes = [n for n in result.notes if n]
    return common.write_sheet(result)


def _build_labeled_county_sheet(
    extracted_dir: Path,
    slug: str,
    category_name: str,
    category_source: str,
    value_name: str,
) -> dict[str, Any]:
    rows = common.read_rows(extracted_dir / f"{slug}.csv")
    out, stats, county_names = blocks.melt_label_above_header_blocks(rows)

    header = ["base_ym", category_name, "sigungu_name", value_name]
    out_rows = [[f"{y}{m:02d}", cat, county, v] for y, m, cat, county, v in out]
    columns = [
        Column("base_ym", "라벨 행(연-월)", "yearmonth", ""),
        Column(category_name, category_source, "string", ""),
        Column("sigungu_name", "구분(헤더 행의 시군구명)", "string", ""),
        Column(value_name, "값", "number", "반올림 없음"),
    ]
    result = common.SheetResult(
        dataset="gwto_kt_tourism_indicators",
        source_csv=f"extracted/gwto_kt_tourism_indicators/{slug}.csv",
        out_csv=f"gwto_kt_tourism_indicators/{slug}.csv",
        header_row_in_source=2,
        rows_in=stats.value_cells,
        header=header,
        rows=out_rows,
        columns=columns,
        key_columns=["base_ym", category_name, "sigungu_name"],
        dropped_rows=stats.dropped_rows_dict(),
        transforms=["unpivot", "월라벨_헤더위_블록반복"],
        notes=[f"인식된 시군구 수: {len([c for c in county_names if c])}"],
    )
    return common.write_sheet(result)
