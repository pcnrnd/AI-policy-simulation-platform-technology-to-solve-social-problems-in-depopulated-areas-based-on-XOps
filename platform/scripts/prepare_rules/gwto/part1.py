"""gwto 시트 01~09."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import common
from ..common import Column
from . import blocks
from .blocks import MeltStats


def build(extracted_dir: Path) -> list[dict[str, Any]]:
    return [
        _build_01(extracted_dir),
        _build_02(extracted_dir),
        build_year_month_sheet(
            extracted_dir,
            slug="03_s1_3_tourists_age",
            max_col=13,
            category_name="age_group",
            category_source="연령",
            value_name="tourist_count",
            value_source="관광객수",
        ),
        build_year_month_sheet(
            extracted_dir,
            slug="04_s1_4_tourists_sex",
            max_col=13,
            category_name="sex",
            category_source="성별",
            value_name="tourist_count",
            value_source="관광객수",
            extra_notes=["2023년 데이터가 두 블록에 중복 등장 — 값이 동일해 뒤 블록으로 덮어써도 무해"],
        ),
        build_year_month_sheet(
            extracted_dir,
            slug="05_s1_5_foreigners",
            max_col=13,
            category_name="nationality",
            category_source="국적/구분",
            value_name="foreigner_index",
            value_source="지수(원본 단위 미상)",
            extra_notes=[
                "값 단위(지수/명 여부) 원본에 명시 없음 — uncertain: 원본 셀 값 그대로 보존"
            ],
        ),
        _build_06(extracted_dir),
        build_year_month_sheet(
            extracted_dir,
            slug="07_s1_6_lodging",
            max_col=13,
            category_name="lodging_nights",
            category_source="구분(무박~7박)",
            value_name="tourist_count",
            value_source="관광객수",
        ),
        build_year_month_sheet(
            extracted_dir,
            slug="08_s1_7_residence_region",
            max_col=26,  # 24개월 폭 블록까지 포함, col26 부터의 부속 비교 패널은 제외
            category_name="residence_sido_name",
            category_source="구분(거주지 광역)",
            value_name="visitor_count",
            value_source="방문객수",
        ),
        _build_09(extracted_dir),
    ]


def _build_01(extracted_dir: Path) -> dict[str, Any]:
    slug = "01_s1_1_nonlocal_monthly"
    rows = common.read_rows(extracted_dir / f"{slug}.csv")
    max_col = 19  # cols 0~18 만 본표, 19~20 은 "(KT) 강원도 광역 데이터인지" 비고열

    group_row = rows[0]
    year_row = rows[1]
    groups: dict[int, str] = {}
    last = ""
    for c in range(1, max_col):
        v = group_row[c].strip() if c < len(group_row) else ""
        if v:
            last = v
        groups[c] = last
    years: dict[int, int] = {}
    for c in range(1, max_col):
        y = blocks.parse_year_only(year_row[c]) if c < len(year_row) else None
        if y:
            years[c] = y

    stats = MeltStats()
    out: list[tuple[int, int, str, str]] = []  # (month, year, group, value)
    for r in range(2, 14):  # rows 2~13 = 1월~12월
        row = rows[r]
        month = blocks.parse_month_only(row[0])
        if month is None:
            stats.blank_rows += 1
            continue
        for c in range(1, max_col):
            if c not in years:
                continue
            value = row[c].strip() if c < len(row) else ""
            if not value:
                continue
            stats.value_cells += 1
            out.append((month, years[c], groups[c], value))
    stats.total_rows += 2  # 누계, 평균 행(14,15)
    stats.blank_rows += 3  # 16~18 빈 행
    side_table_rows = 5  # 19~23 부속 비교표(기준월 대비) — 재계산 가능, 제외

    header = ["base_ym", "traveler_group", "visitor_count"]
    out_rows = [[f"{y}{m:02d}", g, v] for m, y, g, v in out]
    columns = [
        Column("base_ym", "구분(연도)+구분(월)", "yearmonth", "연-월 헤더 결합"),
        Column("traveler_group", "구분(외지인/외국인/전체 관광객)", "string", ""),
        Column("visitor_count", "값", "integer", "반올림 없음"),
    ]
    dropped = stats.dropped_rows_dict()
    dropped["side_table"] = side_table_rows
    result = common.SheetResult(
        dataset="gwto_kt_tourism_indicators",
        source_csv=f"extracted/gwto_kt_tourism_indicators/{slug}.csv",
        out_csv=f"gwto_kt_tourism_indicators/{slug}.csv",
        header_row_in_source=1,
        rows_in=stats.value_cells,
        header=header,
        rows=out_rows,
        columns=columns,
        key_columns=["base_ym", "traveler_group"],
        dropped_rows=dropped,
        transforms=["unpivot", "2단_헤더(그룹+연도)_forward_fill"],
        notes=[
            "rows19~23 부속 비교표(기준월/전월/전년 대비)는 본표에서 재계산 가능해 제외"
        ],
    )
    return common.write_sheet(result)


def _build_02(extracted_dir: Path) -> dict[str, Any]:
    slug = "02_s1_2_total_tourists_monthly"
    rows = common.read_rows(extracted_dir / f"{slug}.csv")
    header = rows[0]

    year_cols = {c: blocks.parse_year_only(header[c]) for c in range(1, 7)}
    latest_year = max(y for y in year_cols.values() if y)

    stats = MeltStats()
    out: list[tuple[int, str, str]] = []  # (base_ym, metric, value)
    for r in range(1, 13):  # rows1~12 = 1~12월
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
            out.append((year * 100 + month, "total_tourists", value))
        for c, metric in ((7, "yoy_ratio"), (8, "mom_ratio")):
            value = row[c].strip() if c < len(row) else ""
            if not value:
                continue
            stats.value_cells += 1
            out.append((latest_year * 100 + month, metric, value))

    stats.total_rows += 2  # 누계(13행), 평균(14행)
    trailing = len(rows) - 15  # 15행부터 끝까지 빈 행/비고
    stats.blank_rows += trailing

    header_out = ["base_ym", "metric", "value"]
    out_rows = [[str(b), m, v] for b, m, v in out]
    columns = [
        Column("base_ym", "구분(월)+연도 헤더 결합", "yearmonth", ""),
        Column(
            "metric",
            "연도/전년대비/전월대비",
            "string",
            "total_tourists|yoy_ratio|mom_ratio",
        ),
        Column("value", "값", "number", "yoy_ratio/mom_ratio 는 0~1 소수(부호 포함)"),
    ]
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
        dropped_rows=stats.dropped_rows_dict(),
        transforms=["unpivot", "wide_metric_to_long"],
        notes=[
            "'20XX년 평균' 열은 같은 연도 12개월 평균과 동일(재계산 가능)해 제외",
            "yoy_ratio/mom_ratio 는 원본에 연도 라벨이 없어 최신 연도(2025)로 간주 — uncertain",
        ],
    )
    return common.write_sheet(result)


def build_year_month_sheet(
    extracted_dir: Path,
    slug: str,
    max_col: int | None,
    category_name: str,
    category_source: str,
    value_name: str,
    value_source: str,
    extra_notes: list[str] | None = None,
) -> dict[str, Any]:
    rows = common.read_rows(extracted_dir / f"{slug}.csv")
    width = max_col or max(len(r) for r in rows)
    out, stats = blocks.melt_year_month_header_rows(rows, width)

    header = ["base_ym", category_name, value_name]
    out_rows = [[f"{y}{m:02d}", cat, v] for cat, y, m, v in out]
    columns = [
        Column("base_ym", "구분(연-월 헤더)", "yearmonth", ""),
        Column(category_name, category_source, "string", ""),
        Column(value_name, value_source, "number", "반올림 없음"),
    ]
    notes = list(extra_notes or [])
    if stats.duplicate_cells:
        notes.append(
            f"블록 간 중복 (category, base_ym) 셀 {stats.duplicate_cells}개 — 뒤 블록 값으로 덮어씀"
        )

    result = common.SheetResult(
        dataset="gwto_kt_tourism_indicators",
        source_csv=f"extracted/gwto_kt_tourism_indicators/{slug}.csv",
        out_csv=f"gwto_kt_tourism_indicators/{slug}.csv",
        header_row_in_source=1,
        rows_in=stats.value_cells + stats.duplicate_cells,
        header=header,
        rows=out_rows,
        columns=columns,
        key_columns=["base_ym", category_name],
        dropped_rows=stats.dropped_rows_dict(),
        transforms=["unpivot", "연도블록_세로반복"],
        notes=notes,
    )
    return common.write_sheet(result)


def _build_06(extracted_dir: Path) -> dict[str, Any]:
    slug = "06_datalab_foreigners"
    rows = common.read_rows(extracted_dir / f"{slug}.csv")
    out, stats = blocks.melt_name_value_pairs(
        rows, label_row_idx=0, data_start_idx=1, pair_start_col=2, period=3
    )

    header = ["base_ym", "region_name", "foreign_visitor_count"]
    out_rows = [[f"{y}{m:02d}", n, v] for y, m, n, v in out]
    columns = [
        Column("base_ym", "라벨 행 연-월", "yearmonth", ""),
        Column("region_name", "(이름,값) 쌍의 이름", "string", ""),
        Column("foreign_visitor_count", "(이름,값) 쌍의 값", "number", "반올림 없음"),
    ]
    result = common.SheetResult(
        dataset="gwto_kt_tourism_indicators",
        source_csv=f"extracted/gwto_kt_tourism_indicators/{slug}.csv",
        out_csv=f"gwto_kt_tourism_indicators/{slug}.csv",
        header_row_in_source=1,
        rows_in=stats.value_cells,
        header=header,
        rows=out_rows,
        columns=columns,
        key_columns=["base_ym", "region_name"],
        dropped_rows=stats.dropped_rows_dict(),
        transforms=["unpivot", "이름값쌍_가로반복"],
        notes=[
            n
            for n in [
                "원 시트명 '데이터랩 외국인'을 따라 foreign_visitor_count 로 명명",
                f"라벨 열이 중복 등장(24년 12월 두 번)해 {stats.duplicate_cells}개 셀을"
                " 뒤 열 값으로 덮어씀(두 값 모두 동일함을 확인)"
                if stats.duplicate_cells
                else "",
            ]
            if n
        ],
    )
    return common.write_sheet(result)


def _build_09(extracted_dir: Path) -> dict[str, Any]:
    slug = "09_s1_8_residence_city"
    rows = common.read_rows(extracted_dir / f"{slug}.csv")
    out, stats = blocks.melt_ranked_pairs(rows, id_col=0)

    header = ["base_ym", "rank", "residence_name", "visitor_count"]
    out_rows = [[f"{y}{m:02d}", rank, name, v] for rank, y, m, name, v in out]
    columns = [
        Column("base_ym", "라벨 행 연-월", "yearmonth", ""),
        Column("rank", "구분(순번)", "integer", "월별 방문객수 순위"),
        Column("residence_name", "거주지 기초", "string", "시도+시군구 원문 그대로"),
        Column("visitor_count", "방문객수", "number", "반올림 없음"),
    ]
    result = common.SheetResult(
        dataset="gwto_kt_tourism_indicators",
        source_csv=f"extracted/gwto_kt_tourism_indicators/{slug}.csv",
        out_csv=f"gwto_kt_tourism_indicators/{slug}.csv",
        header_row_in_source=3,
        rows_in=stats.value_cells,
        header=header,
        rows=out_rows,
        columns=columns,
        key_columns=["base_ym", "rank"],
        dropped_rows=stats.dropped_rows_dict(),
        transforms=["unpivot", "이름값쌍_가로반복"],
        notes=(
            [
                f"블록 간 중복 (base_ym, rank) 셀 {stats.duplicate_cells}개 — 뒤 블록"
                " 값으로 덮어씀"
            ]
            if stats.duplicate_cells
            else []
        ),
    )
    return common.write_sheet(result)
