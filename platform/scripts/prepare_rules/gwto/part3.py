"""gwto 시트 19~26."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import common
from ..common import Column
from . import blocks
from .blocks import MeltStats


def build(extracted_dir: Path) -> list[dict[str, Any]]:
    return [
        _build_19(extracted_dir),
        _build_20(extracted_dir),
        _build_21(extracted_dir),
        _build_22(extracted_dir),
        _build_23(extracted_dir),
        _build_24(extracted_dir),
        _build_25(extracted_dir),
        _build_26(extracted_dir),
    ]


def _build_19(extracted_dir: Path) -> dict[str, Any]:
    slug = "19_s3_4_city_hour"
    rows = common.read_rows(extracted_dir / f"{slug}.csv")
    out, stats = blocks.melt_simple_wide(rows, header_row_idx=0, id_col_count=1)

    header = ["hour_range", "sigungu_name", "tourist_count"]
    out_rows = [[h, county, v] for h, county, v in out]
    columns = [
        Column("hour_range", "구분(시간대)", "string", "예: 00시00분~00시59분"),
        Column("sigungu_name", "헤더 행(시군구명)", "string", ""),
        Column("tourist_count", "값", "number", "반올림 없음"),
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
        key_columns=["hour_range", "sigungu_name"],
        dropped_rows=stats.dropped_rows_dict(),
        transforms=["unpivot", "단일와이드표"],
        notes=[],
    )
    return common.write_sheet(result)


def _build_20(extracted_dir: Path) -> dict[str, Any]:
    slug = "20_s3_5_city_lodging"
    rows = common.read_rows(extracted_dir / f"{slug}.csv")
    header = rows[0]
    county_names = [c.strip() for c in header[2:20]]  # 전체(20번째) 제외, 재계산 가능
    out, stats = blocks.melt_simple_wide(
        rows, header_row_idx=0, id_col_count=2, county_names=county_names
    )

    header_out = ["base_ym", "lodging_nights", "sigungu_name", "tourist_count"]
    out_rows = [[ym, nights, county, v] for ym, nights, county, v in out]
    columns = [
        Column("base_ym", "기준년월", "yearmonth", ""),
        Column("lodging_nights", "구분(무박~7박)", "string", ""),
        Column("sigungu_name", "헤더 행(시군구명)", "string", ""),
        Column("tourist_count", "값", "number", "반올림 없음"),
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
        key_columns=["base_ym", "lodging_nights", "sigungu_name"],
        dropped_rows=stats.dropped_rows_dict(),
        transforms=["unpivot", "단일와이드표"],
        notes=["'전체' 열(18개 시군구 합계, 재계산 가능)은 제외"],
    )
    return common.write_sheet(result)


def _build_21(extracted_dir: Path) -> dict[str, Any]:
    slug = "21_s3_6_city_region"
    rows = common.read_rows(extracted_dir / f"{slug}.csv")
    out, stats = blocks.melt_simple_wide(rows, header_row_idx=1, id_col_count=1)

    header = ["residence_sido_name", "sigungu_name", "visitor_count"]
    out_rows = [[s, county, v] for s, county, v in out]
    columns = [
        Column("residence_sido_name", "구분(거주지 광역)", "string", ""),
        Column("sigungu_name", "헤더 행(시군구명)", "string", ""),
        Column("visitor_count", "값", "number", "반올림 없음"),
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
        key_columns=["residence_sido_name", "sigungu_name"],
        dropped_rows=stats.dropped_rows_dict(),
        transforms=["unpivot", "단일와이드표"],
        notes=[],
    )
    return common.write_sheet(result)


def _build_22(extracted_dir: Path) -> dict[str, Any]:
    slug = "22_s3_7_city_consumption"
    rows = common.read_rows(extracted_dir / f"{slug}.csv")
    out, stats, county_names = blocks.melt_label_above_header_blocks(rows)

    header = ["base_ym", "industry", "sigungu_name", "consumption_krw"]
    out_rows = [[f"{y}{m:02d}", cat, county, v] for y, m, cat, county, v in out]
    columns = [
        Column("base_ym", "라벨 행(연-월)/첫 블록은 '구분'", "yearmonth", ""),
        Column("industry", "구분(식음료/숙박/교통/레저)", "string", ""),
        Column("sigungu_name", "헤더 행(시군구명)", "string", ""),
        Column("consumption_krw", "값", "number", "반올림 없음"),
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
        key_columns=["base_ym", "industry", "sigungu_name"],
        dropped_rows=stats.dropped_rows_dict(),
        transforms=["unpivot", "월라벨_헤더위_블록반복"],
        notes=[
            "첫 블록(헤더 '구분', 라벨 없음, 아마 2025년 1월)은 라벨을 특정할 수 없어 제외"
            " — uncertain"
        ],
    )
    return common.write_sheet(result)


def _parse_group_period(rows: list[list[str]], header_label_row: int, start_col: int):
    label_cell = rows[header_label_row][start_col]
    return blocks.parse_year_month(label_cell)


def _melt_side_by_side_navi(
    rows: list[list[str]], groups: list[tuple[int, list[str]]]
) -> tuple[list[tuple], MeltStats]:
    """23/24: (라벨 행 0, 헤더 행 1) + 그룹별 start_col, 필드명 목록으로 세로로 쌓는다."""
    stats = MeltStats()
    stats.title_rows += 2  # 라벨 행, 헤더 행
    out: list[tuple] = []
    for start_col, fields in groups:
        ym = _parse_group_period(rows, 0, start_col)
        if ym is None:
            continue
        base_ym = ym[0] * 100 + ym[1]
        for r in range(2, len(rows)):
            row = rows[r]
            spot_name = row[start_col].strip() if start_col < len(row) else ""
            if not spot_name:
                stats.blank_rows += 1
                continue
            values = []
            for offset in range(1, len(fields)):
                c = start_col + offset
                values.append(row[c].strip() if c < len(row) else "")
            stats.value_cells += 1
            out.append((base_ym, spot_name, *values))
    return out, stats


def _build_23(extracted_dir: Path) -> dict[str, Any]:
    slug = "23_s4_1_navi_top500_prev_month"
    rows = common.read_rows(extracted_dir / f"{slug}.csv")
    fields_no_delta = ["spot_name", "rank", "region", "category_major", "category_minor"]
    fields_with_delta = fields_no_delta + ["mom_rank_change"]
    groups = [(0, fields_no_delta), (6, fields_with_delta)]
    out, stats = _melt_side_by_side_navi(rows, groups)

    header = [
        "base_ym",
        "spot_name",
        "rank",
        "region",
        "category_major",
        "category_minor",
        "mom_rank_change",
    ]
    out_rows = []
    for row in out:
        base_ym, spot_name, *rest = row
        rest = rest + [""] * (5 - len(rest))
        out_rows.append([str(base_ym), spot_name, *rest])
    columns = [
        Column("base_ym", "라벨 행(연-월) 네비데이터", "yearmonth", ""),
        Column("spot_name", "관광지명", "string", ""),
        Column("rank", "순위", "integer", ""),
        Column("region", "지역", "string", ""),
        Column("category_major", "중분류 카테고리", "string", ""),
        Column("category_minor", "소분류 카테고리", "string", ""),
        Column(
            "mom_rank_change",
            "전월대비증감",
            "string",
            "정수 문자열 또는 'New'(신규 진입)/'-'(변동 없음/미상). 최신 기간"
            " 그룹에만 존재, 그 외 빈 문자열",
        ),
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
        key_columns=["base_ym", "spot_name"],
        dropped_rows=stats.dropped_rows_dict(),
        transforms=["unpivot", "좌우병렬기간표_세로로쌓기"],
        notes=[
            "두번째 그룹의 헤더 없는 마지막 열(순위 재기재로 추정)은 의미가 불확실해 제외"
            " — uncertain",
            "원본 순위(rank)에 드문 중복이 있어(같은 base_ym 안 동일 순위 2건) 키를 rank"
            " 대신 spot_name 으로 사용 — (base_ym, spot_name) 조합도 극소수(2건) 중복 잔존,"
            " 원본 데이터 품질 이슈로 추정",
        ],
    )
    return common.write_sheet(result)


def _build_24(extracted_dir: Path) -> dict[str, Any]:
    slug = "24_s4_2_navi_top500_prev_year"
    rows = common.read_rows(extracted_dir / f"{slug}.csv")
    fields_no_delta = ["spot_name", "rank", "region", "category_major", "category_minor"]
    fields_with_delta = fields_no_delta + ["yoy_rank_change"]
    groups = [(0, fields_no_delta), (6, fields_with_delta)]
    out, stats = _melt_side_by_side_navi(rows, groups)

    header = [
        "base_ym",
        "spot_name",
        "rank",
        "region",
        "category_major",
        "category_minor",
        "yoy_rank_change",
    ]
    out_rows = []
    for row in out:
        base_ym, spot_name, *rest = row
        rest = rest + [""] * (5 - len(rest))
        out_rows.append([str(base_ym), spot_name, *rest])
    columns = [
        Column("base_ym", "라벨 행(연-월) 네비데이터", "yearmonth", ""),
        Column("spot_name", "관광지명", "string", ""),
        Column("rank", "순위", "integer", ""),
        Column("region", "지역", "string", ""),
        Column("category_major", "중분류 카테고리", "string", ""),
        Column("category_minor", "소분류 카테고리", "string", ""),
        Column(
            "yoy_rank_change",
            "전월대비증감(전년 그룹과 비교)",
            "string",
            "정수 문자열 또는 'New'(신규 진입)/'-'(변동 없음/미상). 최신 기간"
            " 그룹에만 존재, 그 외 빈 문자열",
        ),
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
        key_columns=["base_ym", "spot_name"],
        dropped_rows=stats.dropped_rows_dict(),
        transforms=["unpivot", "좌우병렬기간표_세로로쌓기"],
        notes=[
            "원본 순위(rank)에 드문 중복이 있어 키를 rank 대신 spot_name 으로 사용 —"
            " (base_ym, spot_name) 조합도 극소수(3건) 중복 잔존, 원본 데이터 품질 이슈로 추정"
        ],
    )
    return common.write_sheet(result)


def _build_25(extracted_dir: Path) -> dict[str, Any]:
    slug = "25_s4_3_navi_summary_500"
    rows = common.read_rows(extracted_dir / f"{slug}.csv")
    fields_latest = ["spot_name", "rank", "region", "mom_rank_up", "yoy_rank_up"]
    fields_plain = ["spot_name", "rank", "region"]
    groups = [(0, fields_latest), (5, fields_plain), (9, fields_plain)]

    stats = MeltStats()
    stats.title_rows += 2
    out: list[tuple] = []
    for start_col, fields in groups:
        ym = _parse_group_period(rows, 0, start_col)
        if ym is None:
            continue
        base_ym = ym[0] * 100 + ym[1]
        for r in range(2, len(rows)):
            row = rows[r]
            spot_name = row[start_col].strip() if start_col < len(row) else ""
            if not spot_name:
                stats.blank_rows += 1
                continue
            values = [
                row[start_col + off].strip() if start_col + off < len(row) else ""
                for off in range(1, len(fields))
            ]
            stats.value_cells += 1
            out.append((base_ym, spot_name, *values))

    header = ["base_ym", "spot_name", "rank", "region", "mom_rank_up", "yoy_rank_up"]
    out_rows = []
    for row in out:
        base_ym, spot_name, *rest = row
        rest = rest + [""] * (4 - len(rest))
        out_rows.append([str(base_ym), spot_name, *rest])
    columns = [
        Column("base_ym", "라벨 행(연-월) 내비데이터 Top 30", "yearmonth", ""),
        Column("spot_name", "관광지명", "string", ""),
        Column("rank", "순위", "integer", ""),
        Column("region", "지역", "string", ""),
        Column(
            "mom_rank_up",
            "전월대비 상승순위",
            "string",
            "정수 문자열 또는 'New'(신규 진입). 최신 기간 그룹에만 존재, 그 외 빈 문자열",
        ),
        Column(
            "yoy_rank_up",
            "전년대비 상승순위",
            "string",
            "정수 문자열 또는 'New'(신규 진입). 최신 기간 그룹에만 존재, 그 외 빈 문자열",
        ),
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
        key_columns=["base_ym", "rank"],
        dropped_rows=stats.dropped_rows_dict(),
        transforms=["unpivot", "좌우병렬기간표_세로로쌓기"],
        notes=["Top 30 만 존재하는 표라 rank<=30 만 나온다(원본이 이미 상위 30건만 수록)"],
    )
    return common.write_sheet(result)


def _build_26(extracted_dir: Path) -> dict[str, Any]:
    slug = "26_daily_trend"
    rows = common.read_rows(extracted_dir / f"{slug}.csv")
    header_row = rows[0]
    width = len(header_row)

    # 그룹 시작열을 라벨(연도/월)이 등장하는 위치로 찾는다(그룹 폭은 5 또는 6일 수 있음).
    group_starts: list[int] = []
    c = 0
    while c < width:
        cell = header_row[c].strip()
        if cell and (blocks.parse_year_month(cell) or blocks.parse_year_only(cell)):
            group_starts.append(c)
        c += 1

    stats = MeltStats()
    stats.title_rows += 1  # 헤더(라벨) 행
    out: list[tuple[int, str, str, str, str]] = []  # (base_ym, day, visits, day_type, avg_visits)
    seen: dict[tuple[int, str], int] = {}

    for gi, start in enumerate(group_starts):
        end = group_starts[gi + 1] if gi + 1 < len(group_starts) else width
        label_cell = header_row[start].strip()
        ym = blocks.parse_year_month(label_cell)
        if ym is None:
            # '2025년' + '4월' 처럼 연/월이 분리된 헤더 — 다음 비어있지 않은 셀에서 월을 찾는다
            month_cell = ""
            for cc in range(start + 1, end):
                if header_row[cc].strip():
                    month_cell = header_row[cc].strip()
                    break
            year = blocks.parse_year_only(label_cell)
            month = blocks.parse_month_only(month_cell)
            if year is None or month is None:
                stats.title_rows += 1
                continue
            ym = (year, month)
        base_ym = ym[0] * 100 + ym[1]

        for r in range(1, len(rows)):
            row = rows[r]
            day_label = row[start].strip() if start < len(row) else ""
            if not day_label:
                continue
            visits = row[start + 1].strip() if start + 1 < len(row) else ""
            day_type = row[start + 2].strip() if start + 2 < len(row) else ""
            avg_visits = row[start + 3].strip() if start + 3 < len(row) else ""
            if not visits:
                continue
            key = (base_ym, day_label)
            entry = (base_ym, day_label, visits, day_type, avg_visits)
            if key in seen:
                stats.duplicate_cells += 1
                out[seen[key]] = entry
                continue
            seen[key] = len(out)
            stats.value_cells += 1
            out.append(entry)

    header = ["base_ym", "day_label", "visit_count", "day_type", "avg_visit_count"]
    out_rows = [[str(b), d, v, dt, av] for b, d, v, dt, av in out]
    columns = [
        Column("base_ym", "그룹 헤더(연-월)", "yearmonth", ""),
        Column("day_label", "일(예: 1일, 01일)", "string", "원본 표기 그대로(0 유무 혼재)"),
        Column("visit_count", "값(일별 방문객수)", "number", "반올림 없음"),
        Column("day_type", "휴일/평일", "string", "빈 문자열이면 표기 없음"),
        Column(
            "avg_visit_count",
            "요일별 평균값으로 추정",
            "string",
            "uncertain: 원본에 열 이름 없음. 대부분 숫자지만 일부 그룹은 이 열이"
            " day_type(휴일/평일) 또는 '21일~25일' 같은 구간 라벨을 담고 있어"
            " number 로 고정할 수 없음(원본 셀 그대로 보존)",
        ),
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
        key_columns=["base_ym", "day_label"],
        dropped_rows=stats.dropped_rows_dict(),
        transforms=["unpivot", "좌우병렬기간표_세로로쌓기"],
        notes=[
            n
            for n in [
                "열 이름이 원본에 없어 4번째 값을 avg_visit_count(요일 평균 추정)로 명명"
                " — uncertain",
                "day_label 은 '1일'/'01일' 표기가 그룹마다 혼재해 원본 그대로 보존(정규화 안 함)",
                f"같은 (base_ym, day_label) 이 여러 병렬 그룹에 중복 등장 {stats.duplicate_cells}건"
                " — 뒤 그룹 값으로 덮어씀"
                if stats.duplicate_cells
                else "",
            ]
            if n
        ],
    )
    return common.write_sheet(result)
