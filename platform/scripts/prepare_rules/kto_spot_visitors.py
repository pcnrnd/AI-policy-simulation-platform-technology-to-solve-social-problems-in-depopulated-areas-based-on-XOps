"""kto_spot_visitors: 주요관광지점 입장객 통계표 엑셀(.xls) → 관광지점·월별 1행.

입력: 한국문화관광연구원 관광지식정보시스템 통계표(https://know.tour.go.kr/stat/visitStatDis/table.do)를
'월별'·시도·군구·관광지명으로 조회하고 '내려받기(셀병합 해제)'로 받은 파일을 `platform/data/real/kto/`에
둔다(파일 여러 개 가능). 필요한 지점만 하나씩 조회해 받는다 — 대량 내려받기는 하지 않는다.
원본·산출물은 저장소에 넣지 않는다(공공누리 표시가 없는 자료라 이용 협의 대상).

표 모양: 1행 머리(시도·군구·관광지·내/외국인·총계·연도), 2행 머리(연도별 인원계·'YYYY년 MM월'),
3행부터 지점마다 내국인·외국인·합계 행. 월 열만 읽고(연도 인원계·총계 열은 버림), 빈 칸인 달은
아직 공표 전이라 행을 만들지 않는다. 값은 표 그대로 옮긴다(나누기·채우기·추정 없음).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
from pathlib import Path
from typing import Any

from . import common
from .common import Column, SheetResult

DATASET = "kto_spot_visitors"
INPUT_DIR = common.REAL_ROOT / "kto"
SOURCE_URL = "https://know.tour.go.kr/stat/visitStatDis/table.do"

_MONTH_RE = re.compile(r"^(\d{4})년 (\d{2})월$")
_KIND_FIELD = {"내국인": "visitors_domestic", "외국인": "visitors_foreign", "합계": "visitors_total"}
_HEADER_ROWS = 2

COLUMNS = [
    Column("base_ym", "월", "yearmonth", "기준년월(YYYYMM)"),
    Column("sido_name", "시도", "string", ""),
    Column("sigungu_name", "군구", "string", ""),
    Column("spot_name", "관광지", "string", "통계표 관광지명 그대로"),
    Column("visitors_domestic", "내국인", "integer", "표에 없으면 빈 값"),
    Column("visitors_foreign", "외국인", "integer", "표에 없으면 빈 값"),
    Column("visitors_total", "합계", "integer", "통계표 합계 행 값"),
    Column("source_file", "", "string", "내려받은 파일 이름"),
    Column("checked_on", "", "date", "파일을 받은 날짜"),
]


def _count(value: Any) -> int | None:
    """셀 값 → 정수. 빈 칸은 None(공표 전), 정수가 아니면 예외."""
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    number = float(text)
    if not number.is_integer() or number < 0:
        raise ValueError(f"입장객 수가 0 이상의 정수가 아닙니다: {value!r}")
    return int(number)


def parse_rows(rows: list[list[Any]]) -> list[dict[str, Any]]:
    """통계표 셀 행 목록 → [{base_ym, sido_name, sigungu_name, spot_name, visitors_*}] (지점·월 순서 유지)."""
    if len(rows) <= _HEADER_ROWS or [str(c).strip() for c in rows[0][:4]] != ["시도", "군구", "관광지", "내/외국인"]:
        raise ValueError("주요관광지점 입장객 통계표(셀병합 해제) 형식이 아닙니다.")
    months = {}
    for idx, cell in enumerate(rows[1]):
        match = _MONTH_RE.match(str(cell).strip())
        if match:
            months[idx] = int(match.group(1)) * 100 + int(match.group(2))
    if not months:
        raise ValueError("월 열('YYYY년 MM월')이 없습니다 — 기간 구분을 '월별'로 조회한 파일이어야 합니다.")

    records: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for row in rows[_HEADER_ROWS:]:
        sido, sigungu, spot, kind = (str(c).strip() for c in row[:4])
        field = _KIND_FIELD.get(kind)
        if not spot or field is None:
            continue
        for idx, ym in months.items():
            value = _count(row[idx]) if idx < len(row) else None
            if value is None:
                continue
            key = (sido, sigungu, spot, ym)
            record = records.setdefault(key, {
                "base_ym": ym, "sido_name": sido, "sigungu_name": sigungu, "spot_name": spot,
                "visitors_domestic": None, "visitors_foreign": None, "visitors_total": None,
            })
            record[field] = value
    missing = [k for k, r in records.items() if r["visitors_total"] is None]
    if missing:
        raise ValueError(f"합계 행이 없는 지점·월이 있습니다: {missing[:3]}")
    return sorted(records.values(), key=lambda r: (r["sigungu_name"], r["spot_name"], r["base_ym"]))


def read_xls(path: Path) -> list[list[Any]]:
    """.xls 첫 시트의 셀 값 행 목록."""
    try:
        import xlrd  # 선택 의존성 — pip install xlrd (통계표 내려받기 파일이 .xls 형식)
    except ImportError as exc:
        raise RuntimeError("통계표 .xls 를 읽으려면 xlrd 가 필요합니다: pip install xlrd") from exc
    sheet = xlrd.open_workbook(str(path)).sheet_by_index(0)
    return [sheet.row_values(r) for r in range(sheet.nrows)]


def merge_files(parsed: list[tuple[str, str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    """(파일 이름, 받은 날짜, 행들) 여러 개 → 지점·월 1행. 같은 지점·월이 다른 값으로 겹치면 예외."""
    merged: dict[tuple[str, str, int], dict[str, Any]] = {}
    for name, checked, rows in parsed:
        for row in rows:
            key = (row["sigungu_name"], row["spot_name"], row["base_ym"])
            before = merged.get(key)
            if before and before["visitors_total"] != row["visitors_total"]:
                raise ValueError(f"같은 지점·월 값이 파일마다 다릅니다: {key} ({before['source_file']}, {name})")
            if not before:
                merged[key] = {**row, "source_file": name, "checked_on": checked}
    return [merged[k] for k in sorted(merged)]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(_extracted_dir: Path) -> list[dict[str, Any]]:
    files = sorted(INPUT_DIR.glob("*.xls"))
    if not files:
        raise FileNotFoundError(f"통계표 내려받기 파일(.xls)이 없습니다: {INPUT_DIR}")
    parsed = [
        (f.name, dt.date.fromtimestamp(f.stat().st_mtime).isoformat(), parse_rows(read_xls(f)))
        for f in files
    ]
    rows = merge_files(parsed)
    fields = [c.name for c in COLUMNS]
    result = SheetResult(
        dataset=DATASET,
        source_csv=f"real/kto/*.xls ({len(files)}개)",
        out_csv=f"{DATASET}/01_spot_monthly_visitors.csv",
        header_row_in_source=_HEADER_ROWS,
        rows_in=sum(len(p[2]) for p in parsed),
        header=fields,
        rows=[["" if r[f] is None else str(r[f]) for f in fields] for r in rows],
        columns=COLUMNS,
        key_columns=["spot_name", "base_ym"],
        transforms=["unpivot_month_columns", "merge_files"],
        notes=[f"출처: {SOURCE_URL} 통계표 내려받기(셀병합 해제)"]
        + [f"{f.name} sha256 {_sha256(f)}" for f in files],
    )
    return [common.write_sheet(result)]
