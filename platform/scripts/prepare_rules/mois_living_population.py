"""mois_living_population: 행정안전부 생활인구 발표 값 → 지역·분기 1행, 지역별 월평균 체류인구 1행.

입력 1: `platform/data/real/mois/living_population_quarterly.csv` — 행정안전부 인구감소지역 생활인구
분기 발표 첨부의 표 값을 그대로 옮긴 파일(첨부가 HWP·PDF라 표 추출 대신 옮겨 적는다).
머리: sigungu_name, base_quarter(YYYY-Qn), registered_population, stay_population, unit,
source_title, source_url, checked_on. 값은 발표 표 그대로(합산·나누기 없음), 저장소에 넣지 않는다.

입력 2: `platform/data/real/mois/stay_monthly_average.csv` — 같은 발표 첨부의 월별 체류인구를 기간 평균한
지역별 값(시설 추천 계산의 지역 규모, 화면에 값으로 나가지 않음). 머리: sigungu_name, period_from(YYYY-MM),
period_to(YYYY-MM), stay_monthly_average, unit, method, source_title, source_url, checked_on. 저장소에 넣지 않는다.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any

from . import common
from .common import Column, SheetResult

DATASET = "mois_living_population"
INPUT_FILE = common.REAL_ROOT / "mois" / "living_population_quarterly.csv"
AVERAGE_FILE = common.REAL_ROOT / "mois" / "stay_monthly_average.csv"

_QUARTER_RE = re.compile(r"^(\d{4})-Q([1-4])$")
_MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_REQUIRED = [
    "sigungu_name", "base_quarter", "registered_population", "stay_population",
    "unit", "source_title", "source_url", "checked_on",
]

COLUMNS = [
    Column("sigungu_name", "지역", "string", ""),
    Column("base_year", "분기", "integer", "기준 연도"),
    Column("quarter", "분기", "integer", "1~4"),
    Column("registered_population", "등록인구", "number", "발표 표 값 그대로"),
    Column("stay_population", "체류인구", "number", "발표 표 값 그대로"),
    Column("unit", "단위", "string", "발표 표 단위"),
    Column("source_title", "", "string", "발표 자료 이름"),
    Column("source_url", "", "string", ""),
    Column("checked_on", "", "date", "확인한 날짜"),
]

_AVERAGE_REQUIRED = [
    "sigungu_name", "period_from", "period_to", "stay_monthly_average",
    "unit", "method", "source_title", "source_url", "checked_on",
]
AVERAGE_COLUMNS = [
    Column("sigungu_name", "지역", "string", ""),
    Column("period_from", "기간 시작", "string", "YYYY-MM"),
    Column("period_to", "기간 끝", "string", "YYYY-MM"),
    Column("stay_monthly_average", "월평균 체류인구", "number", "기간 월별 체류인구의 평균"),
    Column("unit", "단위", "string", ""),
    Column("method", "", "string", "산정 방법"),
    Column("source_title", "", "string", "발표 자료 이름"),
    Column("source_url", "", "string", ""),
    Column("checked_on", "", "date", "확인한 날짜"),
]


def _number(value: str, field: str) -> str:
    text = value.strip().replace(",", "")
    try:
        number = float(text)
    except ValueError:
        raise ValueError(f"{field} 값이 숫자가 아닙니다: {value!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"{field} 값이 유한한 수가 아닙니다: {value!r}")
    if number < 0:
        raise ValueError(f"{field} 값이 음수입니다: {value!r}")
    return text


def parse_rows(records: list[dict[str, str]]) -> list[list[str]]:
    """입력 CSV 행(dict) → prepared 행. 형식이 틀리거나 같은 지역·분기가 겹치면 예외."""
    if records and (missing := [k for k in _REQUIRED if k not in records[0]]):
        raise ValueError(f"입력 열이 없습니다: {missing}")
    out: dict[tuple[str, int, int], list[str]] = {}
    for rec in records:
        match = _QUARTER_RE.match(rec["base_quarter"].strip())
        if not match:
            raise ValueError(f"base_quarter 는 YYYY-Qn 형식이어야 합니다: {rec['base_quarter']!r}")
        region = rec["sigungu_name"].strip()
        year, quarter = int(match.group(1)), int(match.group(2))
        key = (region, year, quarter)
        if key in out:
            raise ValueError(f"같은 지역·분기가 두 번 있습니다: {key}")
        out[key] = [
            region, str(year), str(quarter),
            _number(rec["registered_population"], "registered_population"),
            _number(rec["stay_population"], "stay_population"),
            rec["unit"].strip(), rec["source_title"].strip(), rec["source_url"].strip(), rec["checked_on"].strip(),
        ]
    return [out[k] for k in sorted(out)]


def parse_average_rows(records: list[dict[str, str]]) -> list[list[str]]:
    """월평균 체류인구 입력 행 → prepared 행. 기간 형식·순서가 틀리거나 같은 지역이 겹치면 예외."""
    if records and (missing := [k for k in _AVERAGE_REQUIRED if k not in records[0]]):
        raise ValueError(f"입력 열이 없습니다: {missing}")
    out: dict[str, list[str]] = {}
    for rec in records:
        region = rec["sigungu_name"].strip()
        start, end = rec["period_from"].strip(), rec["period_to"].strip()
        if not (_MONTH_RE.match(start) and _MONTH_RE.match(end)) or start > end:
            raise ValueError(f"기간은 YYYY-MM 형식의 시작≤끝이어야 합니다: {start!r}~{end!r}")
        if region in out:
            raise ValueError(f"같은 지역이 두 번 있습니다: {region}")
        out[region] = [
            region, start, end, _number(rec["stay_monthly_average"], "stay_monthly_average"),
            *(rec[k].strip() for k in ("unit", "method", "source_title", "source_url", "checked_on")),
        ]
    return [out[k] for k in sorted(out)]


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"생활인구 발표 값 파일이 없습니다: {path}")
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def build(_extracted_dir: Path) -> list[dict[str, Any]]:
    records = _read(INPUT_FILE)
    rows = parse_rows(records)
    result = SheetResult(
        dataset=DATASET,
        source_csv=f"real/mois/{INPUT_FILE.name}",
        out_csv=f"{DATASET}/01_living_population_quarterly.csv",
        header_row_in_source=1,
        rows_in=len(records),
        header=[c.name for c in COLUMNS],
        rows=rows,
        columns=COLUMNS,
        key_columns=["sigungu_name", "base_year", "quarter"],
        transforms=["split_quarter"],
        notes=["행정안전부 인구감소지역 생활인구 분기 발표 첨부 표 값을 옮긴 입력"],
    )
    averages = _read(AVERAGE_FILE)
    average = SheetResult(
        dataset=DATASET,
        source_csv=f"real/mois/{AVERAGE_FILE.name}",
        out_csv=f"{DATASET}/02_stay_monthly_average.csv",
        header_row_in_source=1,
        rows_in=len(averages),
        header=[c.name for c in AVERAGE_COLUMNS],
        rows=parse_average_rows(averages),
        columns=AVERAGE_COLUMNS,
        key_columns=["sigungu_name"],
        transforms=[],
        notes=["같은 발표 첨부의 월별 체류인구를 기간 평균한 지역별 값을 옮긴 입력"],
    )
    return [common.write_sheet(result), common.write_sheet(average)]
