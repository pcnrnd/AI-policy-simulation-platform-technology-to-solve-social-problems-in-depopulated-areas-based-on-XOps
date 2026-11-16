"""mois_living_population: 행정안전부 생활인구 분기 발표 값 → 지역·분기 1행.

입력: `platform/data/real/mois/living_population_quarterly.csv` — 행정안전부 인구감소지역 생활인구
분기 발표 첨부의 표 값을 그대로 옮긴 파일(첨부가 HWP·PDF라 표 추출 대신 옮겨 적는다).
머리: sigungu_name, base_quarter(YYYY-Qn), registered_population, stay_population, unit,
source_title, source_url, checked_on. 값은 발표 표 그대로(합산·나누기 없음), 저장소에 넣지 않는다.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from . import common
from .common import Column, SheetResult

DATASET = "mois_living_population"
INPUT_FILE = common.REAL_ROOT / "mois" / "living_population_quarterly.csv"

_QUARTER_RE = re.compile(r"^(\d{4})-Q([1-4])$")
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


def _number(value: str, field: str) -> str:
    text = value.strip().replace(",", "")
    number = float(text)
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


def build(_extracted_dir: Path) -> list[dict[str, Any]]:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"생활인구 발표 값 파일이 없습니다: {INPUT_FILE}")
    with INPUT_FILE.open(encoding="utf-8-sig", newline="") as f:
        records = list(csv.DictReader(f))
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
    return [common.write_sheet(result)]
