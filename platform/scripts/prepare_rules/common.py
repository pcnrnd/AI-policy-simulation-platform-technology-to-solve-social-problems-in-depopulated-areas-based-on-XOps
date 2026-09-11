"""2단계 전처리 공용 유틸리티: CSV 입출력, schema.json/MANIFEST 작성, 형 변환.

extracted/ CSV 를 읽어 prepared/ CSV + schema.json 을 쓰는 모든 데이터셋 규칙이
공유하는 저수준 헬퍼만 둔다. 시트별 파싱 로직(헤더 오프셋, 언피벗 규칙)은 각
데이터셋 모듈(bccard.py 등)에 둔다.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

EXTRACTED_ROOT = Path(__file__).resolve().parents[2] / "data" / "extracted"
PREPARED_ROOT = Path(__file__).resolve().parents[2] / "data" / "prepared"


def read_rows(csv_path: Path) -> list[list[str]]:
    """extracted/ 의 CSV 를 문자열 행 목록으로 읽는다."""
    with csv_path.open(encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


def write_rows(csv_path: Path, header: list[str], rows: list[list[str]]) -> None:
    """prepared/ CSV 를 UTF-8(BOM 없음)/QUOTE_MINIMAL 로 쓴다."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
        writer.writerows(rows)


def copy_bytes(src: Path, dst: Path) -> None:
    """설명 시트(kind=meta)를 바이트 그대로 복사한다."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


@dataclass
class Column:
    name: str
    source_name: str
    type: str  # string|integer|number|date|yearmonth
    description: str = ""


@dataclass
class SheetResult:
    """데이터 시트(kind=data) 1개의 변환 결과."""

    dataset: str
    source_csv: str  # extracted/ 기준 상대경로
    out_csv: str  # prepared/ 기준 상대경로
    header_row_in_source: int
    rows_in: int
    header: list[str]
    rows: list[list[str]]
    columns: list[Column]
    key_columns: list[str]
    dropped_rows: dict[str, int] = field(default_factory=dict)
    transforms: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def rows_out(self) -> int:
        return len(self.rows)


def write_sheet(result: SheetResult) -> dict[str, Any]:
    """prepared/ CSV + schema.json 을 쓰고 MANIFEST 행을 반환한다."""
    out_path = PREPARED_ROOT / result.out_csv
    write_rows(out_path, result.header, result.rows)

    schema = {
        "dataset": result.dataset,
        "source_csv": result.source_csv,
        "columns": [
            {
                "name": c.name,
                "source_name": c.source_name,
                "type": c.type,
                "description": c.description,
            }
            for c in result.columns
        ],
        "key_columns": result.key_columns,
        "row_count": result.rows_out,
        "dropped_rows": result.dropped_rows,
        "transforms": result.transforms,
        "notes": result.notes,
    }
    schema_path = out_path.with_suffix("").with_suffix(".schema.json")
    schema_path.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    return {
        "dataset": result.dataset,
        "source_csv": result.source_csv,
        "out_csv": result.out_csv,
        "kind": "data",
        "header_row_in_source": result.header_row_in_source,
        "rows_in": result.rows_in,
        "rows_out": result.rows_out,
        "rows_dropped": sum(result.dropped_rows.values()),
        "transforms": ";".join(result.transforms),
        "key_columns": ";".join(result.key_columns),
        "note": " / ".join(result.notes),
    }


def write_meta(
    dataset: str, source_csv: str, out_csv: str, rows_in: int
) -> dict[str, Any]:
    """kind=meta 시트를 바이트 복사하고 MANIFEST 행을 반환한다."""
    src = EXTRACTED_ROOT.parent / source_csv
    dst = PREPARED_ROOT / out_csv
    copy_bytes(src, dst)
    return {
        "dataset": dataset,
        "source_csv": source_csv,
        "out_csv": out_csv,
        "kind": "meta",
        "header_row_in_source": 0,
        "rows_in": rows_in,
        "rows_out": rows_in,
        "rows_dropped": 0,
        "transforms": "copy_bytes",
        "key_columns": "",
        "note": "설명 시트, 바이트 그대로 복사",
    }


def is_blank_row(row: list[str]) -> bool:
    return all(not cell.strip() for cell in row)


def clean_header_cell(value: str) -> str:
    """헤더 셀의 NBSP·앞뒤 공백만 정리한다(값은 건드리지 않음)."""
    return value.replace("\xa0", " ").strip()


def forward_fill(rows: list[list[str]], col_indices: list[int]) -> None:
    """지정한 헤더/차원 컬럼에 한해 병합셀 좌상단 값을 아래로 채운다(제자리 수정)."""
    last: dict[int, str] = {}
    for row in rows:
        for idx in col_indices:
            if idx >= len(row):
                continue
            if row[idx].strip():
                last[idx] = row[idx]
            elif idx in last:
                row[idx] = last[idx]


MANIFEST_FIELDS = [
    "dataset",
    "source_csv",
    "out_csv",
    "kind",
    "header_row_in_source",
    "rows_in",
    "rows_out",
    "rows_dropped",
    "transforms",
    "key_columns",
    "note",
]


def write_manifest(manifest_rows: list[dict[str, Any]]) -> Path:
    manifest_path = PREPARED_ROOT / "MANIFEST.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(manifest_rows)
    return manifest_path
