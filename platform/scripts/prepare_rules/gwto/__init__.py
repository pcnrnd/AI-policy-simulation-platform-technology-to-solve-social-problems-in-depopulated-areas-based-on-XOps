"""gwto_kt_tourism_indicators: 26개 보고서형 와이드 시트 → 시트별 롱포맷 CSV.

전부 언피벗(unpivot) 대상이라 rows_out == rows_in - rows_dropped 검증 대신
"값 셀 수 보존"(원본 non-blank 측정값 셀 수 == rows_out)을 기준으로 삼는다.
시트별 규칙은 part1(01~09)/part2(10~18)/part3(19~26)에 나눠 둔다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import common
from ..common import Column, SheetResult
from .blocks import MeltStats

DATASET = "gwto_kt_tourism_indicators"


def to_result(
    slug: str,
    header_row_in_source: int,
    header: list[str],
    out_rows: list[list[str]],
    columns: list[Column],
    key_columns: list[str],
    stats: MeltStats,
    transforms: list[str],
    notes: list[str],
) -> dict[str, Any]:
    """언피벗 결과를 SheetResult 로 감싸 prepared/ 에 쓰고 MANIFEST 행을 반환한다."""
    result = SheetResult(
        dataset=DATASET,
        source_csv=f"extracted/{DATASET}/{slug}.csv",
        out_csv=f"{DATASET}/{slug}.csv",
        header_row_in_source=header_row_in_source,
        rows_in=stats.value_cells + stats.duplicate_cells,
        header=header,
        rows=out_rows,
        columns=columns,
        key_columns=key_columns,
        dropped_rows=stats.dropped_rows_dict(),
        transforms=["unpivot"] + transforms,
        notes=notes,
    )
    return common.write_sheet(result)


def build(extracted_dir: Path) -> list[dict[str, Any]]:
    from . import part1, part2, part3

    manifest: list[dict[str, Any]] = []
    manifest.extend(part1.build(extracted_dir))
    manifest.extend(part2.build(extracted_dir))
    manifest.extend(part3.build(extracted_dir))
    return manifest
