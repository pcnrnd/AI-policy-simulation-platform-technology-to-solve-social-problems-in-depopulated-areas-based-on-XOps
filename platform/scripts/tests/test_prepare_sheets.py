"""prepare_sheets.py 2단계 전처리 검증: 합성 CSV(tmp_path)만 사용, 실제 extracted/ 는 읽지 않는다."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from prepare_rules import common  # noqa: E402
from prepare_rules.gwto import blocks  # noqa: E402


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    common.write_rows(path, rows[0], rows[1:])


def test_header_offset_rows_are_dropped(tmp_path: Path) -> None:
    """헤더 위 제목/설명 행(header_row_in_source > 1)이 언피벗 결과에 섞이지 않는다."""
    rows = [
        ["[제목] 어떤 설명"],
        ["ㅇ 부가 설명 행"],
        ["구분", "2020년 1월", "2020년 2월"],
        ["카테고리A", "10", "20"],
    ]
    out, stats = blocks.melt_year_month_header_rows(rows, max_col=3)

    assert stats.title_rows == 2
    assert out == [("카테고리A", 2020, 1, "10"), ("카테고리A", 2020, 2, "20")]


def test_summary_rows_are_dropped(tmp_path: Path) -> None:
    """합계/누계/평균 행은 카운트만 되고 언피벗 결과에서 제외된다."""
    rows = [
        ["구분", "2020년 1월", "2020년 2월"],
        ["카테고리A", "10", "20"],
        ["카테고리B", "5", "7"],
        ["합계", "15", "27"],
    ]
    out, stats = blocks.melt_year_month_header_rows(rows, max_col=3)

    assert stats.total_rows == 1
    assert ("합계", 2020, 1, "15") not in out
    assert len(out) == 4  # 카테고리A/B x 2개월


def test_two_row_header_group_unpivot(tmp_path: Path) -> None:
    """2단 헤더(그룹+연-월) 세로 블록 반복을 (entity, year, month, subcat, value)로 언피벗한다."""
    rows = [
        ["구분", "2021년 1월", "", "", "2021년 2월", "", ""],
        ["시군구명", "외지인", "외국인", "전체", "외지인", "외국인", "전체"],
        ["가상시", "100", "5", "105", "110", "6", "116"],
    ]
    out, stats = blocks.melt_grouped_wide_blocks(rows)

    assert stats.value_cells == 6
    assert ("가상시", 2021, 1, "외지인", "100") in out
    assert ("가상시", 2021, 2, "전체", "116") in out


def test_name_value_pair_unpivot(tmp_path: Path) -> None:
    """(이름, 값) 쌍이 가로로 반복되는 구조를 (year, month, name, value)로 언피벗한다."""
    label_row = ["제목", "", "2025년 01월", "", "", "2024년 12월", "", ""]
    data_row = ["", "", "가상동", "111", "", "가상동", "222", ""]
    rows = [label_row, data_row]

    out, stats = blocks.melt_name_value_pairs(
        rows, label_row_idx=0, data_start_idx=1, pair_start_col=2, period=3
    )

    assert stats.value_cells == 2
    assert (2025, 1, "가상동", "111") in out
    assert (2024, 12, "가상동", "222") in out


def test_meta_sheet_is_copied_byte_for_byte(tmp_path: Path) -> None:
    """설명 시트(kind=meta)는 형 변환 없이 바이트 그대로 복사한다."""
    src = tmp_path / "extracted" / "01_overview.csv"
    src.parent.mkdir(parents=True)
    src.write_bytes("데이터명,비고\r\n남원시,\xa0비고텍스트\r\n".encode("utf-8"))

    dst = tmp_path / "prepared" / "01_overview.csv"
    common.copy_bytes(src, dst)

    assert dst.read_bytes() == src.read_bytes()
