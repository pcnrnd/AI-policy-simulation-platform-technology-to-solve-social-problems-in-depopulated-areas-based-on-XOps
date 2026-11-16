"""관광지점 입장객·생활인구 준비 규칙 검증 — 통계표와 같은 모양의 합성 숫자만 쓴다(실제 지점 값 없음)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import load_external_data  # noqa: E402
import prepare_sheets  # noqa: E402
from prepare_rules import common, kto_spot_visitors, mois_living_population  # noqa: E402

_HEAD = ["시도", "군구", "관광지", "내/외국인", "총계(2024.11 ~ 2025.02)", "2024년", "", "", "2025년", "", ""]
_SUB = ["", "", "", "", "", "인원계", "2024년 11월", "2024년 12월", "인원계", "2025년 01월", "2025년 02월"]


def _sheet(*rows: list[object]) -> list[list[object]]:
    return [_HEAD, _SUB, *rows]


def test_month_columns_unpivot_and_year_totals_are_ignored() -> None:
    rows = _sheet(
        ["가도", "나군", "합성관광지", "내국인", 60.0, 30.0, 10.0, 20.0, 30.0, 30.0, ""],
        ["가도", "나군", "합성관광지", "외국인", 6.0, 3.0, 1.0, 2.0, 3.0, 3.0, ""],
        ["가도", "나군", "합성관광지", "합계", 66.0, 33.0, 11.0, 22.0, 33.0, 33.0, ""],
    )
    out = kto_spot_visitors.parse_rows(rows)

    assert [r["base_ym"] for r in out] == [202411, 202412, 202501]  # 빈 칸인 2025-02는 행이 없다
    assert out[0] == {
        "base_ym": 202411, "sido_name": "가도", "sigungu_name": "나군", "spot_name": "합성관광지",
        "visitors_domestic": 10, "visitors_foreign": 1, "visitors_total": 11,
    }


def test_spot_without_foreign_row_keeps_foreign_empty() -> None:
    rows = _sheet(
        ["가도", "나군", "합성섬", "내국인", "1,200", "", "", "", "", "1,200", ""],
        ["가도", "나군", "합성섬", "합계", "1,200", "", "", "", "", "1,200", ""],
    )
    (only,) = kto_spot_visitors.parse_rows(rows)
    assert only["base_ym"] == 202501 and only["visitors_total"] == 1200 and only["visitors_foreign"] is None


@pytest.mark.parametrize(
    "rows, message",
    [
        ([["지역", "a"], _SUB, ["x"]], "형식"),
        ([_HEAD, ["", "", "", "", "", "인원계"], ["가도", "나군", "합성", "합계", 1, 1]], "월 열"),
        (_sheet(["가도", "나군", "합성", "내국인", 1, 1, 1, "", "", "", ""]), "합계 행"),
        (_sheet(["가도", "나군", "합성", "합계", 1, 1, 1.5, "", "", "", ""]), "정수"),
    ],
)
def test_malformed_sheets_are_rejected(rows: list[list[object]], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        kto_spot_visitors.parse_rows(rows)


def test_merge_files_dedupes_equal_rows_and_rejects_conflicts() -> None:
    row = {"base_ym": 202501, "sido_name": "가도", "sigungu_name": "나군", "spot_name": "합성",
           "visitors_domestic": 5, "visitors_foreign": None, "visitors_total": 5}
    merged = kto_spot_visitors.merge_files([("a.xls", "2026-01-01", [row]), ("b.xls", "2026-01-02", [row])])
    assert len(merged) == 1 and merged[0]["source_file"] == "a.xls"

    with pytest.raises(ValueError, match="다릅니다"):
        kto_spot_visitors.merge_files([("a.xls", "d", [row]), ("b.xls", "d", [{**row, "visitors_total": 6}])])


_MOIS_ROW = {
    "sigungu_name": "합성군", "base_quarter": "2025-Q2", "registered_population": "1.5",
    "stay_population": "4.0", "unit": "만 명", "source_title": "합성 발표", "source_url": "https://example.org",
    "checked_on": "2026-01-01",
}


def test_living_population_rows_split_quarter_and_sort() -> None:
    rows = mois_living_population.parse_rows([_MOIS_ROW, {**_MOIS_ROW, "base_quarter": "2025-Q1"}])
    assert [r[:5] for r in rows] == [["합성군", "2025", "1", "1.5", "4.0"], ["합성군", "2025", "2", "1.5", "4.0"]]


@pytest.mark.parametrize(
    "bad, message",
    [
        ({"base_quarter": "2025 2분기"}, "YYYY-Qn"),
        ({"stay_population": "-1"}, "음수"),
    ],
)
def test_living_population_rejects_bad_values(bad: dict[str, str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        mois_living_population.parse_rows([{**_MOIS_ROW, **bad}])


def test_living_population_rejects_duplicates_and_missing_columns() -> None:
    with pytest.raises(ValueError, match="두 번"):
        mois_living_population.parse_rows([_MOIS_ROW, _MOIS_ROW])
    with pytest.raises(ValueError, match="열이 없습니다"):
        mois_living_population.parse_rows([{"sigungu_name": "합성군"}])


def test_only_option_keeps_other_manifest_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(common, "PREPARED_ROOT", tmp_path)
    with (tmp_path / "MANIFEST.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=common.MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerow({k: "x" for k in common.MANIFEST_FIELDS} | {"dataset": "namwon_kt_visitors"})
        writer.writerow({k: "x" for k in common.MANIFEST_FIELDS} | {"dataset": "kto_spot_visitors"})

    kept = prepare_sheets.kept_manifest_rows({"kto_spot_visitors"})
    assert [r["dataset"] for r in kept] == ["namwon_kt_visitors"]


def test_new_datasets_load_into_ext_tables() -> None:
    assert load_external_data.DATASET_PREFIX["kto_spot_visitors"] == "ext_kto"
    assert load_external_data.DATASET_PREFIX["mois_living_population"] == "ext_mois"
    assert load_external_data._table_name("kto_spot_visitors/01_spot_monthly_visitors.csv") == "spot_monthly_visitors"
