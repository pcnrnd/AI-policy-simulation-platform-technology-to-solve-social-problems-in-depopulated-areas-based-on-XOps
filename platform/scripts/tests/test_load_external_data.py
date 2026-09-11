"""load_external_data.py 검증: 합성 schema.json+CSV(tmp_path)만 사용, DB 접속은 하지 않는다."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import load_external_data as led  # noqa: E402


def _write_dataset(tmp_path: Path, *, dataset: str, out_csv: str, columns: list[dict], rows: list[list[str]]) -> Path:
    prepared_dir = tmp_path
    csv_path = prepared_dir / out_csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = [c["name"] for c in columns]
    lines = [",".join(header)] + [",".join(row) for row in rows]
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    schema_path = prepared_dir / (str(Path(out_csv).with_suffix("")) + ".schema.json")
    schema_path.write_text(json.dumps({"columns": columns}), encoding="utf-8")

    manifest_path = prepared_dir / "MANIFEST.csv"
    key_columns = ";".join(c["name"] for c in columns[:1])
    manifest_line = (
        f"{dataset},extracted/{out_csv},{out_csv},data,1,{len(rows)},{len(rows)},0,"
        f"unpivot,{key_columns},note\n"
    )
    header_line = (
        "dataset,source_csv,out_csv,kind,header_row_in_source,rows_in,rows_out,"
        "rows_dropped,transforms,key_columns,note\n"
    )
    if manifest_path.exists():
        existing = manifest_path.read_text(encoding="utf-8")
        manifest_path.write_text(existing + manifest_line, encoding="utf-8")
    else:
        manifest_path.write_text(header_line + manifest_line, encoding="utf-8")
    return prepared_dir


def _sample_columns() -> list[dict]:
    return [
        {"name": "base_ym", "type": "yearmonth"},
        {"name": "category", "type": "string"},
        {"name": "amount", "type": "integer"},
        {"name": "ratio", "type": "number"},
        {"name": "obs_date", "type": "date"},
    ]


def test_ddl_maps_types_and_builds_key_index(tmp_path: Path) -> None:
    """DDL 생성 시 5가지 타입이 올바른 SQL 형으로 매핑되고 key_columns 인덱스가 생성된다."""
    prepared_dir = _write_dataset(
        tmp_path,
        dataset="namwon_bccard_consumption",
        out_csv="namwon_bccard_consumption/02_dong_industry_sales.csv",
        columns=_sample_columns(),
        rows=[["201901", "식음료", "100", "1.5", "2019-01-01"]],
    )
    rows = led._load_manifest_rows(prepared_dir, only=None)
    spec = led._build_spec(rows[0], prepared_dir)

    assert spec.table == "ext_bccard_dong_industry_sales"
    assert spec.columns == (
        ("base_ym", "INTEGER"),
        ("category", "TEXT"),
        ("amount", "BIGINT"),
        ("ratio", "NUMERIC"),
        ("obs_date", "DATE"),
    )

    statements = led.ddl_for(spec)
    assert not any("CREATE SCHEMA" in s for s in statements)
    assert any("CREATE TABLE IF NOT EXISTS ext_bccard_dong_industry_sales" in s for s in statements)
    assert any(
        "CREATE INDEX IF NOT EXISTS ix_ext_bccard_dong_industry_sales_key" in s and "(base_ym)" in s
        for s in statements
    )


def test_table_name_composes_prefix_and_slug(tmp_path: Path) -> None:
    """테이블명은 DATASET_PREFIX 접두 + '_' + slug 로 합성되고 public 스키마에 그대로 쓰인다(점 없음)."""
    prepared_dir = _write_dataset(
        tmp_path,
        dataset="nowon_kt_festival",
        out_csv="nowon_kt_festival/02_daily_visitors.csv",
        columns=_sample_columns(),
        rows=[["202001", "x", "1", "1.0", "2020-01-01"]],
    )
    rows = led._load_manifest_rows(prepared_dir, only=None)
    spec = led._build_spec(rows[0], prepared_dir)
    assert spec.table == "ext_kt_nowon_daily_visitors"
    assert "." not in spec.table


@pytest.mark.parametrize("suffix_len,should_raise", [(52, False), (53, True)])
def test_composed_table_name_63_char_boundary(tmp_path: Path, suffix_len: int, should_raise: bool) -> None:
    """접두(ext_bccard_, 11자) + slug 합성 결과가 63자를 넘으면 줄이지 않고 예외를 낸다."""
    slug = "a" * suffix_len
    prepared_dir = _write_dataset(
        tmp_path,
        dataset="namwon_bccard_consumption",
        out_csv=f"namwon_bccard_consumption/02_{slug}.csv",
        columns=_sample_columns(),
        rows=[["201901", "식음료", "100", "1.5", "2019-01-01"]],
    )
    rows = led._load_manifest_rows(prepared_dir, only=None)
    if should_raise:
        with pytest.raises(ValueError):
            led._build_spec(rows[0], prepared_dir)
    else:
        spec = led._build_spec(rows[0], prepared_dir)
        assert len(spec.table) == 63


def test_unknown_column_type_falls_back_to_text(tmp_path: Path) -> None:
    """schema.json에 매핑 없는 타입이 있으면 경고와 함께 TEXT로 적재한다."""
    columns = [{"name": "base_ym", "type": "yearmonth"}, {"name": "raw", "type": "unknown_type"}]
    prepared_dir = _write_dataset(
        tmp_path,
        dataset="gwto_kt_tourism_indicators",
        out_csv="gwto_kt_tourism_indicators/01_s1_1_nonlocal_monthly.csv",
        columns=columns,
        rows=[["202001", "x"]],
    )
    rows = led._load_manifest_rows(prepared_dir, only=None)
    spec = led._build_spec(rows[0], prepared_dir)
    assert dict(spec.columns)["raw"] == "TEXT"


@pytest.mark.parametrize("bad_name", ["Bad_Name", "1leading", "has space", "has-dash", "a" * 64])
def test_unsafe_identifier_is_rejected(bad_name: str) -> None:
    """대문자·숫자시작·공백·하이픈·63자 초과 식별자는 거부된다."""
    with pytest.raises(ValueError):
        led._check_identifier(bad_name, kind="테스트")


def test_table_name_strips_numeric_prefix_only() -> None:
    """`<NN>_<slug>.csv` 에서 숫자 접두사만 제거하고 slug 내부 숫자는 보존한다."""
    assert led._table_name("namwon_bccard_consumption/02_dong_industry_sales.csv") == "dong_industry_sales"
    assert led._table_name("gwto_kt_tourism_indicators/01_s1_1_nonlocal_monthly.csv") == "s1_1_nonlocal_monthly"


def test_copy_sql_converts_empty_string_to_null(tmp_path: Path) -> None:
    """COPY 문에 NULL '' 옵션이 있어 빈 문자열 셀이 NULL로 적재된다(DB 접속 없이 SQL 문자열만 검증)."""
    prepared_dir = _write_dataset(
        tmp_path,
        dataset="nowon_kt_festival",
        out_csv="nowon_kt_festival/02_daily_visitors.csv",
        columns=_sample_columns(),
        rows=[["202001", "", "", "", ""]],
    )
    rows = led._load_manifest_rows(prepared_dir, only=None)
    spec = led._build_spec(rows[0], prepared_dir)
    assert led._copy_sql(spec) == "COPY ext_kt_nowon_daily_visitors FROM STDIN WITH (FORMAT csv, HEADER true, NULL '')"
    assert spec.csv_path.read_text(encoding="utf-8").splitlines()[1] == "202001,,,,"
