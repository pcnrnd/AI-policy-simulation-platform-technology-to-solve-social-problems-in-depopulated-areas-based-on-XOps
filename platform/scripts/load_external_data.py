"""외부데이터(BC카드·KT·GWTO) prepared CSV → PostgreSQL(xops_dataops) `public` 스키마 적재.

적재 대상: `platform/data/prepared/MANIFEST.csv` 의 `kind == data` 39건과 각
`<NN>_<slug>.schema.json`. 테이블은 `public` 스키마에 두고 이름은
`DATASET_PREFIX` 접두(ext_bccard/ext_kt_namwon/ext_kt_nowon/ext_gwto) +
`_` + slug(파일명에서 NN_ 접두사를 뗀 나머지)로 합성한다. xops-service의
`safety.py` 가 점(.) 없는 단일 식별자만 허용하고 기존 실데이터 테이블도
같은 public 접두 방식이라 스키마를 따로 두지 않는다. 테이블·컬럼·
key_columns 식별자는 정규식 검증을 통과한 것만 SQL 문자열에 넣는다
(파라미터 바인딩이 불가능한 DDL/식별자 위치). 63자를 넘는 합성 이름은
줄이지 않고 예외로 거부한다.

값 적재는 prepared CSV를 그대로 `COPY ... FROM STDIN`으로 스트리밍한다(빈
문자열은 NULL로 변환됨). 재실행 시 테이블별 DELETE 후 다시 넣으므로 멱등이다.

사용법: python platform/scripts/load_external_data.py [--pg-dsn DSN]
  [--prepared-dir DIR] [--only DATASET]... [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger("load_external_data")

DEFAULT_PG_DSN = "postgresql://xops:xops@localhost:5433/xops_dataops"
DEFAULT_PREPARED_DIR = Path(__file__).resolve().parents[1] / "data" / "prepared"

# 식별자 검증 — xops-service safety.SQL_IDENTIFIER_RE 관행(영문 시작, 영숫자·밑줄만).
# 여기서는 prepare_sheets 산출물이 소문자만 쓰므로 소문자로 좁혀 더 엄격히 검증한다.
IDENT_RE = re.compile(r"^[a-z][a-z0-9_]{0,62}$")

DATASET_PREFIX = {
    "namwon_bccard_consumption": "ext_bccard",
    "namwon_kt_visitors": "ext_kt_namwon",
    "nowon_kt_festival": "ext_kt_nowon",
    "gwto_kt_tourism_indicators": "ext_gwto",
}

TYPE_MAP = {
    "string": "TEXT",
    "integer": "BIGINT",
    "number": "NUMERIC",
    "date": "DATE",
    "yearmonth": "INTEGER",
}


@dataclass(frozen=True)
class TableSpec:
    dataset: str
    table: str
    csv_path: Path
    columns: tuple[tuple[str, str], ...]  # (name, sql_type)
    key_columns: tuple[str, ...]
    rows_out: int


def _check_identifier(name: str, *, kind: str) -> str:
    if not IDENT_RE.match(name):
        raise ValueError(f"허용되지 않는 {kind}입니다(소문자·숫자·밑줄, 소문자 시작만): {name!r}")
    return name


def _table_name(out_csv: str) -> str:
    """`<NN>_<slug>.csv` → `<slug>` (NN 접두 제거)."""
    stem = Path(out_csv).stem
    prefix, sep, rest = stem.partition("_")
    return rest if sep and prefix.isdigit() else stem


def _sql_type(col_type: str, *, column: str) -> str:
    mapped = TYPE_MAP.get(col_type)
    if mapped is None:
        _log.warning("컬럼 %s 의 타입 %r 은 매핑이 없어 TEXT로 적재합니다.", column, col_type)
        return "TEXT"
    return mapped


def _load_manifest_rows(prepared_dir: Path, only: set[str] | None) -> list[dict[str, str]]:
    with (prepared_dir / "MANIFEST.csv").open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["kind"] == "data"]
    if only:
        rows = [r for r in rows if r["dataset"] in only]
    return rows


def _build_spec(row: dict[str, str], prepared_dir: Path) -> TableSpec:
    dataset = row["dataset"]
    prefix = DATASET_PREFIX[dataset]
    slug = _table_name(row["out_csv"])
    table = _check_identifier(f"{prefix}_{slug}", kind="테이블명")

    schema_path = prepared_dir / (str(Path(row["out_csv"]).with_suffix("")) + ".schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    columns = tuple(
        (_check_identifier(c["name"], kind="컬럼명"), _sql_type(c["type"], column=c["name"]))
        for c in schema["columns"]
    )
    key_columns = tuple(
        _check_identifier(k, kind="key_columns 컬럼명") for k in row["key_columns"].split(";") if k
    )
    return TableSpec(
        dataset=dataset,
        table=table,
        csv_path=prepared_dir / row["out_csv"],
        columns=columns,
        key_columns=key_columns,
        rows_out=int(row["rows_out"]),
    )


def ddl_for(spec: TableSpec) -> list[str]:
    """CREATE TABLE/INDEX 문 목록(전부 IF NOT EXISTS, public 스키마)."""
    col_defs = ",\n    ".join(f"{name} {sql_type}" for name, sql_type in spec.columns)
    statements = [
        f"CREATE TABLE IF NOT EXISTS {spec.table} (\n    {col_defs}\n);",
    ]
    if spec.key_columns:
        index_name = _check_identifier(f"ix_{spec.table}_key", kind="인덱스명")
        cols = ", ".join(spec.key_columns)
        statements.append(f"CREATE INDEX IF NOT EXISTS {index_name} ON {spec.table} ({cols});")
    return statements


def _copy_sql(spec: TableSpec) -> str:
    """COPY FROM STDIN 문 — CSV 헤더 사용, 빈 문자열은 NULL로 변환."""
    return f"COPY {spec.table} FROM STDIN WITH (FORMAT csv, HEADER true, NULL '')"


def load_table(cur: Any, spec: TableSpec) -> int:
    """DELETE 후 prepared CSV를 COPY FROM STDIN 으로 스트리밍(빈 문자열 → NULL)."""
    cur.execute(f"DELETE FROM {spec.table}")
    with spec.csv_path.open("rb") as f, cur.copy(_copy_sql(spec)) as copy:
        while chunk := f.read(1 << 20):
            copy.write(chunk)
    cur.execute(f"SELECT COUNT(*) FROM {spec.table}")
    return cur.fetchone()[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pg-dsn", default=DEFAULT_PG_DSN)
    parser.add_argument("--prepared-dir", type=Path, default=DEFAULT_PREPARED_DIR)
    parser.add_argument("--only", action="append", default=None, help="특정 dataset만 적재(반복 가능)")
    parser.add_argument("--dry-run", action="store_true", help="DDL만 출력하고 적재하지 않음")
    args = parser.parse_args()

    only = set(args.only) if args.only else None
    rows = _load_manifest_rows(args.prepared_dir, only)
    specs = [_build_spec(r, args.prepared_dir) for r in rows]

    if args.dry_run:
        for spec in specs:
            print("\n".join(ddl_for(spec)))
        return

    import psycopg  # 선택 의존성 — pip install "psycopg[binary]"

    mismatches: list[str] = []
    with psycopg.connect(args.pg_dsn) as conn:
        with conn.cursor() as cur:
            for spec in specs:
                for statement in ddl_for(spec):
                    cur.execute(statement)
            for spec in specs:
                count = load_table(cur, spec)
                label = spec.table
                if count != spec.rows_out:
                    mismatches.append(f"{label}: DB={count} rows_out={spec.rows_out}")
                    _log.error("%s 행수 불일치: DB=%d rows_out=%d", label, count, spec.rows_out)
                else:
                    _log.info("%s: %d rows", label, count)

    if mismatches:
        _log.error("행수 불일치 테이블 %d건: %s", len(mismatches), "; ".join(mismatches))
        sys.exit(1)


if __name__ == "__main__":
    main()
