"""남원 행정동 대응표 생성 — KT `ext_kt_namwon_monthly_dong_visitors` distinct(dong_code, dong_name)
을 기준으로 `platform/backend/xops-service/src/realdata/namwon_dong_map.json`을 만든다.
(data/는 .gitignore 대상·compose 볼륨 마운트 대상이라 코드와 함께 배포되는 src/ 쪽에 둔다.)

BC `ext_bccard_dong_industry_sales.dong_name` distinct 집합과 완전 일치하는지 검증하고,
콘솔에 결과를 보고한다. 불일치가 있어도 파일은 KT 기준으로 생성한다(계약 R1-4).
재실행해도 같은 DB 상태에서는 같은 내용을 만든다(멱등).

사용법: python platform/scripts/build_dong_map.py [--pg-dsn DSN] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger("build_dong_map")

DEFAULT_PG_DSN = "postgresql://xops:xops@localhost:5433/xops_dataops"
DEFAULT_OUT = (
    Path(__file__).resolve().parents[1] / "backend" / "xops-service" / "src" / "realdata" / "namwon_dong_map.json"
)
SOURCE_TABLE = "ext_kt_namwon_monthly_dong_visitors"


def build(pg_dsn: str, out_path: Path) -> dict:
    import psycopg

    with psycopg.connect(pg_dsn, connect_timeout=5) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT DISTINCT dong_code, dong_name FROM {SOURCE_TABLE} ORDER BY dong_code"
            )
            kt_rows = cur.fetchall()
            cur.execute("SELECT DISTINCT dong_name FROM ext_bccard_dong_industry_sales")
            bc_names = {row[0] for row in cur.fetchall()}

    kt_names = {name for _, name in kt_rows}
    missing_in_bc = sorted(kt_names - bc_names)
    extra_in_bc = sorted(bc_names - kt_names)
    if missing_in_bc or extra_in_bc:
        _log.warning(
            "KT/BC dong_name 불일치 — KT에만 있음: %s, BC에만 있음: %s",
            missing_in_bc,
            extra_in_bc,
        )
    else:
        _log.info("KT/BC dong_name 완전 일치 (%d건)", len(kt_names))

    doc = {
        "version": "v1",
        "source": SOURCE_TABLE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entries": [{"dong_name": name, "dong_code": code} for code, name in kt_rows],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _log.info("작성 완료: %s (%d건)", out_path, len(kt_rows))
    return doc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pg-dsn", default=DEFAULT_PG_DSN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    build(args.pg_dsn, args.out)


if __name__ == "__main__":
    main()
