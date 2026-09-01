"""실 공공데이터 적재 — platform/data/real/ → PostGIS(pg 컨테이너).

적재 대상 (카탈로그 ds_08·ds_09 와 1:1):
- geo_admin_boundary   ← admin_boundary_kostat2013.geojson (전국 시군구 251, EPSG:4326)
- tb_welfare_facility  ← 남원 사회복지시설 + 신안 장애인복지시설 CSV

값은 전부 psycopg 파라미터로 바인딩한다(리터럴 조립 없음). 재실행하면 지우고 다시
넣으므로 멱등이다(원본 파일이 단일 진실 원천).

사용법: python platform/scripts/load_real_data.py [--pg-dsn DSN] [--data-dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger("load_real_data")

DEFAULT_PG_DSN = "postgresql://xops:xops@localhost:5433/xops_dataops"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "real"

_DDL = """
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS geo_admin_boundary (
    adm_code      VARCHAR(5)  NOT NULL,
    adm_name      VARCHAR(40) NOT NULL,
    adm_name_eng  VARCHAR(80),
    base_year     VARCHAR(4)  NOT NULL,
    geom_boundary GEOMETRY(MultiPolygon, 4326) NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_admin_boundary_code ON geo_admin_boundary (adm_code);
CREATE INDEX IF NOT EXISTS ix_admin_boundary_geom ON geo_admin_boundary USING GIST (geom_boundary);

CREATE TABLE IF NOT EXISTS tb_welfare_facility (
    facility_id    VARCHAR(12)  NOT NULL,
    region_code    VARCHAR(5)   NOT NULL,
    sigungu_name   VARCHAR(20)  NOT NULL,
    facility_name  VARCHAR(80)  NOT NULL,
    facility_type  VARCHAR(40)  NOT NULL,
    road_address   VARCHAR(200),
    phone          VARCHAR(20),
    staff_count    INTEGER,
    latitude       DOUBLE PRECISION,
    longitude      DOUBLE PRECISION,
    data_base_date VARCHAR(10)  NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_welfare_facility_id ON tb_welfare_facility (facility_id);
"""


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _optional_int(value: str | None, *, field: str) -> int | None:
    """빈 값은 NULL로, 잘못된 정수는 경고 후 NULL로 정규화한다."""
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        _log.warning("%s 값 %r을 정수로 변환할 수 없어 NULL로 적재합니다.", field, value)
        return None


def load_boundaries(cur: Any, data_dir: Path) -> int:
    """전국 시군구 경계 GeoJSON → geo_admin_boundary. Polygon 은 ST_Multi 로 승격."""
    features = json.loads((data_dir / "admin_boundary_kostat2013.geojson").read_text(encoding="utf-8"))[
        "features"
    ]
    cur.execute("DELETE FROM geo_admin_boundary")
    rows = [
        (
            f["properties"]["code"],
            f["properties"]["name"],
            f["properties"].get("name_eng"),
            f["properties"].get("base_year", "2013"),
            json.dumps(f["geometry"]),
        )
        for f in features
    ]
    cur.executemany(
        "INSERT INTO geo_admin_boundary (adm_code, adm_name, adm_name_eng, base_year, geom_boundary)"
        " VALUES (%s, %s, %s, %s, ST_Multi(ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326)))",
        rows,
    )
    return len(rows)


def _namwon_rows(data_dir: Path) -> list[tuple[Any, ...]]:
    """남원시 사회복지시설현황 — 좌표·종사자수 미제공 컬럼은 NULL."""
    rows: list[tuple[Any, ...]] = []
    for i, r in enumerate(_read_csv(data_dir / "namwon_welfare_facilities_20251107.csv"), start=1):
        rows.append(
            (
                f"NW-WF-{i:03d}",
                "45190",
                "남원시",
                r["시설명"],
                "사회복지시설",
                r.get("도로명주소") or None,
                r.get("전화번호") or None,
                None,
                None,
                None,
                r["데이터기준일자"],
            )
        )
    return rows


def _sinan_rows(data_dir: Path) -> list[tuple[Any, ...]]:
    """신안군 장애인복지시설현황 — 위경도·종사원·시설유형 포함."""
    rows: list[tuple[Any, ...]] = []
    for i, r in enumerate(
        _read_csv(data_dir / "sinan_disabled_welfare_facilities_20260608.csv"), start=1
    ):
        rows.append(
            (
                f"SA-WF-{i:03d}",
                "46910",
                "신안군",
                r["시설명"],
                r.get("시설유형") or "장애인복지시설",
                r.get("소재지도로명주소") or None,
                r.get("전화번호") or None,
                _optional_int(r.get("종사원"), field=f"{r.get('시설명', '<unknown>')} 종사원"),
                float(r["위도"]) if r.get("위도") else None,
                float(r["경도"]) if r.get("경도") else None,
                r["데이터기준일자"],
            )
        )
    return rows


def load_facilities(cur: Any, data_dir: Path) -> int:
    cur.execute("DELETE FROM tb_welfare_facility")
    rows = _namwon_rows(data_dir) + _sinan_rows(data_dir)
    cur.executemany(
        "INSERT INTO tb_welfare_facility (facility_id, region_code, sigungu_name, facility_name,"
        " facility_type, road_address, phone, staff_count, latitude, longitude, data_base_date)"
        " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        rows,
    )
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pg-dsn", default=DEFAULT_PG_DSN)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    import psycopg  # 선택 의존성 — pip install "psycopg[binary]"

    with psycopg.connect(args.pg_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(_DDL)
            boundaries = load_boundaries(cur, args.data_dir)
            facilities = load_facilities(cur, args.data_dir)
    _log.info("geo_admin_boundary: %d rows", boundaries)
    _log.info("tb_welfare_facility: %d rows", facilities)


if __name__ == "__main__":
    main()
