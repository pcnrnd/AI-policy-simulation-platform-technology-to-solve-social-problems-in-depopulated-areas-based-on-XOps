"""실데이터 스냅샷 — PG 원천을 (base_ym, dong_code) 단위로 모아 재현 가능한 데이터셋으로 굳힌다.

결측은 0으로 채우지 않는다(R1-5): 집계 행이 없는 (월, 동)은 스냅샷에 없다. 같은 spec으로
같은 DB 상태를 다시 읽으면 같은 `dataset_id`가 나온다(content_hash 기반, R1-6).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core import db
from src.core.settings import get_settings
from src.realdata import pg_reader
from src.realdata.errors import DatasetNotFound, MappingError

# target → model_id (R2). B·C는 이 상수로 model_id를 얻는다.
TARGETS: dict[str, str] = {
    "nonlocal_visitors": "namwon-nonlocal-visitors-next-month",
    "observed_sales_krw": "namwon-observed-sales-next-month",
}

_KT_MONTHLY_TABLE = "ext_kt_namwon_monthly_dong_visitors"
_BC_TABLE = "ext_bccard_dong_industry_sales"
_EXPECTED_DONG_COUNT = 23


@dataclass
class DatasetRecord:
    """스냅샷 1건 — `rows`는 원천 관측 그대로(0 채움 없음)."""

    dataset_id: str
    target: str
    model_id: str
    spec: dict[str, Any]
    quality: dict[str, Any]
    observed_from: int
    observed_to: int
    row_count: int
    excluded_rows: dict[str, int]
    content_hash: str
    file_path: str
    created_at: str
    rows: list[dict[str, Any]] = field(default_factory=list)

    def to_summary(self) -> dict[str, Any]:
        """rows를 뺀 요약 — API 목록/생성 응답에 쓴다."""
        data = asdict(self)
        data.pop("rows")
        return data


def _dong_map() -> tuple[dict[str, str], str]:
    """BC `dong_name` → KT `dong_code` 대응표와 파일 sha256."""
    path = get_settings().realdata_dong_map_path
    raw = path.read_text(encoding="utf-8")
    doc = json.loads(raw)
    mapping = {entry["dong_name"]: entry["dong_code"] for entry in doc["entries"]}
    return mapping, hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _canonical_json(rows: list[dict[str, Any]]) -> str:
    ordered = sorted(rows, key=lambda r: (r["base_ym"], r["dong_code"]))
    return json.dumps(ordered, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _content_hash(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(_canonical_json(rows).encode("utf-8")).hexdigest()


def _quality(rows: list[dict[str, Any]], *, mapping_total: int, mapping_matched: int) -> dict[str, Any]:
    months = sorted({r["base_ym"] for r in rows})
    dongs = sorted({r["dong_code"] for r in rows})
    month_count = len(months)
    dong_count = len(dongs)
    grid = month_count * _EXPECTED_DONG_COUNT
    seen: set[tuple[int, str]] = set()
    duplicate_keys = 0
    negative_count = 0
    for r in rows:
        key = (r["base_ym"], r["dong_code"])
        if key in seen:
            duplicate_keys += 1
        seen.add(key)
        for col, val in r.items():
            if col in ("base_ym", "dong_code") or val is None:
                continue
            if isinstance(val, (int, float)) and val < 0:
                negative_count += 1
    return {
        "month_count": month_count,
        "dong_count": dong_count,
        "expected_dong_count": _EXPECTED_DONG_COUNT,
        "missing_month_dong_count": max(grid - len(rows), 0),
        "mapping_matched": mapping_matched,
        "mapping_total": mapping_total,
        "negative_count": negative_count,
        "duplicate_key_count": duplicate_keys,
        "observed_cell_ratio": (len(rows) / grid) if grid else 0.0,
    }


def _to_number(value: Any) -> float | int | None:
    """pg_reader는 numeric(Decimal)을 문자열로 반환한다 — 여기서 다시 숫자로 되돌린다."""
    if value is None or isinstance(value, (int, float)):
        return value
    return float(value)


def _build_visitors_rows() -> list[dict[str, Any]]:
    raw = pg_reader.fetch_all(
        _KT_MONTHLY_TABLE,
        ["base_ym", "dong_code", "local_visitors", "nonlocal_visitors", "foreign_visitors"],
        order_by=["base_ym", "dong_code"],
    )
    return [
        {
            "base_ym": r["base_ym"],
            "dong_code": r["dong_code"],
            "y": _to_number(r["nonlocal_visitors"]),
            "local_visitors": _to_number(r["local_visitors"]),
            "foreign_visitors": _to_number(r["foreign_visitors"]),
        }
        for r in raw
    ]


def _build_sales_rows() -> tuple[list[dict[str, Any]], list[str]]:
    dong_name_to_code, _ = _dong_map()
    agg = pg_reader.fetch_aggregate(_BC_TABLE, group_by=["base_ym", "dong_name"], sums=["sales_est_krw"])

    unmapped = sorted({r["dong_name"] for r in agg if r["dong_name"] not in dong_name_to_code})
    if unmapped:
        raise MappingError(f"BC dong_name {len(unmapped)}건이 대응표에 없습니다: {unmapped}", unmapped)

    rows = []
    for r in agg:
        dong_code = dong_name_to_code[r["dong_name"]]
        base_ym = r["base_ym"]
        sales = _to_number(r["sales_est_krw"])
        rows.append(
            {
                "base_ym": base_ym,
                "dong_code": dong_code,
                "y": sales,
                "observed_sales_krw": sales,
            }
        )
    rows.sort(key=lambda r: (r["base_ym"], r["dong_code"]))
    return rows, unmapped


def _spec(target: str, spec_version: str, dong_map_sha256: str) -> dict[str, Any]:
    if target == "nonlocal_visitors":
        return {
            "target": target,
            "model_id": TARGETS[target],
            "tables": [_KT_MONTHLY_TABLE],
            "columns": {"y": "nonlocal_visitors", "extra": ["local_visitors", "foreign_visitors"]},
            "aggregation": None,
            "dong_map_sha256": None,
            "spec_version": spec_version,
            "rules": {"fill_missing": False, "lag_within_continuous_run_only": True},
        }
    return {
        "target": target,
        "model_id": TARGETS[target],
        "tables": [_BC_TABLE],
        "columns": {"y": "observed_sales_krw", "extra": []},
        "aggregation": "sum(sales_est_krw) group by (base_ym, dong_name)",
        "dong_map_sha256": dong_map_sha256,
        "spec_version": spec_version,
        "rules": {
            "fill_missing": False,
            "lag_within_continuous_run_only": True,
        },
    }


def _conn() -> sqlite3.Connection:
    return db._conn()  # ponytail: db.py는 루트 소유 — rd_datasets용 공개 CRUD가 아직 없어 기존 연결만 재사용


def _upsert_rd_dataset(record: DatasetRecord) -> str:
    """rd_datasets에 upsert하고 최종 created_at을 반환(재생성 시 기존 값 유지, R1-6)."""
    conn = _conn()
    existing = conn.execute(
        "SELECT created_at FROM rd_datasets WHERE dataset_id = ?", (record.dataset_id,)
    ).fetchone()
    created_at = existing["created_at"] if existing else record.created_at
    conn.execute(
        """
        INSERT INTO rd_datasets
            (dataset_id, spec_json, quality_json, observed_from, observed_to, row_count, content_hash, file_path, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(dataset_id) DO UPDATE SET
            spec_json = excluded.spec_json,
            quality_json = excluded.quality_json,
            observed_from = excluded.observed_from,
            observed_to = excluded.observed_to,
            row_count = excluded.row_count,
            content_hash = excluded.content_hash,
            file_path = excluded.file_path
        """,
        (
            record.dataset_id,
            json.dumps(record.spec, ensure_ascii=False),
            json.dumps(record.quality, ensure_ascii=False),
            record.observed_from,
            record.observed_to,
            record.row_count,
            record.content_hash,
            record.file_path,
            created_at,
        ),
    )
    conn.commit()
    return created_at


def create_dataset(target: str, *, spec_version: str = "v1") -> DatasetRecord:
    """스냅샷 생성 — 같은 spec·같은 DB 상태면 같은 dataset_id(R1-6)."""
    if target not in TARGETS:
        raise ValueError(f"알 수 없는 target입니다: {target!r} (허용: {sorted(TARGETS)})")

    if target == "nonlocal_visitors":
        rows = _build_visitors_rows()
        _, dong_map_sha256 = _dong_map()
        mapping_total = mapping_matched = _EXPECTED_DONG_COUNT
    else:
        rows, _ = _build_sales_rows()
        _, dong_map_sha256 = _dong_map()
        mapping_total = mapping_matched = _EXPECTED_DONG_COUNT

    spec = _spec(target, spec_version, dong_map_sha256)
    content_hash = _content_hash(rows)
    dataset_id = f"ds-{content_hash[:12]}"
    quality = _quality(rows, mapping_total=mapping_total, mapping_matched=mapping_matched)
    observed_from = min((r["base_ym"] for r in rows), default=0)
    observed_to = max((r["base_ym"] for r in rows), default=0)

    settings = get_settings()
    settings.realdata_dataset_dir.mkdir(parents=True, exist_ok=True)
    file_path = settings.realdata_dataset_dir / f"{dataset_id}.json"

    record = DatasetRecord(
        dataset_id=dataset_id,
        target=target,
        model_id=TARGETS[target],
        spec=spec,
        quality=quality,
        observed_from=observed_from,
        observed_to=observed_to,
        row_count=len(rows),
        excluded_rows={"mapping_failed": 0},
        content_hash=content_hash,
        file_path=str(file_path),
        created_at=datetime.now(timezone.utc).isoformat(),
        rows=rows,
    )

    created_at = _upsert_rd_dataset(record)
    record = DatasetRecord(**{**asdict(record), "created_at": created_at})
    file_path.write_text(json.dumps(asdict(record), ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def load_dataset(dataset_id: str) -> DatasetRecord:
    """SQLite 행 + 파일에서 재구성. 없으면 `DatasetNotFound`."""
    row = _conn().execute("SELECT * FROM rd_datasets WHERE dataset_id = ?", (dataset_id,)).fetchone()
    if row is None:
        raise DatasetNotFound(f"dataset_id를 찾을 수 없습니다: {dataset_id!r}")
    path = Path(row["file_path"])
    if not path.exists():
        raise DatasetNotFound(f"데이터셋 파일이 없습니다: {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    return DatasetRecord(**doc)


def list_datasets() -> list[dict[str, Any]]:
    """rd_datasets 목록(요약, rows 제외) — created_at 내림차순."""
    rows = _conn().execute("SELECT * FROM rd_datasets ORDER BY created_at DESC").fetchall()
    out = []
    for r in rows:
        out.append(
            {
                "dataset_id": r["dataset_id"],
                "spec": json.loads(r["spec_json"]),
                "quality": json.loads(r["quality_json"]),
                "observed_from": r["observed_from"],
                "observed_to": r["observed_to"],
                "row_count": r["row_count"],
                "content_hash": r["content_hash"],
                "file_path": r["file_path"],
                "created_at": r["created_at"],
            }
        )
    return out
