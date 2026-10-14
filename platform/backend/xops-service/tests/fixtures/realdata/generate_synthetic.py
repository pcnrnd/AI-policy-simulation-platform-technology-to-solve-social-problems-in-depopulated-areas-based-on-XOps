"""Deterministic test-only fixtures; no database, file input, clock or random source.

Run this file to regenerate the four JSON files, or pass --check to verify them.
Region identifiers and month boundaries exercise the application's mapping and
lag contracts. Every observation, coefficient and metric is invented here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FEATURES = ["y_lag1", "y_lag2", "y_lag3", "month_sin", "month_cos"]
DONGS = [("45190250", "운봉읍"), ("45190310", "주천면"), ("45190320", "수지면")]


def build_fixtures() -> dict[str, object]:
    visitors = []
    sales = []
    for month_index in range(16):
        month = (2022 + month_index // 12) * 100 + month_index % 12 + 1
        for region_index, (code, name) in enumerate(DONGS):
            # Deliberately nonlinear, positive arithmetic toy signals.
            wave = (month_index * 7 + region_index * 3) % 11
            visitors.append({
                "base_ym": month,
                "dong_code": code,
                "local_visitors": 211 + region_index * 31 + month_index * 5 + wave,
                "nonlocal_visitors": 97 + region_index * 19 + month_index * 9 + wave * 2,
                "foreign_visitors": 3 + region_index * 2 + month_index % 5,
            })
            # The two absent months exercise no-zero-fill behaviour.
            if month not in (202209, 202210):
                sales.append({"base_ym": month, "dong_name": name,
                              "sales_est_krw": 17003 + region_index * 2309 + month_index * 613 + wave * 101})

    content_hash = hashlib.sha256(encode(visitors)).hexdigest()
    fixtures: dict[str, object] = {
        "kt_monthly_visitors.json": visitors,
        "bc_dong_industry_sales.json": sales,
    }
    for version_index in (1, 2):
        scale = float(version_index)
        fixtures[f"artifact_v{version_index}.json"] = {
            "model_id": "namwon-nonlocal-visitors-next-month",
            "version": f"v{version_index}-synthetic",
            "dataset_id": f"ds-synthetic-{content_hash[:16]}",
            "content_hash": content_hash,
            "features": FEATURES.copy(),
            "coef": [round(scale * (i + 1) / 37, 8) for i in range(5)],
            "intercept": 13.0 + scale,
            "train_means": {name: float(i + 1) * 7.0 for i, name in enumerate(FEATURES)},
            "train_stds": {name: float(i + 1) + 0.5 for i, name in enumerate(FEATURES)},
            "lambda": 1.0,
            "eval": {
                "period": {"from": 202302, "to": 202304, "n": 9},
                "metrics": {"mae": 2.0 * scale, "rmse": 3.0 * scale, "wape": 0.02 * scale},
                "baseline": {"name": "yoy_or_prev_month", "mae": 9.0, "rmse": 11.0, "wape": 0.09},
            },
            "observed_end_month": 202304,
            "forecast_month": 202305,
            "train_rows": 30,
            "trained_at": "2000-01-01T00:00:00+00:00",
        }
    return fixtures


def encode(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    for name, value in build_fixtures().items():
        expected = encode(value)
        path = directory / name
        if args.check:
            if not path.exists() or path.read_bytes().replace(b"\r\n", b"\n") != expected:
                print(f"FAIL: regenerate synthetic fixture {name}")
                return 1
        else:
            path.write_bytes(expected)
        print(f"{name}: sha256={hashlib.sha256(expected).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
