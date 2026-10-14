"""Ensure fixture provenance stays reproducible and missing-data tests meaningful."""
import hashlib
from pathlib import Path
import runpy

DIRECTORY = Path(__file__).resolve().parents[1] / "fixtures" / "realdata"
GENERATOR = runpy.run_path(str(DIRECTORY / "generate_synthetic.py"))


def test_committed_synthetic_fixtures_are_independently_reproducible():
    first = GENERATOR["build_fixtures"]()
    second = runpy.run_path(str(DIRECTORY / "generate_synthetic.py"))["build_fixtures"]()
    assert first == second
    for name, value in first.items():
        assert (DIRECTORY / name).read_bytes().replace(b"\r\n", b"\n") == GENERATOR["encode"](value)


def test_synthetic_rows_preserve_lag_mapping_and_missing_month_contracts():
    fixtures = GENERATOR["build_fixtures"]()
    visitors = fixtures["kt_monthly_visitors.json"]
    sales = fixtures["bc_dong_industry_sales.json"]
    assert len(visitors) == 48
    assert len({r["base_ym"] for r in visitors}) == 16
    assert len({r["dong_code"] for r in visitors}) == 3
    assert min(r["base_ym"] for r in visitors) == 202201
    assert max(r["base_ym"] for r in visitors) == 202304
    assert all(set(r) == {"base_ym", "dong_code", "local_visitors", "nonlocal_visitors", "foreign_visitors"} for r in visitors)
    assert all(r[k] > 0 for r in visitors for k in ("local_visitors", "nonlocal_visitors", "foreign_visitors"))
    assert len(sales) == 42
    assert len({r["base_ym"] for r in sales}) == 14
    assert not {202209, 202210} & {r["base_ym"] for r in sales}
    assert all(set(r) == {"base_ym", "dong_name", "sales_est_krw"} and r["sales_est_krw"] > 0 for r in sales)


def test_synthetic_artifact_versions_have_no_untraceable_training_values():
    fixtures = GENERATOR["build_fixtures"]()
    content_hash = hashlib.sha256(GENERATOR["encode"](fixtures["kt_monthly_visitors.json"])).hexdigest()
    first = fixtures["artifact_v1.json"]
    second = fixtures["artifact_v2.json"]
    assert first["version"] != second["version"]
    assert first["coef"] != second["coef"]
    for artifact in (first, second):
        assert artifact["dataset_id"].startswith("ds-synthetic-")
        assert artifact["content_hash"] == content_hash
        assert len(artifact["coef"]) == len(artifact["features"]) == 5
        assert set(artifact["train_means"]) == set(artifact["features"])
        assert set(artifact["train_stds"]) == set(artifact["features"])
        assert artifact["forecast_month"] == 202305
        assert artifact["trained_at"] == "2000-01-01T00:00:00+00:00"
