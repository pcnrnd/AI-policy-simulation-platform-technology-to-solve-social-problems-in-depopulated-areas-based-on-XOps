"""MLOps 모니터링 API 통합 테스트."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_metrics_history_has_six_series(client: TestClient) -> None:
    j = client.get("/api/v3/monitoring/metrics").json()
    for k in ("accuracy", "f1", "precision", "recall", "mse", "mae"):
        assert k in j["history"]
        assert k in j["latest"]


def test_compute_classification(client: TestClient) -> None:
    r = client.post("/api/v3/monitoring/metrics/classification", json={"y_true": [1, 0, 1], "y_pred": [1, 0, 1]})
    assert r.json()["f1"] == 1.0


def test_compute_regression(client: TestClient) -> None:
    r = client.post("/api/v3/monitoring/metrics/regression", json={"y_true": [1.0, 2.0], "y_pred": [1.0, 2.0]})
    assert r.json() == {"mse": 0.0, "mae": 0.0}


def test_drift_seed_normal_vs_injected(client: TestClient) -> None:
    normal = client.get("/api/v3/monitoring/drift", params={"drifted": "false"}).json()
    injected = client.get("/api/v3/monitoring/drift", params={"drifted": "true"}).json()
    assert normal["drifted"] is False
    assert injected["drifted"] is True
    assert injected["psi"] > normal["psi"]


def test_drift_custom(client: TestClient) -> None:
    r = client.post("/api/v3/monitoring/drift", json={"reference": [10, 20, 30], "current": [30, 20, 10]})
    assert r.json()["drifted"] is True


def test_outliers_zscore_and_iqr(client: TestClient) -> None:
    z = client.post("/api/v3/monitoring/outliers", json={"values": [10.0] * 30 + [500.0]}).json()
    assert z["method"] == "zscore" and len(z["outliers"]) == 1
    iqr = client.post(
        "/api/v3/monitoring/outliers", params={"method": "iqr"}, json={"values": [1, 2, 3, 4, 5, 100]}
    ).json()
    assert iqr["method"] == "iqr" and len(iqr["outliers"]) == 1


def test_explain_seed_and_ranking(client: TestClient) -> None:
    seed = client.get("/api/v3/monitoring/explain").json()
    assert "features" in seed and seed["backend"] in ("shap", "pure-python-fallback")
    ranked = client.post(
        "/api/v3/monitoring/explain", json={"contributions": {"a": [0.1], "b": [-0.9]}}
    ).json()
    assert ranked["features"][0]["feature"] == "b"
