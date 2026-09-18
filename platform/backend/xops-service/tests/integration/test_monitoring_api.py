"""MLOps 모니터링 API 통합 테스트."""

from __future__ import annotations

from fastapi.testclient import TestClient

# GET 3종의 시드 폴백은 데모 표시 ON 경로다 — 기본값(include_seed=false)은 빈 응답이므로
# 시드 계약을 보는 테스트는 이 파라미터를 명시한다.
_DEMO_ON = {"include_seed": "true"}


def test_metrics_history_has_six_series(client: TestClient) -> None:
    j = client.get("/api/v3/monitoring/metrics", params=_DEMO_ON).json()
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
    normal = client.get("/api/v3/monitoring/drift", params={**_DEMO_ON, "drifted": "false"}).json()
    injected = client.get("/api/v3/monitoring/drift", params={**_DEMO_ON, "drifted": "true"}).json()
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
    seed = client.get("/api/v3/monitoring/explain", params=_DEMO_ON).json()
    assert "features" in seed and seed["backend"] in ("shap", "pure-python-fallback")
    ranked = client.post(
        "/api/v3/monitoring/explain", json={"contributions": {"a": [0.1], "b": [-0.9]}}
    ).json()
    assert ranked["features"][0]["feature"] == "b"


# ── 실측 소스 전환 (데모 OFF 실데이터화) ──────────────────────
def test_metrics_source_is_seed_without_model_id(client: TestClient) -> None:
    """model_id 없이 부르면 시드다 — 그 사실이 응답에 표기돼야 화면이 빈 상태를 고를 수 있다."""
    assert client.get("/api/v3/monitoring/metrics", params=_DEMO_ON).json()["source"] == "seed"


def test_metrics_source_is_seed_for_untrained_model(client: TestClient) -> None:
    """실행 이력이 없는 모델은 실측이 없으므로 시드로 내려가고 measured라고 주장하지 않는다."""
    j = client.get("/api/v3/monitoring/metrics", params={**_DEMO_ON, "model_id": "no-such-model"}).json()
    assert j["source"] == "seed"


# ── 데모 표시 OFF (기본값) ────────────────────────────────────
def test_demo_off_metrics_returns_no_seed_values(client: TestClient) -> None:
    """기본값은 데모 OFF — 실측이 없으면 시드 대신 빈 시계열이고 source 는 null 이다."""
    j = client.get("/api/v3/monitoring/metrics").json()
    assert j == {"history": {}, "latest": {}, "labels": None, "latency_ms": None, "source": None}


def test_demo_off_drift_returns_no_seed_distribution(client: TestClient) -> None:
    """실 추론 입력 수집 경로가 없으므로 데모 OFF에서는 판정을 내지 않는다."""
    j = client.get("/api/v3/monitoring/drift").json()
    assert j["source"] is None
    assert j["reference"] == [] and j["current"] == [] and j["buckets"] == []
    assert "psi" not in j


def test_demo_off_explain_returns_no_seed_features(client: TestClient) -> None:
    j = client.get("/api/v3/monitoring/explain").json()
    assert j["source"] is None and j["features"] == []


def test_metrics_become_measured_after_retrain(client: TestClient, reset_model) -> None:
    """재학습 1회로 실측 지표 추이가 생기고, 학습 시 실측한 추론 지연이 함께 온다."""
    model_id = "settlement-demand"
    reset_model(model_id)
    client.post("/api/v3/orchestration/events", json={"model_id": model_id, "trigger": "manual"})

    j = client.get("/api/v3/monitoring/metrics", params={"model_id": model_id}).json()
    assert j["source"] == "measured"
    assert len(j["labels"]) == len(j["history"]["accuracy"]) >= 1
    for key in ("accuracy", "f1", "precision", "recall", "mse", "mae"):
        assert len(j["history"][key]) == len(j["labels"])
        assert j["latest"][key] == j["history"][key][-1]
    # 지연은 상수가 아니라 학습 시 측정값 — 있으면 유한한 양수여야 한다.
    assert j["latency_ms"] is None or j["latency_ms"] > 0


def test_explain_becomes_measured_after_promotion(client: TestClient, reset_model) -> None:
    """승급된 버전의 아티팩트가 있으면 기여도는 학습된 계수에서 나온다(시드 아님)."""
    model_id = "vital-population"
    reset_model(model_id)
    run = client.post(
        "/api/v3/orchestration/events", json={"model_id": model_id, "trigger": "manual"}
    ).json()

    j = client.get("/api/v3/monitoring/explain", params={"model_id": model_id}).json()
    if run["state"] != "succeeded":  # 승급되지 않았으면 아티팩트가 없어 시드로 남는다
        assert j["source"] == "seed"
        return
    assert j["source"] == "measured"
    values = [abs(f["value"]) for f in j["features"]]
    assert values == sorted(values, reverse=True)  # |기여도| 내림차순


def test_drift_is_labelled_seed(client: TestClient) -> None:
    """PSI 계산은 실계산이지만 입력 분포가 시드라 measured로 표기하지 않는다."""
    assert client.get("/api/v3/monitoring/drift", params=_DEMO_ON).json()["source"] == "seed"
