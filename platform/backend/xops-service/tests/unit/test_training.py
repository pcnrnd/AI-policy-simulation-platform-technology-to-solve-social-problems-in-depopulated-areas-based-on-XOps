"""인프로세스 학습 단위 테스트 — 데이터셋·릿지 회귀·그리드 트레이너."""

from __future__ import annotations

import json

import pytest

from src.mlops.orchestration.evaluator import ranking_key
from src.mlops.training.dataset import build_dataset
from src.mlops.training.model import MeanRegressor, RidgeRegressor, solve_linear_system
from src.mlops.training.trainer import Trainer, artifact_path

# 감소 추세가 뚜렷한 합성 지역 2곳 — 시드 형식과 동일한 최소 입력.
_REGIONS = [
    {"history": [1000, 990, 979, 967, 954, 940, 925], "riskIndex": 0.2, "birthRate": 0.8, "agingIndex": 33.0},
    {"history": [500, 497, 493, 488, 482, 475, 467], "riskIndex": 0.1, "birthRate": 0.7, "agingIndex": 43.0},
]


def _dataset(horizon: int = 1, max_lag: int = 3):
    return build_dataset(_REGIONS, horizon=horizon, max_lag=max_lag, decline_threshold_pct=1.2)


# ── 데이터셋 ──
def test_dataset_shape_is_deterministic() -> None:
    first, second = _dataset(), _dataset()
    # 표본 수 = 지역수 × (N - horizon - max_lag + 1) = 2 × (7-1-3+1)
    assert len(first) == 8
    assert first.rows == second.rows and first.targets == second.targets
    # 관측 창 3개 → 기간별 변화율 2개(오래된 rate_t-2, 최신 rate_t-1) + 정적 지표 3개
    assert first.feature_names == ["rate_t-2", "rate_t-1", "riskIndex", "birthRate", "agingIndex"]


def test_dataset_targets_are_relative_change_and_labels_split() -> None:
    ds = _dataset()
    # 감소 추세이므로 타깃(상대 변화율 %)은 모두 음수여야 한다.
    assert all(target < 0 for target in ds.targets)
    # 라벨이 한쪽으로 쏠리지 않아 분류 지표가 의미를 가진다.
    assert 0 < sum(ds.labels) < len(ds.labels)


def test_dataset_horizon_increases_decline_and_shrinks_samples() -> None:
    near, far = _dataset(horizon=1), _dataset(horizon=3)
    assert len(far) < len(near)
    assert min(far.targets) < min(near.targets)  # 지평이 길면 누적 감소폭이 크다


def test_select_keeps_sample_count_and_drops_oldest_lag() -> None:
    ds = _dataset()
    names_3, rows_3 = ds.select(3)
    names_2, rows_2 = ds.select(2)
    # 표본 집합은 고정 — lag만 줄어 지표를 서로 비교할 수 있다.
    assert len(rows_2) == len(rows_3) == len(ds)
    # lag=2는 최신 변화율 하나만 남긴다(가장 오래된 rate_t-2를 버림).
    assert names_2 == ["rate_t-1", "riskIndex", "birthRate", "agingIndex"]
    assert rows_2[0] == rows_3[0][1:]


def test_select_rejects_out_of_range_lag() -> None:
    ds = _dataset()
    with pytest.raises(ValueError):
        ds.select(4)
    with pytest.raises(ValueError):
        ds.select(1)


def test_build_dataset_validates_inputs() -> None:
    with pytest.raises(ValueError):
        build_dataset(_REGIONS, horizon=0, max_lag=3, decline_threshold_pct=1.2)
    with pytest.raises(ValueError):
        build_dataset(_REGIONS, horizon=1, max_lag=1, decline_threshold_pct=1.2)


def test_build_dataset_raises_when_series_too_short() -> None:
    short = [{"history": [10, 9], "riskIndex": 0.1, "birthRate": 0.7, "agingIndex": 40.0}]
    with pytest.raises(ValueError):
        build_dataset(short, horizon=1, max_lag=3, decline_threshold_pct=1.2)


# ── 시드 손상 입력 검증 (신뢰 경계) ──
# mock_data.json은 컨테이너에 읽기 전용 마운트되는 외부 입력이다. 손상된 값이
# ZeroDivisionError/TypeError로 새면 POST /orchestration/events가 500이 된다.
# 전부 ValueError로 명시 실패해야 오케스트레이터의 안전망이 받아 fallback으로 흐른다.
def _corrupt(**overrides: object) -> list[dict[str, object]]:
    region = {"history": [1000, 990, 979, 967, 954], "riskIndex": 0.2, "birthRate": 0.8, "agingIndex": 33.0}
    region.update(overrides)
    return [region]


def test_build_dataset_rejects_zero_or_negative_in_history() -> None:
    # 0은 변화율 계산의 분모라 ZeroDivisionError를 유발했다.
    with pytest.raises(ValueError, match="0보다 커야"):
        build_dataset(_corrupt(history=[1000, 0, 979, 967, 954]), horizon=1, max_lag=3, decline_threshold_pct=1.2)
    with pytest.raises(ValueError, match="0보다 커야"):
        build_dataset(_corrupt(history=[1000, -5, 979, 967, 954]), horizon=1, max_lag=3, decline_threshold_pct=1.2)


def test_build_dataset_rejects_null_static_metric() -> None:
    # null은 float(None) → TypeError를 유발했다.
    with pytest.raises(ValueError, match="riskIndex가 숫자가 아닙니다"):
        build_dataset(_corrupt(riskIndex=None), horizon=1, max_lag=3, decline_threshold_pct=1.2)


def test_build_dataset_rejects_non_numeric_and_non_finite_values() -> None:
    with pytest.raises(ValueError, match="숫자가 아닙니다"):
        build_dataset(_corrupt(history=[1000, "990", 979, 967, 954]), horizon=1, max_lag=3, decline_threshold_pct=1.2)
    with pytest.raises(ValueError, match="유한한 수가 아닙니다"):
        build_dataset(
            _corrupt(history=[1000, float("nan"), 979, 967, 954]), horizon=1, max_lag=3, decline_threshold_pct=1.2
        )


def test_build_dataset_rejects_missing_fields_and_bad_shapes() -> None:
    without_static = {"history": [1000, 990, 979, 967, 954], "birthRate": 0.8, "agingIndex": 33.0}
    with pytest.raises(ValueError, match="필수 정적 지표 riskIndex"):
        build_dataset([without_static], horizon=1, max_lag=3, decline_threshold_pct=1.2)
    with pytest.raises(ValueError, match="history가 배열이 아니거나"):
        build_dataset(_corrupt(history=None), horizon=1, max_lag=3, decline_threshold_pct=1.2)
    with pytest.raises(ValueError, match="객체가 아닙니다"):
        build_dataset(["not-a-region"], horizon=1, max_lag=3, decline_threshold_pct=1.2)


# ── 선형 대수 ──
def test_solve_linear_system_matches_known_solution() -> None:
    solution = solve_linear_system([[2.0, 1.0], [1.0, 3.0]], [5.0, 10.0])
    assert solution[0] == pytest.approx(1.0, abs=1e-9)
    assert solution[1] == pytest.approx(3.0, abs=1e-9)


def test_solve_linear_system_rejects_singular_matrix() -> None:
    with pytest.raises(ValueError):
        solve_linear_system([[1.0, 2.0], [2.0, 4.0]], [3.0, 6.0])


# ── 릿지 회귀 ──
def test_ridge_recovers_linear_relation_without_regularization() -> None:
    rows = [[1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0], [5.0, 0.0]]
    targets = [3.0, 7.0, 7.0, 11.0, 11.0]  # y = 2*x1 + 2*x2 + 1
    model = RidgeRegressor.fit(rows, targets, ridge_lambda=0.0, feature_names=["x1", "x2"])
    for row, expected in zip(rows, targets):
        assert model.predict_one(row) == pytest.approx(expected, abs=1e-6)


def test_ridge_regularization_shrinks_coefficients() -> None:
    rows = [[1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0], [5.0, 0.0]]
    targets = [3.0, 7.0, 7.0, 11.0, 11.0]
    weak = RidgeRegressor.fit(rows, targets, ridge_lambda=0.01, feature_names=["x1", "x2"])
    strong = RidgeRegressor.fit(rows, targets, ridge_lambda=100.0, feature_names=["x1", "x2"])
    assert sum(abs(c) for c in strong.coefficients) < sum(abs(c) for c in weak.coefficients)


def test_ridge_constant_feature_without_regularization_is_singular() -> None:
    rows = [[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]]
    with pytest.raises(ValueError):
        RidgeRegressor.fit(rows, [1.0, 2.0, 3.0], ridge_lambda=0.0, feature_names=["x", "const"])


def test_ridge_constant_feature_survives_with_regularization() -> None:
    rows = [[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]]
    model = RidgeRegressor.fit(rows, [1.0, 2.0, 3.0], ridge_lambda=1.0, feature_names=["x", "const"])
    assert model.predict_one([2.0, 5.0]) == pytest.approx(2.0, abs=0.5)


def test_ridge_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        RidgeRegressor.fit([[1.0]], [1.0, 2.0], ridge_lambda=0.1, feature_names=["x"])
    with pytest.raises(ValueError):
        RidgeRegressor.fit([], [], ridge_lambda=0.1, feature_names=["x"])


def test_ridge_to_dict_has_all_artifact_fields() -> None:
    ds = _dataset()
    names, rows = ds.select(3)
    model = RidgeRegressor.fit(rows, ds.targets, ridge_lambda=1.0, feature_names=names)
    payload = model.to_dict()
    assert payload["kind"] == "ridge-regression"
    assert payload["feature_names"] == names
    # 예측을 재현하려면 계수·절편·표준화 통계가 모두 있어야 한다.
    assert len(payload["coefficients"]) == len(payload["means"]) == len(payload["stds"]) == len(names)
    assert payload["ridge_lambda"] == 1.0
    assert payload["intercept"] == pytest.approx(model.intercept, abs=1e-8)


def test_mean_regressor_predicts_target_mean() -> None:
    model = MeanRegressor.fit([1.0, 2.0, 6.0])
    assert model.predict_one([0.0]) == pytest.approx(3.0, abs=1e-9)
    with pytest.raises(ValueError):
        MeanRegressor.fit([])


# ── 승급 우선순위 정렬 키 ──
def test_ranking_key_prefers_higher_f1_then_falls_through() -> None:
    better_f1 = {"f1": 0.9, "accuracy": 0.1, "mae": 9.0, "mse": 9.0}
    worse_f1 = {"f1": 0.8, "accuracy": 0.99, "mae": 0.1, "mse": 0.1}
    assert ranking_key(better_f1) > ranking_key(worse_f1)
    # f1 동점이면 accuracy로 넘어간다.
    tie_low = {"f1": 0.9, "accuracy": 0.5, "mae": 1.0, "mse": 1.0}
    tie_high = {"f1": 0.9, "accuracy": 0.7, "mae": 1.0, "mse": 1.0}
    assert ranking_key(tie_high) > ranking_key(tie_low)
    # 오차 지표는 낮을수록 우수하게 부호가 뒤집힌다.
    assert ranking_key({"mae": 0.1}) > ranking_key({"mae": 0.5})


# ── 트레이너 ──
def test_trainer_produces_measured_metrics_and_artifact() -> None:
    result = Trainer().train("population-forecast", horizon=1, version="vUNIT")
    assert set(result.candidate_metrics) == {"mse", "mae", "accuracy", "precision", "recall", "f1"}
    # 그리드 전체(lag 2종 × lambda 3종)를 평가했다.
    assert len(result.candidates) == 6
    assert result.dataset["rows"] > 0 and result.dataset["protocol"] == "leave-one-out"
    assert result.latency_ms > 0.0
    # 최고 후보는 그리드 안의 어떤 후보보다 나쁘지 않다.
    assert ranking_key(result.candidate_metrics) == max(ranking_key(c["metrics"]) for c in result.candidates)
    # 아티팩트 파일이 실제로 기록되고 다시 읽힌다.
    path = artifact_path("population-forecast", "vUNIT")
    assert path.is_file()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["metrics"] == result.candidate_metrics
    assert loaded["model"]["kind"] == "ridge-regression"


def test_trainer_beats_intercept_only_baseline_on_seed() -> None:
    result = Trainer().train("population-forecast", horizon=1, version="vUNIT2")
    # 학습된 후보가 절편-only 기준선보다 우수해야 최초 승급이 성립한다.
    assert ranking_key(result.candidate_metrics) > ranking_key(result.baseline_metrics)


def test_trainer_is_deterministic() -> None:
    first = Trainer().train("population-forecast", horizon=1, version="vUNIT3")
    second = Trainer().train("population-forecast", horizon=1, version="vUNIT3")
    assert first.candidate_metrics == second.candidate_metrics
    assert first.best == second.best


def test_longer_horizon_is_a_harder_problem() -> None:
    near = Trainer().train("population-forecast", horizon=1, version="vH1")
    far = Trainer().train("settlement-demand", horizon=3, version="vH3")
    # 지평이 길수록 회귀 오차가 커진다(문제가 실제로 어려워진다).
    assert far.candidate_metrics["mae"] > near.candidate_metrics["mae"]


def test_artifact_path_is_scoped_by_model_and_version() -> None:
    path = artifact_path("population-forecast", "v9.9")
    assert path.name == "v9.9.json"
    assert path.parent.name == "population-forecast"
