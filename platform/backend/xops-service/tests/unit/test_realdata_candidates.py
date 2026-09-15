"""R3-2/3/4/5 후보 관리 단위 테스트 — apply 4조건 각각의 실패 케이스, 성공, restore, retrain_needed."""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.core.db import _conn
from src.realdata import candidates

_MODEL_ID = "namwon-nonlocal-visitors-next-month"
# A 실제 아티팩트 형태: features/coef/train_means/train_stds는 모두 같은 순서의 병렬 리스트다
# (dict가 아니다) — B-fix로 스텁도 실제 시그니처에 맞췄다.
_FEATURES = ["y_lag1", "y_lag2"]
_MEANS = [100.0, 90.0]
_STDS = [10.0, 9.0]
_COEF = [1.0, 2.0]
_INTERCEPT = 5.0


class FakeRealdataError(Exception):
    pass


class FakeErrors:
    RealdataError = FakeRealdataError


class FakeSnapshot:
    def load_dataset(self, dataset_id: str) -> dict[str, Any]:
        return {"rows": []}

    def create_dataset(self, target: str, *, spec_version: str = "v1") -> Any:
        return self._fresh

    def __init__(self, fresh: Any = None) -> None:
        self._fresh = fresh


def _manual_predict(row: dict[str, float]) -> float:
    return _INTERCEPT + sum(_COEF[i] * (row[f] - _MEANS[i]) / _STDS[i] for i, f in enumerate(_FEATURES))


class FakeModels:
    def __init__(self, *, predict_offset: float = 0.0) -> None:
        self._predict_offset = predict_offset

    def load_artifact(self, model_id: str, version: str) -> dict[str, Any]:
        return {
            "model_id": model_id,
            "version": version,
            "features": _FEATURES,
            "coef": _COEF,
            "intercept": _INTERCEPT,
            "train_means": _MEANS,
            "train_stds": _STDS,
        }

    def build_feature_rows(self, dataset: Any, *, for_month: int | None = None) -> list[dict[str, float]]:
        return [{"y_lag1": 110.0, "y_lag2": 95.0}, {"y_lag1": 105.0, "y_lag2": 92.0}]

    def predict(self, artifact: dict[str, Any], rows: list[dict[str, float]]) -> list[float]:
        return [_manual_predict(row) + self._predict_offset for row in rows]


def _patch_stubs(monkeypatch: pytest.MonkeyPatch, *, models: FakeModels | None = None, snapshot: FakeSnapshot | None = None) -> None:
    monkeypatch.setattr(candidates, "_import_models", lambda: models or FakeModels())
    monkeypatch.setattr(candidates, "_import_snapshot", lambda: snapshot or FakeSnapshot())
    monkeypatch.setattr(candidates, "_import_errors", lambda: FakeErrors)


@pytest.fixture(autouse=True)
def _clean_tables():
    conn = _conn()
    conn.execute("DELETE FROM rd_datasets")
    conn.execute("DELETE FROM rd_model_candidates")
    conn.execute("DELETE FROM rd_active_models")
    conn.commit()
    yield
    conn.execute("DELETE FROM rd_datasets")
    conn.execute("DELETE FROM rd_model_candidates")
    conn.execute("DELETE FROM rd_active_models")
    conn.commit()


def _insert_dataset(dataset_id: str, quality: dict[str, Any], *, content_hash: str = "hash-1") -> None:
    spec = {"target": "nonlocal_visitors", "spec_version": "v1"}
    _conn().execute(
        "INSERT INTO rd_datasets "
        "(dataset_id, spec_json, quality_json, observed_from, observed_to, row_count, content_hash, file_path, created_at) "
        "VALUES (?, ?, ?, 202301, 202310, 10, ?, '/tmp/x.json', '2026-01-01T00:00:00+00:00')",
        (dataset_id, json.dumps(spec), json.dumps(quality), content_hash),
    )
    _conn().commit()


def _insert_candidate(
    model_id: str,
    version: str,
    dataset_id: str,
    *,
    mae: float,
    baseline_mae: float,
    eval_period: dict[str, Any] | None = None,
    status: str = "candidate",
) -> None:
    metrics = {"mae": mae, "rmse": mae * 1.2, "wape": 0.1, "eval_period": eval_period or {"from": 202307, "to": 202309, "n": 6}}
    baseline = {"name": "yoy", "mae": baseline_mae, "rmse": baseline_mae * 1.2, "wape": 0.2}
    _conn().execute(
        "INSERT INTO rd_model_candidates (model_id, version, dataset_id, artifact_path, metrics_json, baseline_json, status) "
        "VALUES (?, ?, ?, 'artifact-path', ?, ?, ?)",
        (model_id, version, dataset_id, json.dumps(metrics), json.dumps(baseline), status),
    )
    _conn().commit()


_GOOD_QUALITY = {"mapping_matched": 23, "mapping_total": 23, "duplicate_key_count": 0}


def test_apply_rejects_incomplete_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_stubs(monkeypatch)
    _insert_dataset("ds-1", {"mapping_matched": 20, "mapping_total": 23, "duplicate_key_count": 0})
    _insert_candidate(_MODEL_ID, "v1", "ds-1", mae=5.0, baseline_mae=10.0)

    with pytest.raises(candidates.ApplyRejected) as exc:
        candidates.apply(_MODEL_ID, "v1", "tester")
    assert any("매핑" in r for r in exc.value.reasons)


def test_apply_rejects_duplicate_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_stubs(monkeypatch)
    _insert_dataset("ds-1", {"mapping_matched": 23, "mapping_total": 23, "duplicate_key_count": 2})
    _insert_candidate(_MODEL_ID, "v1", "ds-1", mae=5.0, baseline_mae=10.0)

    with pytest.raises(candidates.ApplyRejected) as exc:
        candidates.apply(_MODEL_ID, "v1", "tester")
    assert any("중복" in r for r in exc.value.reasons)


def test_apply_rejects_reload_inconsistency(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_stubs(monkeypatch, models=FakeModels(predict_offset=1.0))
    _insert_dataset("ds-1", _GOOD_QUALITY)
    _insert_candidate(_MODEL_ID, "v1", "ds-1", mae=5.0, baseline_mae=10.0)

    with pytest.raises(candidates.ApplyRejected) as exc:
        candidates.apply(_MODEL_ID, "v1", "tester")
    assert any("재로딩" in r for r in exc.value.reasons)


def test_apply_rejects_when_mae_not_better_than_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_stubs(monkeypatch)
    _insert_dataset("ds-1", _GOOD_QUALITY)
    _insert_candidate(_MODEL_ID, "v1", "ds-1", mae=15.0, baseline_mae=10.0)

    with pytest.raises(candidates.ApplyRejected) as exc:
        candidates.apply(_MODEL_ID, "v1", "tester")
    assert any("MAE" in r for r in exc.value.reasons)


def test_apply_rejects_when_not_better_than_active_same_period(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_stubs(monkeypatch)
    _insert_dataset("ds-1", _GOOD_QUALITY)
    period = {"from": 202307, "to": 202309, "n": 6}
    _insert_candidate(_MODEL_ID, "v-active", "ds-1", mae=4.0, baseline_mae=10.0, eval_period=period, status="applied")
    _conn().execute(
        "INSERT INTO rd_active_models (model_id, version, applied_at, previous_version) VALUES (?, 'v-active', '2026-01-01T00:00:00+00:00', NULL)",
        (_MODEL_ID,),
    )
    _conn().commit()
    _insert_candidate(_MODEL_ID, "v-new", "ds-1", mae=6.0, baseline_mae=10.0, eval_period=period)

    with pytest.raises(candidates.ApplyRejected) as exc:
        candidates.apply(_MODEL_ID, "v-new", "tester")
    assert any("활성 모델 대비" in r for r in exc.value.reasons)


def test_apply_succeeds_and_supersedes_previous_active(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_stubs(monkeypatch)
    _insert_dataset("ds-1", _GOOD_QUALITY)
    period = {"from": 202307, "to": 202309, "n": 6}
    _insert_candidate(_MODEL_ID, "v-active", "ds-1", mae=8.0, baseline_mae=10.0, eval_period=period, status="applied")
    _conn().execute(
        "INSERT INTO rd_active_models (model_id, version, applied_at, previous_version) VALUES (?, 'v-active', '2026-01-01T00:00:00+00:00', NULL)",
        (_MODEL_ID,),
    )
    _conn().commit()
    _insert_candidate(_MODEL_ID, "v-new", "ds-1", mae=5.0, baseline_mae=10.0, eval_period=period)

    active = candidates.apply(_MODEL_ID, "v-new", "tester")

    assert active["version"] == "v-new"
    assert active["previous_version"] == "v-active"
    assert candidates.get_candidate(_MODEL_ID, "v-active")["status"] == "superseded"
    assert candidates.get_candidate(_MODEL_ID, "v-new")["status"] == "applied"


def test_apply_unknown_candidate_raises_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_stubs(monkeypatch)
    with pytest.raises(candidates.CandidateNotFound):
        candidates.apply(_MODEL_ID, "does-not-exist", "tester")


def test_restore_reactivates_superseded_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_stubs(monkeypatch)
    _insert_dataset("ds-1", _GOOD_QUALITY)
    _insert_candidate(_MODEL_ID, "v-old", "ds-1", mae=5.0, baseline_mae=10.0, status="superseded")
    _insert_candidate(_MODEL_ID, "v-current", "ds-1", mae=4.0, baseline_mae=10.0, status="applied")
    _conn().execute(
        "INSERT INTO rd_active_models (model_id, version, applied_at, previous_version) VALUES (?, 'v-current', '2026-01-01T00:00:00+00:00', 'v-old')",
        (_MODEL_ID,),
    )
    _conn().commit()

    active = candidates.restore(_MODEL_ID, "v-old", "tester", note="rollback")

    assert active["version"] == "v-old"
    assert candidates.get_candidate(_MODEL_ID, "v-old")["status"] == "restored"
    assert candidates.get_candidate(_MODEL_ID, "v-current")["status"] == "superseded"


def test_restore_rejects_when_status_is_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_stubs(monkeypatch)
    _insert_dataset("ds-1", _GOOD_QUALITY)
    _insert_candidate(_MODEL_ID, "v1", "ds-1", mae=5.0, baseline_mae=10.0, status="candidate")

    with pytest.raises(candidates.ApplyRejected):
        candidates.restore(_MODEL_ID, "v1", "tester")


@pytest.mark.parametrize("switch_back", ["apply", "restore"])
def test_previously_restored_version_can_be_replaced_and_restored_again(
    monkeypatch: pytest.MonkeyPatch, switch_back: str
) -> None:
    _patch_stubs(monkeypatch)
    _insert_dataset("ds-1", _GOOD_QUALITY)
    _insert_candidate(_MODEL_ID, "v-old", "ds-1", mae=8.0, baseline_mae=10.0)
    _insert_candidate(_MODEL_ID, "v-new", "ds-1", mae=4.0, baseline_mae=10.0)
    candidates.apply(_MODEL_ID, "v-old", "tester")
    candidates.apply(_MODEL_ID, "v-new", "tester")
    candidates.restore(_MODEL_ID, "v-old", "tester")
    getattr(candidates, switch_back)(_MODEL_ID, "v-new", "tester")

    assert candidates.get_candidate(_MODEL_ID, "v-old")["status"] == "superseded"
    active = candidates.restore(_MODEL_ID, "v-old", "tester")
    assert active["version"] == "v-old"
    assert candidates.get_candidate(_MODEL_ID, "v-new")["status"] == "superseded"


def test_restore_accepts_a_previously_restored_historical_version(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_stubs(monkeypatch)
    _insert_dataset("ds-1", _GOOD_QUALITY)
    _insert_candidate(_MODEL_ID, "v-old", "ds-1", mae=8.0, baseline_mae=10.0, status="restored")
    _insert_candidate(_MODEL_ID, "v-new", "ds-1", mae=4.0, baseline_mae=10.0)
    candidates.apply(_MODEL_ID, "v-new", "tester")

    assert candidates.restore(_MODEL_ID, "v-old", "tester")["version"] == "v-old"


def test_retrain_needed_false_without_active_model(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_stubs(monkeypatch)
    assert candidates.retrain_needed(_MODEL_ID) is False


def test_retrain_needed_true_when_content_hash_changed(monkeypatch: pytest.MonkeyPatch) -> None:
    fresh = type("Fresh", (), {"content_hash": "hash-2"})()
    _patch_stubs(monkeypatch, snapshot=FakeSnapshot(fresh=fresh))
    _insert_dataset("ds-1", _GOOD_QUALITY, content_hash="hash-1")
    _insert_candidate(_MODEL_ID, "v-active", "ds-1", mae=5.0, baseline_mae=10.0, status="applied")
    _conn().execute(
        "INSERT INTO rd_active_models (model_id, version, applied_at, previous_version) VALUES (?, 'v-active', '2026-01-01T00:00:00+00:00', NULL)",
        (_MODEL_ID,),
    )
    _conn().commit()

    assert candidates.retrain_needed(_MODEL_ID) is True


def test_retrain_needed_true_when_drift_psi_above_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    fresh = type("Fresh", (), {"content_hash": "hash-1"})()  # 동일 해시 — 신규 데이터 아님
    _patch_stubs(monkeypatch, snapshot=FakeSnapshot(fresh=fresh))
    _insert_dataset("ds-1", _GOOD_QUALITY, content_hash="hash-1")
    _insert_candidate(_MODEL_ID, "v-active", "ds-1", mae=5.0, baseline_mae=10.0, status="applied")
    _conn().execute(
        "INSERT INTO rd_active_models (model_id, version, applied_at, previous_version) VALUES (?, 'v-active', '2026-01-01T00:00:00+00:00', NULL)",
        (_MODEL_ID,),
    )
    _conn().commit()

    from src.realdata import monitoring

    monkeypatch.setattr(
        monitoring,
        "drift",
        lambda model_id, version: {"status": "ok", "data": {"features": [{"feature": "y_lag1", "psi": 0.4}], "target": None}},
    )

    assert candidates.retrain_needed(_MODEL_ID) is True
