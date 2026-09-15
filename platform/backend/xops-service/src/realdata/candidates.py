"""R3-2/3/4/5 후보 관리 — candidate CRUD, apply(4조건), restore, retrain_needed.

apply 조건 ②(아티팩트 재로딩 일관성)의 구현 정의는 이 파일 `_reload_consistency_reasons`의
docstring에 명시했다(완료 보고에도 동일 내용을 옮긴다) — 계약 문면의 "아티팩트 eval의
예측과 일치"는 eval에 개별 예측값이 없어 그대로 적용할 수 없어, 재로딩-재계산 일관성으로
대체했다.
"""

from __future__ import annotations

import json
from types import ModuleType
from typing import Any

from src.core.db import _conn
from src.core.settings import get_settings
from src.realdata.pg_reader import RealdataUnavailable

_RELOAD_TOLERANCE = 1e-6


class ApplyRejected(Exception):
    """apply/restore 4조건 중 하나라도 실패 — API는 400 + reasons로 변환한다."""

    def __init__(self, reasons: list[str]) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = reasons


class CandidateNotFound(Exception):
    """model_id/version에 해당하는 후보가 없음 — API는 404로 변환한다."""


def _import_models() -> ModuleType:
    from src.realdata import models

    return models


def _import_snapshot() -> ModuleType:
    from src.realdata import snapshot

    return snapshot


def _import_errors() -> ModuleType:
    from src.realdata import errors

    return errors


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _as_dict(obj: Any) -> dict[str, Any]:
    """TrainOutcome의 metrics/baseline/eval_period 필드가 dict든 dataclass든 JSON화."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "_asdict"):
        return dict(obj._asdict())
    if hasattr(obj, "__dict__"):
        return dict(vars(obj))
    return {"value": obj}


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """TrainOutcome이 dict/dataclass 어느 쪽이든 같은 방식으로 필드를 읽는다."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


# ── rd_model_candidates CRUD ─────────────────────────────────
def register_candidate(model_id: str, dataset_id: str, outcome: Any) -> str:
    """학습 성공 시 후보 등록(R3-2, status=candidate). eval_period은 metrics_json 안에
    함께 저장한다(apply 조건④가 "같은 평가 구간"을 비교해야 하는데 rd_model_candidates
    테이블에는 별도 eval_period 컬럼이 없어, metrics_json 내부 키로 보존하는 방식을 택했다)."""
    version = _field(outcome, "version")
    artifact_path = str(_field(outcome, "artifact_path", ""))
    metrics = _as_dict(_field(outcome, "metrics"))
    metrics = {**metrics, "eval_period": _as_dict(_field(outcome, "eval_period"))}
    baseline = _as_dict(_field(outcome, "baseline"))

    _conn().execute(
        "INSERT INTO rd_model_candidates "
        "(model_id, version, dataset_id, artifact_path, metrics_json, baseline_json, status) "
        "VALUES (?, ?, ?, ?, ?, ?, 'candidate') "
        "ON CONFLICT(model_id, version) DO UPDATE SET "
        "dataset_id = excluded.dataset_id, artifact_path = excluded.artifact_path, "
        "metrics_json = excluded.metrics_json, baseline_json = excluded.baseline_json",
        (
            model_id,
            version,
            dataset_id,
            artifact_path,
            json.dumps(metrics, ensure_ascii=False),
            json.dumps(baseline, ensure_ascii=False),
        ),
    )
    _conn().commit()
    return version


def _row_to_dict(row: Any) -> dict[str, Any]:
    data = dict(row)
    data["metrics"] = json.loads(data.pop("metrics_json"))
    data["baseline"] = json.loads(data.pop("baseline_json"))
    return data


def list_candidates(model_id: str) -> list[dict[str, Any]]:
    rows = _conn().execute(
        "SELECT * FROM rd_model_candidates WHERE model_id = ? ORDER BY version DESC", (model_id,)
    ).fetchall()
    return [_row_to_dict(row) for row in rows]


def get_candidate(model_id: str, version: str) -> dict[str, Any] | None:
    row = _conn().execute(
        "SELECT * FROM rd_model_candidates WHERE model_id = ? AND version = ?", (model_id, version)
    ).fetchone()
    return _row_to_dict(row) if row else None


def get_active(model_id: str) -> dict[str, Any] | None:
    row = _conn().execute("SELECT * FROM rd_active_models WHERE model_id = ?", (model_id,)).fetchone()
    return dict(row) if row else None


def _set_candidate_status(model_id: str, version: str, status: str, decided_by: str, note: str | None = None) -> None:
    _conn().execute(
        "UPDATE rd_model_candidates SET status = ?, decided_at = ?, decided_by = ?, note = COALESCE(?, note) "
        "WHERE model_id = ? AND version = ?",
        (status, _now_iso(), decided_by, note, model_id, version),
    )


def _upsert_active(model_id: str, version: str, previous_version: str | None) -> None:
    _conn().execute(
        "INSERT INTO rd_active_models (model_id, version, applied_at, previous_version) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(model_id) DO UPDATE SET "
        "version = excluded.version, applied_at = excluded.applied_at, previous_version = excluded.previous_version",
        (model_id, version, _now_iso(), previous_version),
    )


# ── apply 조건 ①~④ ──────────────────────────────────────────
# A(snapshot.py `_quality`)의 실제 quality_json 키 — B-fix에서 실 구현과 대조해 확정.
_REQUIRED_QUALITY_KEYS = ("mapping_matched", "mapping_total", "duplicate_key_count")


def _quality_reasons(dataset_id: str) -> list[str]:
    """조건①: R1-7 품질(매핑 23/23·중복 0). rd_datasets.quality_json은 작성자 A
    `snapshot._quality()`가 채운다 — 실제 키는 `mapping_matched`/`mapping_total`/
    `duplicate_key_count`(B-fix로 `duplicate_keys` 가정을 실제 키명으로 교정). 키가
    아예 없으면(스키마 불일치·구버전 스냅샷) 조건을 무조건 통과시키지 않고 명시적으로
    거부한다."""
    row = _conn().execute("SELECT quality_json FROM rd_datasets WHERE dataset_id = ?", (dataset_id,)).fetchone()
    if row is None:
        return [f"데이터셋 품질 정보를 찾을 수 없습니다: {dataset_id}"]
    quality = json.loads(row["quality_json"])

    missing = [key for key in _REQUIRED_QUALITY_KEYS if key not in quality]
    if missing:
        return [f"quality 필드 없음: {', '.join(missing)}"]

    reasons: list[str] = []
    matched, total = quality["mapping_matched"], quality["mapping_total"]
    if matched != total:
        reasons.append(f"행정동 매핑이 완전하지 않습니다: {matched}/{total}")

    duplicates = quality["duplicate_key_count"]
    if duplicates != 0:
        reasons.append(f"중복 키가 있습니다: {duplicates}")

    return reasons


def _reload_consistency_reasons(candidate: dict[str, Any]) -> list[str]:
    """조건②(구현 정의): 계약 문면은 "load_artifact+predict 재계산이 아티팩트 eval의 예측과
    일치"라고 하지만, A 인터페이스의 아티팩트 JSON에는 `eval`이 지표(metrics/baseline)만 담고
    개별 표본 예측값을 담지 않는다(계약 본문도 "예측값이 없으면 학습행 3개로 재계산 후
    일관성만 확인"으로 대비책을 명시). 그래서 이 구현은 조건②를 다음으로 정의한다:

      "load_artifact()로 다시 읽은 아티팩트에 대해, 그 데이터셋에서 build_feature_rows()로
      만든 학습행 중 최대 3개를 뽑아 (a) predict(artifact, rows) 배치 결과와 (b) 같은
      아티팩트의 coef/intercept/train_means/train_stds로 이 함수가 직접 계산한 선형식
      예측값이 1e-6 이내로 일치하는가."

    즉 "아티팩트가 재로딩 후에도 자기 자신의 계수로 정확히 재현되는가"를 검증한다.

    B-fix: A `models.predict(artifact, feature_rows)`는 단일 행이 아니라 행 리스트를 받아
    리스트를 반환하는 배치 함수이고, `build_feature_rows(dataset, *, for_month=None)`는
    B가 가정했던 `features` 인자를 받지 않는다(항상 전체 피처 컬럼을 만든다). 또한
    `train_means`/`train_stds`는 dict가 아니라 `features`와 같은 순서의 리스트다 — 전부
    실제 시그니처에 맞춰 고쳤다."""
    models_mod = _import_models()
    snapshot_mod = _import_snapshot()
    errors_mod = _import_errors()

    try:
        artifact = models_mod.load_artifact(candidate["model_id"], candidate["version"])
    except errors_mod.RealdataError as exc:
        return [f"아티팩트 재로딩 실패: {exc}"]

    try:
        dataset = snapshot_mod.load_dataset(candidate["dataset_id"])
        rows = models_mod.build_feature_rows(dataset)[:3]
    except errors_mod.RealdataError as exc:
        return [f"재계산용 학습행을 만들 수 없습니다: {exc}"]

    if not rows:
        return ["재계산에 사용할 학습행이 없습니다."]

    features = artifact["features"]
    coef = artifact["coef"]
    intercept = artifact["intercept"]
    means = artifact["train_means"]
    stds = artifact["train_stds"]

    predicted_batch = models_mod.predict(artifact, rows)
    for row, predicted in zip(rows, predicted_batch):
        manual = intercept + sum(
            coef[i] * (row[feature] - means[i]) / (stds[i] or 1e-9)
            for i, feature in enumerate(features)
        )
        if abs(predicted - manual) > _RELOAD_TOLERANCE:
            return [f"재로딩 예측이 재계산과 불일치합니다: predict={predicted} manual={manual}"]

    return []


def _metrics_reasons(candidate: dict[str, Any]) -> list[str]:
    """조건③: model MAE < 기준선 MAE."""
    mae = candidate["metrics"].get("mae")
    baseline_mae = candidate["baseline"].get("mae")
    if mae is None or baseline_mae is None or not (mae < baseline_mae):
        return [f"MAE가 기준선보다 개선되지 않았습니다: model={mae} baseline={baseline_mae}"]
    return []


def _active_comparison_reasons(model_id: str, candidate: dict[str, Any]) -> list[str]:
    """조건④: 활성 모델이 있으면 같은 평가 구간에서 MAE 개선. 평가 구간이 다르면(예: 신규
    데이터로 관측 종료월이 밀린 경우) 직접 비교가 성립하지 않아 이 조건은 건너뛴다(통과) —
    완료 보고에 명시한 해석."""
    active = get_active(model_id)
    if active is None:
        return []
    active_candidate = get_candidate(model_id, active["version"])
    if active_candidate is None:
        return []

    same_period = active_candidate["metrics"].get("eval_period") == candidate["metrics"].get("eval_period")
    if not same_period:
        return []

    active_mae = active_candidate["metrics"].get("mae")
    new_mae = candidate["metrics"].get("mae")
    if active_mae is None or new_mae is None or not (new_mae < active_mae):
        return [f"활성 모델 대비 MAE가 개선되지 않았습니다: new={new_mae} active={active_mae}"]
    return []


def apply(model_id: str, version: str, decided_by: str) -> dict[str, Any]:
    """`POST /realdata/models/{model_id}/candidates/{version}/apply`(R3-3)."""
    candidate = get_candidate(model_id, version)
    if candidate is None:
        raise CandidateNotFound(f"후보를 찾을 수 없습니다: {model_id}/{version}")

    reasons: list[str] = []
    reasons += _quality_reasons(candidate["dataset_id"])
    reasons += _reload_consistency_reasons(candidate)
    reasons += _metrics_reasons(candidate)
    reasons += _active_comparison_reasons(model_id, candidate)
    if reasons:
        raise ApplyRejected(reasons)

    previous = get_active(model_id)
    if previous is not None and previous["version"] != version:
        _conn().execute(
            "UPDATE rd_model_candidates SET status = 'superseded', decided_at = ?, decided_by = ? "
            "WHERE model_id = ? AND version = ? AND status IN ('applied', 'restored')",
            (_now_iso(), decided_by, model_id, previous["version"]),
        )

    _upsert_active(model_id, version, previous["version"] if previous else None)
    _set_candidate_status(model_id, version, "applied", decided_by)
    _conn().commit()
    return get_active(model_id)  # type: ignore[return-value]


def restore(model_id: str, version: str, decided_by: str, note: str | None = None) -> dict[str, Any]:
    """R3-4 — 한 번 이상 반영된 버전은 이전 복원 이력이 있어도 다시 복원할 수 있다."""
    candidate = get_candidate(model_id, version)
    if candidate is None:
        raise CandidateNotFound(f"후보를 찾을 수 없습니다: {model_id}/{version}")
    if candidate["status"] not in ("applied", "superseded", "restored"):
        raise ApplyRejected([f"복원 가능한 상태가 아닙니다(status={candidate['status']})"])

    previous = get_active(model_id)
    if previous is not None and previous["version"] != version:
        _conn().execute(
            "UPDATE rd_model_candidates SET status = 'superseded', decided_at = ?, decided_by = ? "
            "WHERE model_id = ? AND version = ? AND status IN ('applied', 'restored')",
            (_now_iso(), decided_by, model_id, previous["version"]),
        )

    _upsert_active(model_id, version, previous["version"] if previous else None)
    _set_candidate_status(model_id, version, "restored", decided_by, note=note)
    _conn().commit()
    return get_active(model_id)  # type: ignore[return-value]


def _content_hash_changed(dataset_id: str) -> bool:
    """B-fix: A `snapshot.create_dataset(target, *, spec_version="v1")`는 spec dict 전체가
    아니라 `target` 문자열을 받는다 — 원래 코드는 `create_dataset(spec)`을 그대로 넘겨
    실제 A 모듈에서 TypeError가 났다(`target not in TARGETS`류 검증 이전에 인자 개수부터
    불일치). 저장된 spec_json에서 target/spec_version만 뽑아 실제 시그니처로 재호출한다."""
    row = _conn().execute(
        "SELECT spec_json, content_hash FROM rd_datasets WHERE dataset_id = ?", (dataset_id,)
    ).fetchone()
    if row is None:
        return False
    spec = json.loads(row["spec_json"])
    target = spec.get("target")
    spec_version = spec.get("spec_version", "v1")
    if not target:
        return False
    snapshot_mod = _import_snapshot()
    errors_mod = _import_errors()
    try:
        fresh = snapshot_mod.create_dataset(target, spec_version=spec_version)
    except (errors_mod.RealdataError, ValueError, RealdataUnavailable):
        # RealdataUnavailable(pg_reader, RuntimeError 계열)은 errors.RealdataError와 별개
        # 계층이다 — PG 장애로 재학습 필요 여부를 못 구했다고 GET /models 전체가 500이
        # 되면 안 되므로(다른 필드는 정상 응답 가능) false로 완화한다.
        return False
    fresh_hash = _field(fresh, "content_hash")
    return bool(fresh_hash) and fresh_hash != row["content_hash"]


def retrain_needed(model_id: str) -> bool:
    """R3-5: 활성 모델의 dataset spec으로 재생성한 스냅샷의 content_hash가 다르면(신규 데이터)
    또는 드리프트 PSI≥0.2면 true. 자동 학습은 실행하지 않는다(표시만)."""
    active = get_active(model_id)
    if active is None:
        return False
    candidate = get_candidate(model_id, active["version"])
    if candidate is None:
        return False

    if _content_hash_changed(candidate["dataset_id"]):
        return True

    from src.realdata import monitoring

    result = monitoring.drift(model_id, active["version"])
    if result.get("status") != "ok":
        return False
    data = result.get("data") or {}
    threshold = get_settings().psi_threshold
    for item in data.get("features", []):
        if (item.get("psi") or 0.0) >= threshold:
            return True
    target = data.get("target")
    if target and (target.get("psi") or 0.0) >= threshold:
        return True
    return False
