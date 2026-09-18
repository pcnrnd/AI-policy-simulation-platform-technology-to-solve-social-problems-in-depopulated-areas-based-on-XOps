"""실데이터(rd_*) → 기존 모니터링·오케스트레이션 GET 스키마 어댑터.

데모 표시 OFF에서 모니터·오케스트레이터 화면이 **기존 UI 그대로** 실데이터를 읽게 하려고,
`/realdata/*` 라우터가 이미 계산한 값을 `/monitoring/*`·`/orchestration/*` 응답 모양으로
옮겨 담는다. 계산 로직은 여기서 새로 만들지 않는다 — 평가·드리프트·설명은 전부
`src.realdata.monitoring`, 학습 job은 `src.realdata.jobs`가 원본이다.

응답의 `source`/`metrics_source`는 항상 `"realdata"`다. 시드(`"seed"`)·시드 학습 실측
(`"measured"`)과 섞이지 않게 별도 표기를 쓴다 — 화면이 이 표기로 회귀 지표 라벨을 고른다.

인증: 호출부(라우터)가 `optional_auth("data:read")`로 토큰을 확인한 뒤에만 이 모듈을
부른다. 토큰 없는 공개 GET에는 실데이터가 실리지 않는다.
"""

from __future__ import annotations

from typing import Any

from src.core.logger import get_logger
from src.mlops.monitoring.drift import DriftDetector
from src.realdata import candidates as candidates_mod
from src.realdata import jobs as jobs_mod
from src.realdata import monitoring as rd_monitoring

_logger = get_logger("xops.monitoring.realdata")
_drift = DriftDetector()

# 화면 표시명 — 프런트 constants/models.js의 REALDATA_MODEL_LABELS와 같은 문구를 쓴다.
MODEL_NAMES: dict[str, str] = {
    "namwon-nonlocal-visitors-next-month": "남원 타지역 방문객(익월)",
    "namwon-observed-sales-next-month": "남원 관측 소비(익월)",
}

# 파이프라인 카탈로그 1행씩 노출할 때 쓰는 이름.
PIPELINE_NAMES: dict[str, str] = {
    "namwon-nonlocal-visitors-next-month": "남원 방문객 재학습(실데이터)",
    "namwon-observed-sales-next-month": "남원 소비 재학습(실데이터)",
}

SOURCE = "realdata"
# 회귀 지표 계열 — 6대 분류 지표(accuracy·f1…) 자리에 이 계열이 들어간다.
SERIES = ("mae", "rmse", "wape", "baseline_mae")
# 드리프트 대표 피처. 타깃의 1개월 랙이라 실제 분포 변화를 드러낸다 — month_sin 같은 달력
# 파생 피처는 PSI가 구조적으로 커서 대표값으로 쓰면 계절성을 드리프트로 읽게 만든다.
_REPRESENTATIVE_FEATURE = "y_lag1"
_PIPELINE_PREFIX = "realdata-"
_JOB_STATE_MAP = {"saved": "succeeded"}


def is_realdata_model(model_id: str | None) -> bool:
    return model_id in MODEL_NAMES


def _candidates_oldest_first(model_id: str) -> list[dict[str, Any]]:
    """후보를 버전 오름차순으로 — 지표 추이 x축이 학습 순서를 따라야 한다."""
    return sorted(candidates_mod.list_candidates(model_id), key=lambda c: c["version"])


def resolve_version(model_id: str) -> str | None:
    """표시 기준 버전 — 반영된 활성 버전이 있으면 그것, 없으면 최신 후보."""
    active = candidates_mod.get_active(model_id)
    if active:
        return active["version"]
    rows = _candidates_oldest_first(model_id)
    return rows[-1]["version"] if rows else None


# ── /monitoring/metrics ─────────────────────────────────────
def metrics(model_id: str) -> dict[str, Any] | None:
    """후보 버전별 검증 지표 추이(MAE·RMSE·WAPE + 기준선 MAE). 후보가 없으면 None."""
    rows = _candidates_oldest_first(model_id)
    if not rows:
        return None
    history = {
        "mae": [row["metrics"].get("mae") for row in rows],
        "rmse": [row["metrics"].get("rmse") for row in rows],
        "wape": [row["metrics"].get("wape") for row in rows],
        "baseline_mae": [row["baseline"].get("mae") for row in rows],
    }
    latest_row = rows[-1]
    return {
        "history": history,
        "latest": {key: values[-1] for key, values in history.items()},
        "labels": [row["version"] for row in rows],
        # 실데이터 학습은 추론 지연을 측정하지 않는다 — 없는 값을 0으로 메우지 않는다.
        "latency_ms": None,
        "source": SOURCE,
        "model_id": model_id,
        "version": resolve_version(model_id),
        "metrics_kind": "regression",
        "baseline": latest_row["baseline"],
        "eval_period": latest_row["metrics"].get("eval_period"),
    }


# ── /monitoring/drift ───────────────────────────────────────
def _representative(features: list[dict[str, Any]]) -> dict[str, Any] | None:
    for feature in features:
        if feature.get("feature") == _REPRESENTATIVE_FEATURE and feature.get("edges"):
            return feature
    usable = [f for f in features if f.get("edges")]
    return max(usable, key=lambda f: f["psi"]) if usable else None


def _bucket_labels(edges: list[float]) -> list[str]:
    return [f"{edges[i]:.4g}~{edges[i + 1]:.4g}" for i in range(len(edges) - 1)]


def drift(model_id: str) -> dict[str, Any] | None:
    """대표 피처의 참조/현재 분포 + PSI·KL 판정. 비교 구간이 없으면 None."""
    version = resolve_version(model_id)
    if version is None:
        return None
    result = rd_monitoring.drift(model_id, version)
    data = result.get("data") or {}
    if result.get("status") != "ok" or data.get("status") != "ok":
        return None
    feature = _representative(data.get("features") or [])
    if feature is None:
        return None
    # PSI/KL 판정은 시드 경로와 같은 DriftDetector로 낸다 — 표의 PSI(_psi_for_field)와
    # 같은 도수를 넣으므로 두 값이 어긋나지 않는다.
    detected = _drift.detect(feature["reference_counts"], feature["current_counts"])
    return {
        "psi": detected.psi,
        "kl": detected.kl,
        "psi_threshold": detected.psi_threshold,
        "kl_threshold": detected.kl_threshold,
        "drifted": detected.drifted,
        "buckets": _bucket_labels(feature["edges"]),
        "reference": feature["reference_counts"],
        "current": feature["current_counts"],
        "source": SOURCE,
        "model_id": model_id,
        "version": version,
        "feature": feature["feature"],
        "kind": data.get("kind"),
        "retrain": None,
    }


# ── /monitoring/explain ─────────────────────────────────────
def _default_explain_key(model_id: str, version: str) -> tuple[int, str] | None:
    """설명 기본 좌표(기준월·행정동) — 검증 구간의 마지막 달과 그 달 첫 행정동(코드순)."""
    candidate = candidates_mod.get_candidate(model_id, version)
    if candidate is None:
        return None
    base_ym = (candidate["metrics"].get("eval_period") or {}).get("to")
    if base_ym is None:
        return None
    # A 모듈(models/snapshot/errors) 접근은 `rd_monitoring`의 지연 import 지점을 그대로 쓴다.
    # 여기서 따로 import하면 간접화 지점이 둘이 되어, 한쪽만 대체된 상태로 도는 경로가 생긴다.
    models = rd_monitoring._import_models()
    snapshot = rd_monitoring._import_snapshot()
    errors = rd_monitoring._import_errors()

    try:
        dataset = snapshot.load_dataset(candidate["dataset_id"])
    except errors.RealdataError as exc:
        _logger.warning(f"realdata explain dataset load failed model={model_id} reason={exc}")
        return None
    codes = sorted(
        row["dong_code"]
        for row in models.build_feature_rows(dataset, for_month=base_ym)
        if row.get("dong_code")
    )
    return (base_ym, codes[0]) if codes else None


def explain(model_id: str) -> dict[str, Any] | None:
    """선형 SHAP 기여도(phi)를 기존 특징 중요도 스키마로. 좌표는 기본값을 쓴다."""
    version = resolve_version(model_id)
    if version is None:
        return None
    key = _default_explain_key(model_id, version)
    if key is None:
        return None
    base_ym, dong_code = key
    result = rd_monitoring.explain(model_id, version, base_ym, dong_code)
    if result.get("status") != "ok":
        return None
    data = result["data"]
    features = sorted(
        ({"feature": c["feature"], "value": c["phi"]} for c in data["contributions"]),
        key=lambda row: abs(row["value"]),
        reverse=True,
    )
    return {
        "features": features,
        "source": SOURCE,
        "model_id": model_id,
        "version": version,
        "base_ym": base_ym,
        "dong_code": dong_code,
        "basis": f"선형 SHAP 기여도(phi) · 기준월 {base_ym} · 행정동 {dong_code}",
        "prediction": data["prediction"],
        "base_value": data["base_value"],
        "note": data["note"],
    }


# ── /orchestration/models ───────────────────────────────────
def models() -> list[dict[str, Any]]:
    """실데이터 모델 목록 — 대상 모델 select가 이름·표시 버전을 여기서 읽는다."""
    out: list[dict[str, Any]] = []
    for model_id, name in MODEL_NAMES.items():
        version = resolve_version(model_id)
        if version is None:
            continue
        candidate = candidates_mod.get_candidate(model_id, version)
        out.append(
            {
                "model_id": model_id,
                "name": name,
                "version": version,
                "next_version": None,
                "metrics": {k: candidate["metrics"].get(k) for k in ("mae", "rmse", "wape")} if candidate else {},
                "metrics_source": SOURCE,
                "source": SOURCE,
            }
        )
    return out


# ── /orchestration/pipelines ────────────────────────────────
def pipeline_id_for(model_id: str) -> str:
    return f"{_PIPELINE_PREFIX}{model_id}"


def pipelines() -> list[dict[str, Any]]:
    """실데이터 학습을 카탈로그 1행으로. 등록 파이프라인이 아니라 실행 job의 진입점이다."""
    out: list[dict[str, Any]] = []
    for model_id, name in PIPELINE_NAMES.items():
        job_rows = jobs_mod.list_jobs(model_id)
        version = resolve_version(model_id)
        if version is None and not job_rows:
            continue
        active = candidates_mod.get_active(model_id)
        out.append(
            {
                "id": pipeline_id_for(model_id),
                "name": name,
                "model_id": model_id,
                "experiment": "realdata-namwon-ridge",
                "trigger_policy": "수동 실행 · 후보 승인 후 반영",
                "base_version": active["version"] if active else None,
                "candidate_version": version,
                # 하단 실데이터 학습 패널과 같은 입력으로 실행하도록 최근 스냅샷을 함께 내려준다.
                "dataset_id": job_rows[0]["dataset_id"] if job_rows else None,
                "created_at": job_rows[-1]["requested_at"] if job_rows else None,
                "source": SOURCE,
            }
        )
    return out


# ── /orchestration/runs · /runs/{id}/logs ───────────────────
def _job_to_run(job: dict[str, Any]) -> dict[str, Any]:
    model_id = job["model_id"]
    version = job.get("candidate_version")
    candidate = candidates_mod.get_candidate(model_id, version) if version else None
    return {
        "run_id": job["job_id"],
        "pipeline_id": pipeline_id_for(model_id),
        "model_id": model_id,
        "trigger": "realdata",
        "state": _JOB_STATE_MAP.get(job["state"], job["state"]),
        "active_version": version,
        "started_at": job.get("started_at") or job["requested_at"],
        "finished_at": job.get("finished_at"),
        "stages": [],
        "dataset_id": job["dataset_id"],
        "error": job.get("error"),
        "candidate_metrics": {k: candidate["metrics"].get(k) for k in ("mae", "rmse", "wape")} if candidate else None,
        "source": SOURCE,
    }


def runs(pipeline_id: str | None = None) -> list[dict[str, Any]]:
    """실데이터 학습 job 이력(최신 우선). `list_jobs`가 이미 requested_at 내림차순이다."""
    if pipeline_id is not None and not pipeline_id.startswith(_PIPELINE_PREFIX):
        return []
    model_id = pipeline_id[len(_PIPELINE_PREFIX) :] if pipeline_id else None
    if model_id is not None and not is_realdata_model(model_id):
        return []
    return [_job_to_run(job) for job in jobs_mod.list_jobs(model_id)]


def run_logs(run_id: str) -> dict[str, Any] | None:
    """job 레코드의 상태 전이를 로그 라인으로. 별도 로그 적재 경로는 없다."""
    job = jobs_mod.get_job(run_id)
    if job is None:
        return None
    run = _job_to_run(job)
    lines = [
        {
            "ts": job["requested_at"],
            "level": "INFO",
            "message": f"학습 요청 접수 — 모델 {job['model_id']} · 데이터셋 {job['dataset_id']}",
        }
    ]
    if job.get("started_at"):
        lines.append({"ts": job["started_at"], "level": "INFO", "message": "스냅샷 로딩·학습 시작"})
    if job.get("finished_at"):
        if job.get("error"):
            lines.append({"ts": job["finished_at"], "level": "ERROR", "message": f"학습 실패 — {job['error']}"})
        else:
            metrics_text = ""
            if run["candidate_metrics"]:
                metrics_text = " · " + " ".join(
                    f"{key.upper()} {value:.6g}"
                    for key, value in run["candidate_metrics"].items()
                    if value is not None
                )
            lines.append(
                {
                    "ts": job["finished_at"],
                    "level": "INFO",
                    "message": f"후보 등록 — {job.get('candidate_version')}{metrics_text}",
                }
            )
    return {"run_id": run_id, "state": run["state"], "logs": lines, "source": SOURCE}
