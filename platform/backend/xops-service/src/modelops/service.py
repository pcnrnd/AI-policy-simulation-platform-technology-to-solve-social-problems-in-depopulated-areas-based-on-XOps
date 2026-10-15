from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from src.core.settings import get_settings
from src.mlops.training.dataset import build_dataset
from src.mlops.training.model import RidgeRegressor
from src.modelops import datasets, store
from src.realdata import evaluate, models, snapshot

_guard = threading.Lock()
_locks: dict[tuple[str, str], threading.Lock] = {}


def instances(mode: str) -> list[dict[str, Any]]:
    return store.list_assets(mode, "instance")


def register_instance(mode: str, body: dict[str, Any], actor: str) -> dict[str, Any]:
    dataset, _ = datasets.load(mode, body["dataset_id"], body["dataset_hash"])
    model_type = body["model_type"]
    if model_type not in store.MODEL_TYPES:
        raise HTTPException(400, "모델 종류를 선택하세요.")
    for instance in instances(mode):
        if (instance["model_type"], instance["region_id"], instance["target"]) == (model_type, dataset["region_id"], dataset["target"]):
            return instance
    instance = store.save(mode, "instance", {
        "id": store.new_id(body.get("id_prefix", "modelops_")), "created_at": store.now(),
        "name": body.get("name") or store.MODEL_TYPES[model_type], "model_type": model_type,
        "region_id": dataset["region_id"], "target": dataset["target"],
        "active_version": None, "previous_version": None,
    }, create=True)
    store.audit(mode, "instance_registered", instance["id"], actor)
    return instance


def register_pipeline(mode: str, body: dict[str, Any], actor: str) -> dict[str, Any]:
    if any(row["id"] == body["id"] for row in store.list_assets(mode, "pipeline")):
        raise HTTPException(409, "이미 등록된 파이프라인 ID입니다.")
    dataset, _ = datasets.load(mode, body["dataset_id"], body["dataset_hash"])
    instance = register_instance(mode, body, actor)
    pipeline = store.save(mode, "pipeline", {
        **body, "instance_id": instance["id"], "region_id": dataset["region_id"],
        "target": dataset["target"], "created_at": store.now(),
    }, create=True)
    store.audit(mode, "pipeline_registered", pipeline["id"], actor, dataset_id=dataset["id"], dataset_hash=dataset["content_hash"])
    return pipeline


def _train(mode: str, dataset: dict[str, Any], document: dict[str, Any], version_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if mode == "real":
        document = {**document, "dataset_id": dataset["id"], "file_path": dataset["private_path"]}
        record = snapshot.DatasetRecord(**document)
        outcome = models.train(record.model_id, record.dataset_id, version_id, dataset=record)
        artifact = json.loads(Path(outcome.artifact_path).read_text(encoding="utf-8"))
        return artifact, outcome.metrics, outcome.baseline
    training = build_dataset([document["region"]], horizon=1, max_lag=2, decline_threshold_pct=1.0)
    if len(training.rows) < 3:
        raise HTTPException(400, "학습에 필요한 관측 기간을 확인하세요.")
    names, rows = training.select(2)
    predictions, baselines = [], []
    for index in range(len(rows)):
        fold_x = [row for i, row in enumerate(rows) if i != index]
        fold_y = [y for i, y in enumerate(training.targets) if i != index]
        fit = RidgeRegressor.fit(fold_x, fold_y, ridge_lambda=1.0, feature_names=names)
        predictions.append(fit.predict([rows[index]])[0])
        baselines.append(sum(fold_y) / len(fold_y))
    fit = RidgeRegressor.fit(rows, training.targets, ridge_lambda=1.0, feature_names=names).to_dict()
    artifact = {"features": names, "coef": fit["coefficients"], "intercept": fit["intercept"],
                "train_means": fit["means"], "train_stds": fit["stds"], "lambda": fit["ridge_lambda"]}
    return artifact, evaluate.metric_bundle(training.targets, predictions), {"name": "leave_one_out_mean", **evaluate.metric_bundle(training.targets, baselines)}


def start_job(mode: str, pipeline_id: str, actor: str) -> dict[str, Any]:
    pipeline = store.get(mode, "pipeline", pipeline_id)
    datasets.load(mode, pipeline["dataset_id"], pipeline["dataset_hash"])
    key = (mode, pipeline["instance_id"])
    with _guard:
        lock = _locks.setdefault(key, threading.Lock())
    if not lock.acquire(blocking=False):
        raise HTTPException(409, "선택한 모델의 학습이 진행 중입니다.")
    try:
        if any(job["instance_id"] == key[1] and job["state"] in ("queued", "running") for job in store.list_assets(mode, "job")):
            raise HTTPException(409, "선택한 모델의 학습이 진행 중입니다.")
        prefix = pipeline.get("id_prefix", "modelops_")
        job = store.save(mode, "job", {"id": store.new_id(prefix), "version_id": store.new_id(prefix),
            "pipeline_id": pipeline_id, "instance_id": pipeline["instance_id"], "model_type": pipeline["model_type"],
            "dataset_id": pipeline["dataset_id"], "dataset_hash": pipeline["dataset_hash"],
            "created_at": store.now(), "state": "queued", "actor": actor}, create=True)
    except Exception:
        lock.release()
        raise
    threading.Thread(target=_run_job, args=(mode, job, lock), daemon=True).start()
    return job


def _run_job(mode: str, job: dict[str, Any], lock: threading.Lock) -> None:
    stage = "preparing"  # 실패 시 어느 단계에서 멈췄는지 화면 노드에 표시한다.
    try:
        job = store.save(mode, "job", {**job, "state": "running", "started_at": store.now()})
        dataset, document = datasets.load(mode, job["dataset_id"], job["dataset_hash"])
        instance = store.get(mode, "instance", job["instance_id"])
        stage = "training"
        artifact, metrics, baseline = _train(mode, dataset, document, job["version_id"])
        stage = "registering"
        artifact.update({"model_id": instance["id"], "model_type": instance["model_type"],
            "mode": mode, "source": mode, "region_id": instance["region_id"], "target": instance["target"],
            "version": job["version_id"], "dataset_id": dataset["id"], "content_hash": dataset["content_hash"],
            "dataset_path": dataset["private_path"], "job_id": job["id"]})
        path = get_settings().model_artifact_dir / "modelops" / mode / instance["id"] / (job["version_id"] + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(artifact, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        store.save(mode, "version", {"id": job["version_id"], "instance_id": instance["id"],
            "model_type": instance["model_type"], "region_id": instance["region_id"], "target": instance["target"],
            "dataset_id": dataset["id"], "dataset_hash": dataset["content_hash"], "source_dataset_id": dataset["source_dataset_id"],
            "observed_from": dataset["observed_from"], "observed_to": dataset["observed_to"], "features": artifact["features"],
            "job_id": job["id"], "metrics": metrics, "baseline": baseline, "artifact_path": str(path),
            "artifact_hash": hashlib.sha256(path.read_bytes()).hexdigest(), "created_at": store.now(), "status": "candidate"}, create=True)
        store.audit(mode, "training_saved", instance["id"], job["actor"], job_id=job["id"], version_id=job["version_id"], dataset_hash=dataset["content_hash"])
        job = store.save(mode, "job", {**job, "state": "saved", "finished_at": store.now(), "artifact_path": str(path)})
    except Exception as exc:
        store.save(mode, "job", {**job, "state": "failed", "finished_at": store.now(), "error": type(exc).__name__, "failed_stage": stage})
        store.audit(mode, "training_failed", job["instance_id"], job["actor"], job_id=job["id"], reason=type(exc).__name__)
    finally:
        lock.release()


def versions(mode: str, instance_id: str | None = None) -> list[dict[str, Any]]:
    """버전 목록 — status는 저장값이 아니라 instance의 활성·이전 버전으로 파생한다. instance_id가 없으면 모드 전체."""
    if instance_id is not None:
        store.get(mode, "instance", instance_id)
    instances = {row["id"]: row for row in store.list_assets(mode, "instance")}

    def status(version: dict[str, Any]) -> str:
        instance = instances.get(version["instance_id"], {})
        return "active" if version["id"] == instance.get("active_version") else "previous" if version["id"] == instance.get("previous_version") else "candidate"

    return [{**version, "status": status(version)} for version in store.list_assets(mode, "version")
            if instance_id is None or version["instance_id"] == instance_id]


def load_artifact(mode: str, instance_id: str, version_id: str) -> dict[str, Any]:
    store.get(mode, "instance", instance_id)
    version = store.get(mode, "version", version_id)
    if version["instance_id"] != instance_id:
        raise HTTPException(404, "현재 모델의 버전을 선택하세요.")
    datasets.load(mode, version["dataset_id"], version["dataset_hash"])
    path = Path(version["artifact_path"])
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != version["artifact_hash"]:
        raise HTTPException(409, "학습 결과의 내용을 확인하세요.")
    return json.loads(path.read_text(encoding="utf-8"))


def activate(mode: str, instance_id: str, version_id: str, actor: str, *, rollback: bool = False) -> dict[str, Any]:
    instance = store.get(mode, "instance", instance_id)
    version = store.get(mode, "version", version_id)
    if any(version[key] != instance[key] for key in ("model_type", "region_id", "target")) or version["instance_id"] != instance_id:
        raise HTTPException(400, "현재 모델의 버전을 선택하세요.")
    if rollback and version_id != instance["previous_version"]:
        raise HTTPException(400, "이전 활성 버전을 선택하세요.")
    datasets.load(mode, version["dataset_id"], version["dataset_hash"])
    job = store.get(mode, "job", version["job_id"])
    path = Path(version["artifact_path"])
    if job["state"] != "saved" or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != version["artifact_hash"]:
        raise HTTPException(409, "학습 결과와 파일을 확인한 뒤 다시 시도하세요.")
    if not rollback and (version["metrics"].get("mae") is None or version["baseline"].get("mae") is None or version["metrics"]["mae"] >= version["baseline"]["mae"]):
        raise HTTPException(400, "검증 성능이 기준선을 개선하지 못했습니다. 학습데이터를 확인하세요.")
    if instance["active_version"] == version_id:
        return instance
    instance = store.save(mode, "instance", {**instance, "previous_version": instance["active_version"], "active_version": version_id, "activated_at": store.now()})
    store.audit(mode, "version_rolled_back" if rollback else "version_activated", instance_id, actor, version_id=version_id, previous_version=instance["previous_version"])
    return instance


def recover_jobs() -> None:
    for mode in ("mock", "real"):
        for job in store.list_assets(mode, "job"):
            if job["state"] in ("queued", "running"):
                store.save(mode, "job", {**job, "state": "failed", "error": "restart", "finished_at": store.now()})
