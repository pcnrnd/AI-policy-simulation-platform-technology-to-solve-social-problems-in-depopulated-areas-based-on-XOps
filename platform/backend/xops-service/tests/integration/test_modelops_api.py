from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from src.core.db import _conn
from src.modelops import service, store
from src.realdata import snapshot

BASE = "/api/v3/modelops"


@pytest.fixture(autouse=True)
def clean_modelops():
    _conn().execute("DELETE FROM mo_assets")
    _conn().commit()
    yield
    _conn().execute("DELETE FROM mo_assets")
    _conn().commit()


def headers(auth, mode):
    return {**auth, "X-Data-Mode": mode}


def register(client, auth, mode="mock"):
    h = headers(auth, mode)
    sources = client.get(BASE + "/available-datasets", headers=h).json()["data"]
    response = client.post(BASE + "/datasets", headers=h, json={"name": "학습 입력", "source_dataset_id": sources[0]["id"], "source_hash": sources[0]["content_hash"], "id_prefix": "test_"})
    assert response.status_code == 201
    return response.json()


def pipeline(client, auth, ds, model_type="factor_analysis", mode="mock"):
    response = client.post(BASE + "/pipelines", headers=headers(auth, mode), json={"id": "test_" + model_type, "name": "등록 학습", "model_type": model_type,
        "dataset_id": ds["id"], "dataset_hash": ds["content_hash"], "id_prefix": "test_"})
    assert response.status_code == 201
    return response.json()


def wait_job(client, h, job_id):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        row = client.get(BASE + "/jobs/" + job_id, headers=h).json()
        if row["state"] in ("saved", "failed"):
            return row
        time.sleep(0.01)
    raise AssertionError("job not terminal")


def test_types_and_mode_scoped_dataset_registration(client, auth_headers):
    ds = register(client, auth_headers)
    assert ds["source"] == ds["mode"] == "mock"
    assert Path(ds["private_path"]).is_file()
    assert ds["private_hash"] == hashlib.sha256(Path(ds["private_path"]).read_bytes()).hexdigest()
    catalog = client.get(BASE + "/catalog", headers=headers(auth_headers, "mock")).json()
    assert {row["model_type"] for row in catalog["types"]} == {"factor_analysis", "simulation"}
    assert catalog["datasets"][0]["id"] == ds["id"]
    assert client.get(BASE + "/datasets", headers=headers(auth_headers, "real")).json()["data"] == []
    p = pipeline(client, auth_headers, ds)
    assert p["dataset_hash"] == ds["content_hash"]
    assert client.post(BASE + "/pipelines/" + p["id"] + "/run", headers=headers(auth_headers, "real")).status_code == 404


def test_registration_rejects_unknown_hash_and_third_type(client, auth_headers):
    h = headers(auth_headers, "mock")
    assert client.post(BASE + "/datasets", headers=h, json={"name": "x", "source_dataset_id": "missing", "source_hash": "a" * 64}).status_code == 400
    source = client.get(BASE + "/available-datasets", headers=h).json()["data"][0]
    assert client.post(BASE + "/datasets", headers=h, json={"name": "x", "source_dataset_id": source["id"], "source_hash": "a" * 64}).status_code == 409
    ds = register(client, auth_headers)
    body = {"name": "x", "model_type": "population", "dataset_id": ds["id"], "dataset_hash": ds["content_hash"]}
    assert client.post(BASE + "/instances", headers=h, json=body).status_code == 422
    body["model_type"] = "simulation"
    body["dataset_hash"] = "a" * 64
    assert client.post(BASE + "/instances", headers=h, json=body).status_code == 409
    assert client.post(BASE + "/datasets", json={}).status_code == 401
    assert client.get(BASE + "/catalog", headers=headers(auth_headers, "other")).status_code == 400


@pytest.mark.parametrize("model_type", ["factor_analysis", "simulation"])
def test_actual_mock_training_keeps_unique_versions_and_audit(client, auth_headers, model_type):
    h = headers(auth_headers, "mock")
    ds = register(client, auth_headers)
    p = pipeline(client, auth_headers, ds, model_type)
    jobs = []
    for _ in range(2):
        accepted = client.post(BASE + "/pipelines/" + p["id"] + "/run", headers=h)
        assert accepted.status_code == 202
        job = wait_job(client, h, accepted.json()["id"])
        assert job["state"] == "saved"
        jobs.append(job)
    assert jobs[0]["id"] != jobs[1]["id"]
    assert jobs[0]["version_id"] != jobs[1]["version_id"]
    assert jobs[0]["artifact_path"] != jobs[1]["artifact_path"]
    assert all(Path(job["artifact_path"]).is_file() for job in jobs)
    url = BASE + "/instances/" + p["instance_id"] + "/versions"
    versions = client.get(url, headers=h).json()["data"]
    assert len(versions) == 2
    assert all(row["dataset_hash"] == ds["content_hash"] for row in versions)
    artifact = client.get(url + "/" + jobs[0]["version_id"] + "/artifact", headers=h).json()
    assert artifact["model_type"] == model_type and artifact["mode"] == "mock"
    assert len(artifact["features"]) == len(artifact["coef"]) == len(artifact["train_means"])
    first = client.post(url + "/" + jobs[0]["version_id"] + "/activate", headers=h)
    assert first.status_code == 200
    assert client.post(url + "/" + jobs[0]["version_id"] + "/activate", headers=h).json()["active_version"] == jobs[0]["version_id"]
    second = client.post(url + "/" + jobs[1]["version_id"] + "/activate", headers=h)
    assert second.status_code == 200
    rolled = client.post(url + "/" + jobs[0]["version_id"] + "/rollback", headers=h)
    assert rolled.status_code == 200
    assert rolled.json()["active_version"] == jobs[0]["version_id"]
    assert client.get(url, headers=headers(auth_headers, "real")).status_code == 404
    actions = {row["action"] for row in client.get(BASE + "/catalog", headers=h).json()["audit"]}
    assert {"dataset_registered", "instance_registered", "pipeline_registered", "training_saved", "version_activated", "version_rolled_back"} <= actions


def test_job_conflict_failure_and_restart_recovery(client, auth_headers, monkeypatch):
    ds = register(client, auth_headers)
    p = pipeline(client, auth_headers, ds)
    h = headers(auth_headers, "mock")
    entered, release = threading.Event(), threading.Event()
    def failing(*args):
        entered.set()
        release.wait(5)
        raise ValueError("invalid input")
    monkeypatch.setattr(service, "_train", failing)
    accepted = client.post(BASE + "/pipelines/" + p["id"] + "/run", headers=h)
    assert entered.wait(2)
    assert client.post(BASE + "/pipelines/" + p["id"] + "/run", headers=h).status_code == 409
    release.set()
    assert wait_job(client, h, accepted.json()["id"])["state"] == "failed"
    stale = {**accepted.json(), "id": "test_stale", "state": "running"}
    store.save("mock", "job", stale, create=True)
    service.recover_jobs()
    assert store.get("mock", "job", stale["id"])["error"] == "restart"
    assert client.delete(BASE + "/pipelines/" + p["id"], headers=h).status_code == 200
    assert client.get(BASE + "/jobs/" + accepted.json()["id"], headers=h).status_code == 200


def test_private_registration_rejects_tampering(client, auth_headers):
    ds = register(client, auth_headers)
    Path(ds["private_path"]).write_text("{}", encoding="utf-8")
    body = {"name": "x", "model_type": "simulation", "dataset_id": ds["id"], "dataset_hash": ds["content_hash"]}
    assert client.post(BASE + "/instances", headers=headers(auth_headers, "mock"), json=body).status_code == 409


def test_real_registration_copies_source_and_trains_that_copy(client, auth_headers, monkeypatch, tmp_path):
    from src.realdata.features import add_month
    rows = [{"base_ym": add_month(202101, i), "dong_code": "45190250", "y": 100 + 3 * i + (i % 3),
             "nonlocal_visitors": 100 + 3 * i + (i % 3), "local_visitors": 200, "foreign_visitors": 1} for i in range(36)]
    record = snapshot.DatasetRecord(dataset_id="ds-abc123def456", target="nonlocal_visitors", model_id=snapshot.TARGETS["nonlocal_visitors"],
        spec={"target": "nonlocal_visitors", "model_id": snapshot.TARGETS["nonlocal_visitors"]}, quality={}, observed_from=202101,
        observed_to=202312, row_count=len(rows), excluded_rows={}, content_hash=snapshot._content_hash(rows), file_path=str(tmp_path / "original.json"), created_at=store.now(), rows=rows)
    Path(record.file_path).write_text(json.dumps(asdict(record)), encoding="utf-8")
    original_hash = hashlib.sha256(Path(record.file_path).read_bytes()).hexdigest()
    monkeypatch.setattr(snapshot, "list_datasets", lambda: [asdict(record)])
    monkeypatch.setattr(snapshot, "load_dataset", lambda _: record)
    ds = register(client, auth_headers, "real")
    assert ds["features"] and ds["private_path"] != record.file_path
    p = pipeline(client, auth_headers, ds, "simulation", "real")
    h = headers(auth_headers, "real")
    job = client.post(BASE + "/pipelines/" + p["id"] + "/run", headers=h).json()
    terminal = wait_job(client, h, job["id"])
    assert terminal["state"] == "saved"
    artifact = json.loads(Path(terminal["artifact_path"]).read_text())
    assert artifact["dataset_id"] == ds["id"] and artifact["content_hash"] == ds["content_hash"]
    assert artifact["source"] == "real"
    assert hashlib.sha256(Path(record.file_path).read_bytes()).hexdigest() == original_hash


def test_version_scope_integrity_and_activation_gate(client, auth_headers):
    ds = register(client, auth_headers)
    p = pipeline(client, auth_headers, ds)
    h = headers(auth_headers, "mock")
    job = wait_job(client, h, client.post(BASE + "/pipelines/" + p["id"] + "/run", headers=h).json()["id"])
    url = BASE + "/instances/" + p["instance_id"] + "/versions/" + job["version_id"]
    assert client.post(url + "/rollback", headers=h).status_code == 400
    assert client.get(url + "/artifact", headers=headers(auth_headers, "real")).status_code == 404
    second = pipeline(client, auth_headers, ds, "simulation")
    wrong_url = BASE + "/instances/" + second["instance_id"] + "/versions/" + job["version_id"]
    assert client.get(wrong_url + "/artifact", headers=h).status_code == 404
    assert client.post(wrong_url + "/activate", headers=h).status_code == 400
    version = store.get("mock", "version", job["version_id"])
    store.save("mock", "version", {**version, "metrics": {**version["metrics"], "mae": version["baseline"]["mae"]}})
    assert client.post(url + "/activate", headers=h).status_code == 400
    assert store.get("mock", "instance", p["instance_id"])["active_version"] is None
    store.save("mock", "version", version)
    Path(version["artifact_path"]).write_text("{}", encoding="utf-8")
    assert client.get(url + "/artifact", headers=h).status_code == 409
    assert client.post(url + "/activate", headers=h).status_code == 409


def test_duplicate_pipeline_does_not_create_extra_instance(client, auth_headers):
    ds = register(client, auth_headers)
    p = pipeline(client, auth_headers, ds)
    h = headers(auth_headers, "mock")
    before = client.get(BASE + "/catalog", headers=h).json()["instances"]
    body = {"id": p["id"], "name": "다른 모델", "model_type": "simulation", "dataset_id": ds["id"], "dataset_hash": ds["content_hash"]}
    assert client.post(BASE + "/pipelines", headers=h, json=body).status_code == 409
    assert client.get(BASE + "/catalog", headers=h).json()["instances"] == before


def test_pipeline_experiment_is_saved_and_bounded(client, auth_headers):
    ds = register(client, auth_headers)
    h = headers(auth_headers, "mock")
    body = {"id": "test_exp", "name": "실험 학습", "model_type": "factor_analysis", "dataset_id": ds["id"], "dataset_hash": ds["content_hash"], "experiment": "EXP-POP-DECLINE-031"}
    assert client.post(BASE + "/pipelines", headers=h, json=body).status_code == 201
    assert client.get(BASE + "/catalog", headers=h).json()["pipelines"][0]["experiment"] == "EXP-POP-DECLINE-031"
    assert client.post(BASE + "/pipelines", headers=h, json={**body, "id": "test_exp2", "experiment": "x" * 65}).status_code == 422
    plain = pipeline(client, auth_headers, ds, "simulation")
    assert plain["experiment"] == ""


def test_catalog_lists_versions_in_current_mode_only(client, auth_headers):
    ds = register(client, auth_headers)
    p = pipeline(client, auth_headers, ds)
    h = headers(auth_headers, "mock")
    job = wait_job(client, h, client.post(BASE + "/pipelines/" + p["id"] + "/run", headers=h).json()["id"])
    versions = client.get(BASE + "/catalog", headers=h).json()["versions"]
    assert [row["id"] for row in versions] == [job["version_id"]]
    assert versions[0]["instance_id"] == p["instance_id"]
    assert client.get(BASE + "/catalog", headers=headers(auth_headers, "real")).json()["versions"] == []


def test_failed_job_records_failed_stage(client, auth_headers, monkeypatch):
    ds = register(client, auth_headers)
    p = pipeline(client, auth_headers, ds)
    h = headers(auth_headers, "mock")
    def failing(*args):
        raise ValueError("invalid input")
    monkeypatch.setattr(service, "_train", failing)
    job = wait_job(client, h, client.post(BASE + "/pipelines/" + p["id"] + "/run", headers=h).json()["id"])
    assert job["state"] == "failed" and job["failed_stage"] == "training"
    original = service.datasets.load
    calls = []
    def load_once(*args):
        calls.append(args)
        if len(calls) > 1:
            raise ValueError("tampered")
        return original(*args)
    monkeypatch.setattr(service.datasets, "load", load_once)
    job = wait_job(client, h, client.post(BASE + "/pipelines/" + p["id"] + "/run", headers=h).json()["id"])
    assert job["failed_stage"] == "preparing"
