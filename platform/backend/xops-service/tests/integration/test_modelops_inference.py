from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from src.core.db import _conn
from src.modelops import inference, store
from src.realdata import snapshot
from src.realdata.features import add_month

BASE = "/api/v3/modelops"


@pytest.fixture(autouse=True)
def clean_assets():
    _conn().execute("DELETE FROM mo_assets")
    _conn().commit()
    yield
    _conn().execute("DELETE FROM mo_assets")
    _conn().commit()


def train(client, auth, mode="mock", kind="simulation", activate=True):
    h = {**auth, "X-Data-Mode": mode}
    source = client.get(BASE+"/available-datasets", headers=h).json()["data"][0]
    ds = client.post(BASE+"/datasets", headers=h, json={"name": "추론 입력", "source_dataset_id": source["id"], "source_hash": source["content_hash"]}).json()
    p = client.post(BASE+"/pipelines", headers=h, json={"id": "test_predict_"+kind, "name": "추론 검증", "model_type": kind, "dataset_id": ds["id"], "dataset_hash": ds["content_hash"]}).json()
    job = client.post(BASE+"/pipelines/"+p["id"]+"/run", headers=h).json()
    deadline = time.monotonic()+10
    while time.monotonic() < deadline:
        job = client.get(BASE+"/jobs/"+job["id"], headers=h).json()
        if job["state"] in ("saved", "failed"):
            break
    assert job["state"] == "saved"
    url = BASE+"/instances/"+p["instance_id"]+"/versions/"+job["version_id"]
    if activate:
        assert client.post(url+"/activate", headers=h).status_code == 200
    return h, ds, p, job, url


def body(context, changes=None):
    proof = context["provenance"]
    return {**{key: proof[key] for key in ("region_id", "target", "dataset_id", "dataset_hash", "dong_code")}, "adjustments": changes or {}}


@pytest.mark.parametrize("kind", ["factor_analysis", "simulation"])
def test_active_mock_prediction_reconstructs_and_preserves_every_asset(client, auth_headers, kind):
    h, ds, p, job, url = train(client, auth_headers, kind=kind)
    before = [tuple(row) for row in _conn().execute("SELECT * FROM mo_assets ORDER BY mode,kind,id")]
    context = client.get(url+"/inputs", headers=h).json()
    result = client.post(url+"/predict", headers=h, json=body(context)).json()
    artifact = client.get(url+"/artifact", headers=h).json()
    expected = artifact["intercept"]+sum(coef*(result["inputs"][name]-mean)/std for name,coef,mean,std in zip(artifact["features"],artifact["coef"],artifact["train_means"],artifact["train_stds"]))
    assert result["prediction"] == pytest.approx(expected)
    assert result["intercept"]+sum(row["value"] for row in result["contributions"]) == pytest.approx(expected)
    assert result["delta"] == 0 and result["mode"] == result["source"] == "mock"
    assert result["provenance"]["dataset_id"] == ds["id"] and result["provenance"]["dataset_hash"] == ds["content_hash"]
    assert result["provenance"]["version_id"] == job["version_id"] and result["provenance"]["job_id"] == job["id"]
    assert result["provenance"]["model_type"] == kind and result["inference_id"].startswith("inference_")
    assert [tuple(row) for row in _conn().execute("SELECT * FROM mo_assets ORDER BY mode,kind,id")] == before
    if kind == "factor_analysis":
        assert result["recommendations"] == []
    else:
        assert result["recommendations"]
        assert [row["delta"] for row in result["recommendations"]] == sorted((row["delta"] for row in result["recommendations"]), reverse=True)
        for row in result["recommendations"]:
            index = artifact["features"].index(row["feature"])
            assert row["delta"] == pytest.approx(artifact["coef"][index]*row["input_delta"]/artifact["train_stds"][index])


def test_inference_rejects_inactive_foreign_mode_and_missing_auth(client, auth_headers):
    h, ds, p, job, url = train(client, auth_headers, activate=False)
    assert client.get(url+"/inputs", headers=h).status_code == 409
    assert client.get(url+"/inputs").status_code == 401
    assert client.get(url+"/inputs", headers={**auth_headers,"X-Data-Mode":"real"}).status_code == 404
    assert client.post(url+"/activate", headers=h).status_code == 200
    context = client.get(url+"/inputs", headers=h).json()
    assert client.post(url+"/predict", headers={**auth_headers,"X-Data-Mode":"real"}, json=body(context)).status_code == 404


@pytest.mark.parametrize("field,value", [("region_id","shinan"),("target","other"),("dataset_id","missing"),("dataset_hash","a"*64)])
def test_prediction_checks_registered_basis(client, auth_headers, field, value):
    h, ds, p, job, url = train(client, auth_headers)
    context = client.get(url+"/inputs", headers=h).json()
    request = body(context)
    request[field] = value
    assert client.post(url+"/predict", headers=h, json=request).status_code == 409


def test_prediction_rejects_invalid_inputs_and_hash_tampering(client, auth_headers):
    h, ds, p, job, url = train(client, auth_headers)
    context = client.get(url+"/inputs", headers=h).json()
    assert client.post(url+"/predict", headers=h, json={}).status_code == 422
    assert client.post(url+"/predict", headers=h, json=body(context,{"not_a_feature":1})).status_code == 422
    feature = context["controls"][0]["feature"]
    assert client.post(url+"/predict", headers=h, json=body(context,{feature:True})).status_code == 422
    invalid = json.dumps(body(context,{feature:float('nan')}))
    assert client.post(url+"/predict", headers=h, content=invalid).status_code == 422
    assert client.get(url+"/inputs?dong_code=missing", headers=h).status_code == 422
    outside = client.post(url+"/predict", headers=h, json=body(context,{feature:context["controls"][0]["max"]+1})).json()
    assert feature in outside["outside_observed_range"]
    Path(ds["private_path"]).write_text("{}",encoding="utf-8")
    assert client.post(url+"/predict", headers=h, json=body(context)).status_code == 409


def test_real_inference_uses_copied_observations_and_existing_formula(client, auth_headers, monkeypatch, tmp_path):
    rows = [{"base_ym":add_month(202101,i),"dong_code":"45190250","y":100+3*i+(i%3),"nonlocal_visitors":100+3*i+(i%3),"local_visitors":200,"foreign_visitors":1} for i in range(36)]
    original = tmp_path/"original.json"
    record = snapshot.DatasetRecord(dataset_id="ds-testreal",target="nonlocal_visitors",model_id=snapshot.TARGETS["nonlocal_visitors"],
        spec={"target":"nonlocal_visitors","model_id":snapshot.TARGETS["nonlocal_visitors"]},quality={},observed_from=202101,observed_to=202312,
        row_count=len(rows),excluded_rows={},content_hash=snapshot._content_hash(rows),file_path=str(original),created_at=store.now(),rows=rows)
    original.write_text(json.dumps(asdict(record)),encoding="utf-8")
    before_hash = hashlib.sha256(original.read_bytes()).hexdigest()
    monkeypatch.setattr(snapshot,"list_datasets",lambda:[asdict(record)])
    monkeypatch.setattr(snapshot,"load_dataset",lambda _:record)
    h, ds, p, job, url = train(client,auth_headers,mode="real")
    context = client.get(url+"/inputs",headers=h).json()
    assert context["provenance"]["prediction_period"] == 202401
    assert context["provenance"]["evaluation_period"] == {"from":202310,"to":202312,"n":3}
    assert context["inputs"]["y_lag1"] == rows[-1]["y"]
    result = client.post(url+"/predict",headers=h,json=body(context,{"y_lag1":context["inputs"]["y_lag1"]+1})).json()
    artifact = client.get(url+"/artifact",headers=h).json()
    idx = artifact["features"].index("y_lag1")
    assert result["delta"] == pytest.approx(artifact["coef"][idx]/artifact["train_stds"][idx])
    assert result["provenance"]["dataset_id"] == ds["id"]
    assert client.post(url+"/predict",headers=h,json=body(context,{"month_sin":0})).status_code == 422
    assert hashlib.sha256(original.read_bytes()).hexdigest() == before_hash


def test_missing_year_observation_keeps_existing_fallback_and_overflow_is_error():
    assert inference._adjust({"has_yoy":0,"y_lag1":10,"y_yoy":10},{"y_lag1":20})["y_yoy"] == 20
    artifact={"features":["x"],"coef":[1e300],"train_means":[0],"train_stds":[1],"intercept":0}
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as error:
        inference._calculate(artifact,{"x":1e300})
    assert error.value.status_code == 422
    stable={"features":["x"],"coef":[1e200],"train_means":[0],"train_stds":[1e200],"intercept":0}
    assert inference._calculate(stable,{"x":1e200})[0] == 1e200
    with pytest.raises(HTTPException):
        inference._difference(1e308,-1e308)
