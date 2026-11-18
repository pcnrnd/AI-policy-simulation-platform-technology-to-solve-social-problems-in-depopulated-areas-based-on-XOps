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


def test_mock_factor_panel_trains_activates_and_explains_with_named_factors(client, auth_headers):
    from src.modelops import factor_panel
    h = {**auth_headers, "X-Data-Mode": "mock"}
    sources = client.get(BASE+"/available-datasets", headers=h).json()["data"]
    source = next(row for row in sources if row["id"] == "mock_factor_namwon")
    assert sources[0]["id"] == "mock_namwon"  # 기존 지역 인구 추이 원천은 그대로 앞에 있다
    ds = client.post(BASE+"/datasets", headers=h, json={"name": "요인 패널", "source_dataset_id": source["id"], "source_hash": source["content_hash"]}).json()
    assert ds["features"] == factor_panel.FEATURES and ds["row_count"] == 120 and ds["target"] == "living_population"
    p = client.post(BASE+"/pipelines", headers=h, json={"id": "test_factor_panel", "name": "요인 패널 학습", "model_type": "factor_analysis",
        "dataset_id": ds["id"], "dataset_hash": ds["content_hash"]}).json()
    job = client.post(BASE+"/pipelines/"+p["id"]+"/run", headers=h).json()
    deadline = time.monotonic()+10
    while time.monotonic() < deadline and job["state"] not in ("saved", "failed"):
        job = client.get(BASE+"/jobs/"+job["id"], headers=h).json()
    assert job["state"] == "saved"
    url = BASE+"/instances/"+p["instance_id"]+"/versions/"+job["version_id"]
    assert client.post(url+"/activate", headers=h).status_code == 200  # MAE < 기준선
    context = client.get(url+"/inputs", headers=h).json()
    assert context["dong_options"] == [{"code": "all", "name": "전체"}] and len(context["controls"]) == 9
    proof = context["provenance"]
    assert proof["evaluation_period"] == {"from": 202401, "to": 202412, "n": 12} and proof["prediction_period"] == 202412
    assert proof["baseline"]["name"] == "same_month_last_year" and proof["metrics"]["mae"] < proof["baseline"]["mae"]
    result = client.post(url+"/predict", headers=h, json=body(context)).json()
    assert [row["feature"] for row in result["contributions"]] == factor_panel.FEATURES
    assert result["intercept"]+sum(row["value"] for row in result["contributions"]) == pytest.approx(result["prediction"])
    assert result["recommendations"] == []
    # 기존 지역 인구 추이 원천의 학습·추론 경로는 그대로다(공존).
    legacy = client.post(BASE+"/datasets", headers=h, json={"name": "인구 추이", "source_dataset_id": "mock_namwon", "source_hash": sources[0]["content_hash"]}).json()
    assert legacy["features"] == ["rate_t-1", "riskIndex", "birthRate", "agingIndex"]


# ── ① 분석 기간·비교 기준(factor-period) ──
def period_body(context, start, end, baseline="train_mean"):
    proof = context["provenance"]
    return {**{key: proof[key] for key in ("region_id", "target", "dataset_id", "dataset_hash", "dong_code")},
        "period_from": start, "period_to": end, "baseline": baseline}


def real_model(client, auth, monkeypatch, tmp_path, skip=()):
    # 합성 관측 36개월(2021-01~2023-12), skip 달은 원천 공백 — 실데이터 값은 쓰지 않는다.
    rows = [{"base_ym": add_month(202101, i), "dong_code": "45190250", "y": 100+3*i+(i % 3)*7, "nonlocal_visitors": 100+3*i,
        "local_visitors": 200, "foreign_visitors": 1} for i in range(36) if add_month(202101, i) not in skip]
    original = tmp_path/"period.json"
    record = snapshot.DatasetRecord(dataset_id="ds-testperiod", target="nonlocal_visitors", model_id=snapshot.TARGETS["nonlocal_visitors"],
        spec={"target": "nonlocal_visitors", "model_id": snapshot.TARGETS["nonlocal_visitors"]}, quality={}, observed_from=202101, observed_to=202312,
        row_count=len(rows), excluded_rows={}, content_hash=snapshot._content_hash(rows), file_path=str(original), created_at=store.now(), rows=rows)
    original.write_text(json.dumps(asdict(record)), encoding="utf-8")
    monkeypatch.setattr(snapshot, "list_datasets", lambda: [asdict(record)])
    monkeypatch.setattr(snapshot, "load_dataset", lambda _: record)
    h, ds, p, job, url = train(client, auth, mode="real", kind="factor_analysis")
    return h, url, client.get(url+"/inputs", headers=h).json(), rows


def assert_identity(result):
    for row in result["months"]:
        if row["status"] == "used":
            assert row["prediction"]-row["reference_prediction"] == pytest.approx(sum(c["value"] for c in row["contributions"]))


def test_period_train_mean_month_matches_predict_and_aggregates(client, auth_headers, monkeypatch, tmp_path):
    h, url, context, rows = real_model(client, auth_headers, monkeypatch, tmp_path)
    before = [tuple(row) for row in _conn().execute("SELECT * FROM mo_assets ORDER BY mode,kind,id")]
    predicted = client.post(url+"/predict", headers=h, json=body(context)).json()
    one = client.post(url+"/factor-period", headers=h, json=period_body(context, 202401, 202401)).json()
    month = one["months"][0]
    assert month["status"] == "used" and month["actual"] is None and month["reference_prediction"] == one["intercept"]
    assert month["prediction"] == pytest.approx(predicted["prediction"])
    assert [c["value"] for c in month["contributions"]] == pytest.approx([c["value"] for c in predicted["contributions"]])
    year = client.post(url+"/factor-period", headers=h, json=period_body(context, 202301, 202312)).json()
    assert year["summary"]["used_months"] == 12 and year["summary"]["skipped"] == []
    assert year["provenance"]["period_from"] == 202301 and year["provenance"]["period_baseline"] == "train_mean"
    assert year["months"][0]["actual"] == next(row["y"] for row in rows if row["base_ym"] == 202301)
    assert_identity(year)
    total = {row["feature"]: row["value"] for row in year["summary"]["total"]}
    mean = {row["feature"]: row["value"] for row in year["summary"]["mean"]}
    for name, value in total.items():
        assert value == pytest.approx(sum(c["value"] for m in year["months"] for c in m["contributions"] if c["feature"] == name))
        assert mean[name] == pytest.approx(value/12)
    assert year["summary"]["total_delta"] == pytest.approx(sum(m["prediction"]-m["reference_prediction"] for m in year["months"]))
    assert [tuple(row) for row in _conn().execute("SELECT * FROM mo_assets ORDER BY mode,kind,id")] == before  # 읽기 전용


def test_period_prev_year_cancels_season_and_prev_period_shifts_by_length(client, auth_headers, monkeypatch, tmp_path):
    h, url, context, _ = real_model(client, auth_headers, monkeypatch, tmp_path)
    result = client.post(url+"/factor-period", headers=h, json=period_body(context, 202305, 202310, "prev_year")).json()
    assert_identity(result)
    for row in result["months"]:
        assert row["reference_ym"] == add_month(row["base_ym"], -12)
        season = [c for c in row["contributions"] if c["feature"] in ("month_sin", "month_cos")]
        assert len(season) == 2 and all(c["value"] == pytest.approx(0, abs=1e-9) for c in season)
    shifted = client.post(url+"/factor-period", headers=h, json=period_body(context, 202305, 202310, "prev_period")).json()
    assert [row["reference_ym"] for row in shifted["months"]] == [add_month(row["base_ym"], -6) for row in shifted["months"]]
    assert_identity(shifted)


def test_period_skips_gap_months_without_filling_and_excludes_them_from_mean(client, auth_headers, monkeypatch, tmp_path):
    h, url, context, _ = real_model(client, auth_headers, monkeypatch, tmp_path, skip=(202206, 202207))
    result = client.post(url+"/factor-period", headers=h, json=period_body(context, 202206, 202307, "prev_year")).json()
    status = {row["base_ym"]: row["status"] for row in result["months"]}
    # 공백 달과 그 뒤 랙 3개월이 모자란 달은 입력 없음, 이듬해 같은 달은 기준 없음
    assert [t for t, s in status.items() if s == "no_input"] == [202206, 202207, 202208, 202209, 202210]
    assert [t for t, s in status.items() if s == "no_reference"] == [202306, 202307]
    assert result["summary"]["used_months"] == 7 and len(result["summary"]["skipped"]) == 7
    total = {row["feature"]: row["value"] for row in result["summary"]["total"]}
    assert all(row["value"] == pytest.approx(total[row["feature"]]/7) for row in result["summary"]["mean"])
    assert all(m["contributions"] == [] and m["prediction"] is None for m in result["months"] if m["status"] != "used")
    only_gap = client.post(url+"/factor-period", headers=h, json=period_body(context, 202206, 202207))
    assert only_gap.status_code == 422 and "계산할 수 있는 달이 없습니다" in only_gap.json()["detail"]


@pytest.mark.parametrize("start,end,baseline", [
    (202313, 202401, "train_mean"),   # 월 형식
    (202312, 202301, "train_mean"),   # 시작 > 끝
    (202101, 202401, "train_mean"),   # 36개월 초과(37)
    (202103, 202110, "train_mean"),   # 관측 시작+3 이전
    (202312, 202402, "train_mean"),   # 예측월(관측 다음 달) 이후
    (202301, 202312, "last_month"),   # 알 수 없는 비교 기준(스키마)
    (202104, 202104, "prev_year"),    # 기준 달이 없어 계산할 달 0개
])
def test_period_rejects_invalid_requests(client, auth_headers, monkeypatch, tmp_path, start, end, baseline):
    h, url, context, _ = real_model(client, auth_headers, monkeypatch, tmp_path)
    assert client.post(url+"/factor-period", headers=h, json=period_body(context, start, end, baseline)).status_code == 422


def test_period_boundaries_and_basis(client, auth_headers, monkeypatch, tmp_path):
    h, url, context, _ = real_model(client, auth_headers, monkeypatch, tmp_path)
    full = client.post(url+"/factor-period", headers=h, json=period_body(context, 202104, 202401)).json()  # 허용 범위 양 끝, 34개월
    assert full["summary"]["used_months"] == 34 and full["months"][-1]["actual"] is None
    one = period_body(context, 202301, 202301)
    assert client.post(url+"/factor-period", headers=h, json={**one, "dataset_hash": "a"*64}).status_code == 409
    assert client.post(url+"/factor-period", headers=h, json={**one, "extra": 1}).status_code == 422
    assert client.post(url+"/factor-period", headers={**h, "X-Data-Mode": "mock"}, json=one).status_code == 404
    assert client.post(url+"/factor-period", json=one).status_code == 401


def test_period_demo_panel_and_legacy_mock_model(client, auth_headers):
    h = {**auth_headers, "X-Data-Mode": "mock"}
    source = next(row for row in client.get(BASE+"/available-datasets", headers=h).json()["data"] if row["id"] == "mock_factor_namwon")
    ds = client.post(BASE+"/datasets", headers=h, json={"name": "요인 패널", "source_dataset_id": source["id"], "source_hash": source["content_hash"]}).json()
    p = client.post(BASE+"/pipelines", headers=h, json={"id": "test_period_panel", "name": "요인 패널 학습", "model_type": "factor_analysis",
        "dataset_id": ds["id"], "dataset_hash": ds["content_hash"]}).json()
    job = client.post(BASE+"/pipelines/"+p["id"]+"/run", headers=h).json()
    deadline = time.monotonic()+10
    while time.monotonic() < deadline and job["state"] not in ("saved", "failed"):
        job = client.get(BASE+"/jobs/"+job["id"], headers=h).json()
    url = BASE+"/instances/"+p["instance_id"]+"/versions/"+job["version_id"]
    assert client.post(url+"/activate", headers=h).status_code == 200
    context = client.get(url+"/inputs", headers=h).json()
    predicted = client.post(url+"/predict", headers=h, json=body(context)).json()
    last = client.post(url+"/factor-period", headers=h, json=period_body(context, 202412, 202412)).json()
    assert last["months"][0]["prediction"] == pytest.approx(predicted["prediction"])
    year = client.post(url+"/factor-period", headers=h, json=period_body(context, 201601, 201612, "prev_year")).json()
    assert year["summary"]["used_months"] == 12 and all(m["actual"] is not None for m in year["months"])
    assert_identity(year)
    # 2016-01 이전은 전년 기준 달이 없다, 관측 기간 밖은 거절
    assert client.post(url+"/factor-period", headers=h, json=period_body(context, 201501, 201512, "prev_year")).status_code == 422
    assert client.post(url+"/factor-period", headers=h, json=period_body(context, 201412, 201501)).status_code == 422
    # 옛 지역 인구 추이 원천 모델(행에 기준월 없음)은 기간 분석을 하지 않는다
    legacy_h, _, _, _, legacy_url = train(client, auth_headers, kind="factor_analysis")
    legacy = client.get(legacy_url+"/inputs", headers=legacy_h).json()
    response = client.post(legacy_url+"/factor-period", headers=legacy_h, json=period_body(legacy, 202412, 202412))
    assert response.status_code == 422 and "지원하지 않습니다" in response.json()["detail"]
