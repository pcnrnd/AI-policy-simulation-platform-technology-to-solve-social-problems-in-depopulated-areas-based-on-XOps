"""원격 예측 모델(Chronos-2) 비교 — 원격 서버 호출은 흉내 낸다. 값은 모두 합성이다."""
from __future__ import annotations

import io
import json
import logging
import time
from dataclasses import asdict
from urllib.error import HTTPError, URLError

import pytest

from src.core.db import _conn
from src.core.settings import get_settings
from src.modelops import remote_forecast, store
from src.realdata import snapshot
from src.realdata.features import add_month

BASE = "/api/v3/modelops"
URL = "http://remote.test:18100"
TOKEN = "t" * 40  # 합성 토큰


@pytest.fixture(autouse=True)
def clean_assets():
    _conn().execute("DELETE FROM mo_assets")
    _conn().commit()
    yield
    _conn().execute("DELETE FROM mo_assets")
    _conn().commit()


class FakeRemote:
    """urlopen 대역 — 받은 요청을 기록하고 계열마다 [low, median, high]를 돌려준다."""

    def __init__(self, *, status: int = 200, exc: Exception | None = None, quantiles: list[float] | None = None):
        self.status, self.exc, self.quantiles, self.calls = status, exc, quantiles, []

    def __call__(self, request, *, timeout):
        payload = json.loads(request.data)
        self.calls.append({"url": request.full_url, "timeout": timeout, "token": request.get_header("X-forecast-token"), "json": payload})
        if self.exc:
            raise self.exc
        if self.status != 200:
            raise HTTPError(request.full_url, self.status, "error", {}, io.BytesIO(b"{}"))
        rows = [[self.quantiles or [index + 1.0, index + 2.0, index + 3.0]] for index in range(len(payload["series"]))]
        return io.BytesIO(json.dumps({"quantiles": rows}).encode())


@pytest.fixture()
def remote(monkeypatch):
    def install(fake: FakeRemote | None = None, *, configured: bool = True) -> FakeRemote:
        fake = fake or FakeRemote()
        settings = get_settings()
        monkeypatch.setattr(settings, "remote_forecast_url", URL if configured else "")
        monkeypatch.setattr(settings, "remote_forecast_token", type(settings.remote_forecast_token)(TOKEN if configured else ""))
        monkeypatch.setattr(remote_forecast, "urlopen", fake)
        return fake
    return install


def train(client, auth, source_id: str, kind: str, mode: str = "mock"):
    h = {**auth, "X-Data-Mode": mode}
    sources = client.get(BASE + "/available-datasets", headers=h).json()["data"]
    source = next(row for row in sources if row["id"] == source_id)
    ds = client.post(BASE + "/datasets", headers=h, json={"name": "비교", "source_dataset_id": source["id"], "source_hash": source["content_hash"]}).json()
    p = client.post(BASE + "/pipelines", headers=h, json={"id": "test_remote_" + kind, "name": "비교", "model_type": kind,
                                                         "dataset_id": ds["id"], "dataset_hash": ds["content_hash"]}).json()
    job = client.post(BASE + "/pipelines/" + p["id"] + "/run", headers=h).json()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and job["state"] not in ("saved", "failed"):
        job = client.get(BASE + "/jobs/" + job["id"], headers=h).json()
    assert job["state"] == "saved"
    url = BASE + "/instances/" + p["instance_id"] + "/versions/" + job["version_id"]
    assert client.post(url + "/activate", headers=h).status_code == 200
    return h, url


def baseline(client, h, url, dong_code=None):
    context = client.get(url + "/inputs", headers=h, params={"dong_code": dong_code} if dong_code else None).json()
    proof = context["provenance"]
    body = {**{key: proof[key] for key in ("region_id", "target", "dataset_id", "dataset_hash", "dong_code")}, "adjustments": {}}
    return client.post(url + "/predict", headers=h, json=body).json()


def test_demo_panel_sends_one_series_with_token_and_returns_range_beside_ridge(client, auth_headers, remote):
    fake = remote()
    h, url = train(client, auth_headers, "mock_factor_namwon", "factor_analysis")
    result = client.get(url + "/forecast-compare", headers=h).json()
    ridge = baseline(client, h, url)
    assert result["mode"] == result["source"] == "mock"
    assert result["ridge"]["prediction"] == pytest.approx(ridge["baseline_prediction"])
    assert result["provenance"]["prediction_period"] == 202412
    assert result["remote"] == {"status": "ok", **remote_forecast.REMOTE_MODEL, "quantile_levels": [0.1, 0.5, 0.9],
                                "low": 1.0, "median": 2.0, "high": 3.0}
    (call,) = fake.calls
    assert call["url"] == URL + "/forecast" and call["token"] == TOKEN
    assert call["timeout"] == remote_forecast.TIMEOUT_SECONDS
    # 예측월(2024-12) 앞 119개월 계열 하나, 1개월 앞, 분위수 3개.
    assert len(call["json"]["series"]) == 1 and len(call["json"]["series"][0]) == 119
    assert call["json"]["prediction_length"] == 1 and call["json"]["quantile_levels"] == [0.1, 0.5, 0.9]
    catalog = client.get(BASE + "/catalog", headers=h).json()
    assert catalog["remote_types"] == [{**remote_forecast.REMOTE_MODEL, "available": True}]


@pytest.mark.parametrize("fake", [
    FakeRemote(status=401), FakeRemote(status=500), FakeRemote(exc=TimeoutError("timed out")),
    FakeRemote(exc=URLError("refused")), FakeRemote(quantiles=[3.0, 2.0, 1.0]), FakeRemote(quantiles=[1.0, float("nan"), 3.0]),
])
def test_remote_failure_keeps_ridge_and_never_logs_token(client, auth_headers, remote, caplog, fake):
    remote(fake)
    h, url = train(client, auth_headers, "mock_factor_namwon", "factor_analysis")
    with caplog.at_level(logging.WARNING):
        result = client.get(url + "/forecast-compare", headers=h).json()
    assert result["remote"] == {"status": "unavailable"}
    assert result["ridge"]["prediction"] == pytest.approx(baseline(client, h, url)["baseline_prediction"])
    assert TOKEN not in caplog.text and "remote forecast unavailable" in caplog.text


def test_unconfigured_remote_is_not_called_and_catalog_says_unavailable(client, auth_headers, remote):
    fake = remote(configured=False)
    h, url = train(client, auth_headers, "mock_factor_namwon", "factor_analysis")
    assert client.get(url + "/forecast-compare", headers=h).json()["remote"] == {"status": "unavailable"}
    assert fake.calls == []
    assert client.get(BASE + "/catalog", headers=h).json()["remote_types"][0]["available"] is False
    assert client.get(url + "/forecast-compare").status_code == 401


def test_legacy_region_dataset_has_no_remote_series(client, auth_headers, remote):
    fake = remote()
    h, url = train(client, auth_headers, "mock_namwon", "simulation")
    assert client.get(url + "/forecast-compare", headers=h).json()["remote"] == {"status": "unavailable"}
    assert fake.calls == []


def test_real_snapshot_sends_dong_batch_with_gaps_and_picks_selected_dong(client, auth_headers, remote, monkeypatch, tmp_path):
    fake = remote()
    first, second = "45190250", "45190310"
    rows = [{"base_ym": add_month(202101, i), "dong_code": code, "y": base + 3 * i + (i % 3), "nonlocal_visitors": base + 3 * i,
             "local_visitors": 200, "foreign_visitors": 1}
            for code, base in ((first, 100), (second, 300)) for i in range(36) if not (code == second and 5 <= i < 8)]
    record = snapshot.DatasetRecord(dataset_id="ds-testremote", target="nonlocal_visitors", model_id=snapshot.TARGETS["nonlocal_visitors"],
        spec={"target": "nonlocal_visitors", "model_id": snapshot.TARGETS["nonlocal_visitors"]}, quality={}, observed_from=202101,
        observed_to=202312, row_count=len(rows), excluded_rows={}, content_hash=snapshot._content_hash(rows),
        file_path=str(tmp_path / "original.json"), created_at=store.now(), rows=rows)
    (tmp_path / "original.json").write_text(json.dumps(asdict(record)), encoding="utf-8")
    monkeypatch.setattr(snapshot, "list_datasets", lambda: [asdict(record)])
    monkeypatch.setattr(snapshot, "load_dataset", lambda _: record)
    h, url = train(client, auth_headers, "ds-testremote", "simulation", mode="real")
    result = client.get(url + "/forecast-compare", headers=h, params={"dong_code": second}).json()
    assert result["provenance"]["prediction_period"] == 202401 and result["provenance"]["dong_code"] == second
    assert result["ridge"]["prediction"] == pytest.approx(baseline(client, h, url, second)["baseline_prediction"])
    sent = fake.calls[0]["json"]["series"]
    # 대응표 순서의 동 묶음(관측 없는 동 제외), 2021-01~2023-12 36개월, 공백 달은 None.
    assert len(sent) == 2 and all(len(row) == 36 for row in sent)
    assert sent[1][5:8] == [None, None, None] and sent[0][0] == 100
    assert (result["remote"]["low"], result["remote"]["median"]) == (2.0, 3.0)  # 두 번째 계열 값
