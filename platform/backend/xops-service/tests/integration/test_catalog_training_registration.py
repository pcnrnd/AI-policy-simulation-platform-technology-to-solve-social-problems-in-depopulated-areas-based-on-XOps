"""실제 격리 SQLite 표 → 기존 SQL 어댑터 → 불변 등록본. 네트워크 PG 왕복과 구분한다."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from src.auth.jwt import issue_jwt
from src.core import db
from src.core.settings import get_settings
from src.dataops.adapters import InMemoryAdapter, SampleAdapter
from src.dataops.backends import SqlAdapter
from src.dataops.results import ExecutionResult
from src.dataops import service as data_service
from src.modelops import catalog_datasets, datasets, store

BASE = "/api/v3/modelops"


class FixtureConnection:
    """SQL·fetchmany는 sqlite3가 실제 실행하고 커서 수명만 psycopg 형태로 맞춘다."""
    def __init__(self, path):
        self.connection = sqlite3.connect(path)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.connection.close()

    def cursor(self):
        connection = self.connection

        class Cursor:
            def __enter__(self):
                self.raw = connection.cursor()
                return self.raw

            def __exit__(self, *_):
                self.raw.close()

        return Cursor()


@pytest.fixture()
def catalog_fixture(client, auth_headers, monkeypatch, tmp_path):
    h = {**auth_headers, "X-Data-Mode": "real"}
    db._conn().execute("DELETE FROM mo_assets")
    db._conn().commit()
    root = tmp_path / "managed"
    monkeypatch.setattr(get_settings(), "model_artifact_dir", root)
    original = tmp_path / "catalog-origin.sqlite"
    with sqlite3.connect(original) as conn:
        conn.execute("CREATE TABLE selected_a(row_id INTEGER, value REAL, label TEXT)")
        conn.execute("CREATE TABLE selected_b(row_id INTEGER, value REAL, label TEXT)")
        conn.executemany("INSERT INTO selected_a VALUES (?,?,?)", [(i, float(i), f"원본A-{i}") for i in range(1, 451)])
        conn.executemany("INSERT INTO selected_b VALUES (?,?,?)", [(i, float(1000 + i), f"원본B-{i}") for i in range(1, 51)])
    schemas = []
    for suffix in ("a", "b"):
        body = {"id": "fixture_catalog_" + suffix, "label": "선택 원본 " + suffix.upper(),
                "source": "RDB · PostgreSQL", "object": "selected_" + suffix,
                "columns": [{"name": "row_id", "type": "INTEGER"}, {"name": "value", "type": "REAL"},
                            {"name": "label", "type": "TEXT"}],
                "range": {"column": "row_id", "from": 1, "to": 450}}
        response = client.post("/api/v3/dataops/catalog", headers=h, json=body)
        assert response.status_code == 201
        schemas.append(response.json())
    monkeypatch.setattr(SqlAdapter, "_connect", lambda _: FixtureConnection(original))
    factory = lambda _: SqlAdapter("fixture SQL", "", timeout=1, max_rows=200)
    monkeypatch.setattr(catalog_datasets, "get_adapter", factory)
    monkeypatch.setattr(data_service, "get_adapter", factory)
    yield {"headers": h, "schemas": schemas, "original": original, "root": root}
    for schema in schemas:
        db.delete_user_source(schema["id"])
    db._conn().execute("DELETE FROM mo_assets")
    db._conn().commit()


def prepare(client, fixture, **query):
    body = {"source_id": fixture["schemas"][0]["id"], **query}
    return client.post(BASE + "/datasets/prepare", headers=fixture["headers"], json=body)


def register(client, fixture, prepared, **changes):
    body = {"source_id": prepared["source"]["id"], "filter": prepared["query"]["filter"],
            "sort": prepared["query"]["sort"], "columns": prepared["query"]["columns"],
            "source_hash": prepared["content_hash"], "name": "카탈로그 학습 자료", "purpose": "training", **changes}
    return client.post(BASE + "/datasets/from-catalog", headers=fixture["headers"], json=body)


def assert_no_registration(fixture):
    assert store.list_assets("real", "dataset") == []
    assert store.list_assets("real", "audit") == []
    assert not list(fixture["root"].rglob("*.json"))
    assert not list(fixture["root"].rglob("*.pending"))


def test_catalog_prepare_is_read_only_and_registers_all_rows_beyond_preview(client, catalog_fixture):
    f = catalog_fixture
    original_hash = hashlib.sha256(f["original"].read_bytes()).hexdigest()
    # 원본 목록 조회는 학습 등록을 만들지 않는다.
    catalog = client.get("/api/v3/dataops/catalog", headers=f["headers"]).json()
    assert f["schemas"][0]["id"] in {row["id"] for row in catalog}
    assert client.get(BASE + "/datasets", headers=f["headers"]).json()["data"] == []
    preview = client.get("/api/v3/dataops/" + f["schemas"][0]["id"], headers=f["headers"], params={"page_size": 20}).json()
    assert preview["source_kind"] == "database" and len(preview["rows"]) == 20
    assert preview["pagination"]["total"] == 450
    prepared = prepare(client, f).json()
    assert prepared["complete"] and prepared["row_count"] == 450
    assert prepared["preview_count"] == len(prepared["preview_rows"]) == 5
    assert prepared["query"]["selection"] == "all_matching_rows"
    assert_no_registration(f)
    response = register(client, f, prepared)
    assert response.status_code == 201
    registered = response.json()
    assert registered["row_count"] == 450 and registered["source_name"] == "선택 원본 A"
    assert registered["purpose"] == "training" and registered["status"] == "registered"
    assert registered["training_compatible"] is False
    path = Path(registered["private_path"])
    document = json.loads(path.read_text(encoding="utf-8"))
    assert len(document["rows"]) == registered["row_count"]
    assert {row["row_id"] for row in document["rows"]} == set(range(1, 451))
    assert document["schema"] == prepared["schema"] and document["query"] == prepared["query"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == registered["private_hash"]
    assert hashlib.sha256(f["original"].read_bytes()).hexdigest() == original_hash
    assert client.get(BASE + "/datasets", headers=f["headers"]).json()["data"] == [registered]
    detail = client.get(BASE + "/datasets/" + registered["id"], headers=f["headers"], params={"include_rows": True}).json()
    assert detail["source_dataset_id"] == f["schemas"][0]["id"]
    assert len(detail["rows"]) == 100 and detail["rows_truncated"] is True
    assert "private_path" not in detail
    # 일반 표형 자료 등록은 성공하지만, 기존 모델 학습 DTO로 오인하지 않는다.
    incompatible = client.post(BASE + "/instances", headers=f["headers"], json={
        "model_type": "simulation", "dataset_id": registered["id"], "dataset_hash": registered["content_hash"]})
    assert incompatible.status_code == 400 and "매핑" in incompatible.json()["detail"]
    assert store.list_assets("real", "instance") == []
    # 원본이 변경돼도 등록본의 내용은 고정된다.
    frozen_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    with sqlite3.connect(f["original"]) as conn:
        conn.execute("UPDATE selected_a SET value=9999 WHERE row_id=1")
    assert datasets.load("real", registered["id"])[1] == document
    assert hashlib.sha256(path.read_bytes()).hexdigest() == frozen_hash


def test_two_selected_sources_conditions_projection_and_source_identity(client, catalog_fixture):
    f = catalog_fixture
    first = prepare(client, f, filter="value >= 40", sort="value", columns=["row_id", "value"]).json()
    assert first["row_count"] == 411
    assert all(set(row) == {"row_id", "value"} for row in first["preview_rows"])
    second = prepare(client, f, source_id=f["schemas"][1]["id"]).json()
    assert second["row_count"] == 50 and second["content_hash"] != first["content_hash"]
    # 다른 원본·조건으로 해시만 재사용해서 등록할 수 없다.
    assert register(client, f, first, source_id=f["schemas"][1]["id"]).status_code == 409
    assert register(client, f, first, filter="value >= 41").status_code == 409
    assert_no_registration(f)
    a = register(client, f, first).json()
    b = register(client, f, second, purpose="validation").json()
    assert a["source_dataset_id"] != b["source_dataset_id"] and a["row_count"] == 411 and b["row_count"] == 50
    assert {row["id"] for row in client.get(BASE + "/datasets", headers=f["headers"]).json()["data"]} == {a["id"], b["id"]}
    assert all(row["value"] >= 40 for row in datasets.load("real", a["id"])[1]["rows"])


@pytest.mark.parametrize(("query", "status"), [
    ({"source_id": "missing-source"}, 404), ({"filter": "value > 99999"}, 422),
    ({"filter": "missing = 1"}, 400), ({"filter": "value > 1 OR 1=1"}, 400),
    ({"sort": "missing"}, 400), ({"columns": ["missing"]}, 400),
    ({"columns": ["value", "value"]}, 400), ({"columns": []}, 422),
    ({"page_size": 20}, 422),
])
def test_invalid_or_empty_query_writes_nothing(client, catalog_fixture, query, status):
    assert prepare(client, catalog_fixture, **query).status_code == status
    # 준비를 건너뛴 직접 등록도 같은 입력 경계로 검증한다.
    body = {"source_id": catalog_fixture["schemas"][0]["id"], "name": "x", "purpose": "training", "source_hash": "a" * 64, **query}
    assert client.post(BASE + "/datasets/from-catalog", headers=catalog_fixture["headers"], json=body).status_code == status
    assert_no_registration(catalog_fixture)


def test_source_change_and_bounded_read_truncation_write_nothing(client, catalog_fixture, monkeypatch):
    f = catalog_fixture
    prepared = prepare(client, f).json()
    with sqlite3.connect(f["original"]) as conn:
        conn.execute("DELETE FROM selected_a WHERE row_id=1")
    assert register(client, f, prepared).status_code == 409
    assert_no_registration(f)
    monkeypatch.setattr(get_settings(), "training_dataset_max_rows", 20)
    assert prepare(client, f).status_code == 413
    assert register(client, f, prepared).status_code == 413
    assert_no_registration(f)


@pytest.mark.parametrize(("sqlstate", "status", "message"), [
    ("42P01", 422, "실제 자료"), ("42703", 422, "컬럼"), ("42501", 403, "권한"),
])
def test_storage_errors_are_actionable_without_driver_secrets(client, catalog_fixture, monkeypatch, sqlstate, status, message):
    f = catalog_fixture
    prepared = prepare(client, f).json()
    class StorageError(RuntimeError):
        pass
    error = StorageError("private-connection-secret")
    error.sqlstate = sqlstate
    def fail(*_args, **_kwargs):
        raise error
    monkeypatch.setattr(SqlAdapter, "read_complete", fail)
    response = register(client, f, prepared)
    assert response.status_code == status and message in response.json()["detail"]
    assert "private-connection-secret" not in response.text
    assert_no_registration(f)


@pytest.mark.parametrize("kind", ["fallback", "sample", "exception", "partial", "not_executed", "missing_column", "nested", "nonfinite"])
def test_unverified_adapter_data_is_never_registered(client, catalog_fixture, monkeypatch, kind):
    f = catalog_fixture
    prepared = prepare(client, f).json()
    if kind == "fallback":
        monkeypatch.setattr(catalog_datasets, "get_adapter", lambda _: InMemoryAdapter("fallback"))
    elif kind == "sample":
        monkeypatch.setattr(catalog_datasets, "get_adapter", lambda _: SampleAdapter("sample", [{"row_id": 1, "value": 1, "label": "sample"}]))
    else:
        def invalid(*_args, **_kwargs):
            if kind == "exception":
                raise RuntimeError("private-dsn-must-never-leak")
            if kind == "not_executed":
                return ExecutionResult.not_executed("private-dsn-must-never-leak")
            row = {"row_id": 1, "value": 1, "label": "x"}
            if kind == "missing_column":
                row.pop("value")
            elif kind == "nested":
                row["value"] = {"nested": 1}
            elif kind == "nonfinite":
                row["value"] = float("nan")
            return ExecutionResult(rows=[row], total=2 if kind == "partial" else 1, executed=True)
        monkeypatch.setattr(SqlAdapter, "read_complete", invalid)
    response = register(client, f, prepared)
    assert response.status_code in (409, 422, 503)
    assert "private-dsn-must-never-leak" not in response.text
    assert_no_registration(f)


def test_auth_mode_and_unsupported_format_boundaries(client, catalog_fixture, monkeypatch):
    f = catalog_fixture
    prepared = prepare(client, f).json()
    body = {"source_id": f["schemas"][0]["id"]}
    assert client.post(BASE + "/datasets/prepare", json=body).status_code == 401
    assert client.post(BASE + "/datasets/from-catalog", json={**body, "name": "x", "purpose": "training", "source_hash": prepared["content_hash"]}).status_code == 401
    mock = {**f["headers"], "X-Data-Mode": "mock"}
    assert client.post(BASE + "/datasets/prepare", headers=mock, json=body).status_code == 404
    readonly = {"Authorization": "Bearer " + issue_jwt(scope="data:read"), "X-Data-Mode": "real"}
    assert client.post(BASE + "/datasets/prepare", headers=readonly, json=body).status_code == 200
    write = {**body, "name": "x", "purpose": "training", "source_hash": prepared["content_hash"]}
    assert client.post(BASE + "/datasets/from-catalog", headers=readonly, json=write).status_code == 401
    # 지원하지 않는 형식과 누락된 범위 컬럼도 저장 전에 설명한다.
    schema = db.get_user_source(body["source_id"])
    for changed in ({**schema, "source": "NoSQL · MongoDB"},
                    {**schema, "columns": [{"name": "value", "type": "JSONB"}]},
                    {**schema, "range": {"column": "missing", "from": 1, "to": 2}}):
        class Catalog:
            def get(self, _):
                return changed
        monkeypatch.setattr(catalog_datasets, "get_catalog", lambda: Catalog())
        response = client.post(BASE + "/datasets/prepare", headers=f["headers"], json=body)
        assert response.status_code in (400, 422)
    assert_no_registration(f)


def test_duplicate_registration_is_serialized_and_purpose_is_preserved(client, catalog_fixture):
    f = catalog_fixture
    prepared = prepare(client, f).json()
    with ThreadPoolExecutor(2) as pool:
        responses = list(pool.map(lambda _: register(client, f, prepared), range(2)))
    assert all(response.status_code == 201 for response in responses)
    assert len({response.json()["id"] for response in responses}) == 1
    assert len(store.list_assets("real", "dataset")) == len(store.list_assets("real", "audit")) == 1
    assert len(list(f["root"].rglob("*.json"))) == 1
    second_purpose = register(client, f, prepared, purpose="test")
    assert second_purpose.status_code == 201 and second_purpose.json()["id"] != responses[0].json()["id"]


def test_storage_failure_rolls_back_asset_audit_and_owned_file(client, catalog_fixture, monkeypatch):
    f = catalog_fixture
    prepared = prepare(client, f).json()
    calls = 0
    def guard():
        nonlocal calls
        calls += 1
        if calls == 3:  # 파일 생성·dataset INSERT 뒤 audit INSERT 전에 실패한다.
            raise RuntimeError("write failure")
    # write_guard context does not cross TestClient's thread; inject at the common guard boundary.
    monkeypatch.setattr(store, "check_write_guard", guard)
    with pytest.raises(RuntimeError, match="write failure"):
        register(client, f, prepared)
    assert_no_registration(f)


def test_registered_detail_is_mode_scoped_and_deletion_keeps_original(client, catalog_fixture):
    f = catalog_fixture
    registered = register(client, f, prepare(client, f).json()).json()
    before = hashlib.sha256(f["original"].read_bytes()).hexdigest()
    url = BASE + "/datasets/" + registered["id"]
    assert client.get(url, headers={**f["headers"], "X-Data-Mode": "mock"}).status_code == 404
    assert client.get(url).status_code == 401
    assert client.delete(url, headers=f["headers"]).status_code == 200
    assert not Path(registered["private_path"]).exists()
    assert hashlib.sha256(f["original"].read_bytes()).hexdigest() == before


def test_mock_catalog_registration_stays_mock_and_obeys_range(client, catalog_fixture, monkeypatch):
    f = catalog_fixture
    schema = {**f["schemas"][0], "mode": "mock", "is_seed": True,
              "range": {"column": "row_id", "from": 2, "to": 3}}
    rows = [{"row_id": i, "value": float(i * 10), "label": "예시"} for i in range(1, 5)]
    class Catalog:
        def get(self, _):
            return schema
    monkeypatch.setattr(catalog_datasets, "get_catalog", lambda: Catalog())
    monkeypatch.setattr(catalog_datasets, "get_adapter", lambda _: SampleAdapter("mock", rows))
    h = {**f["headers"], "X-Data-Mode": "mock"}
    prepared = client.post(BASE + "/datasets/prepare", headers=h, json={"source_id": schema["id"]}).json()
    assert prepared["mode"] == "mock" and prepared["row_count"] == 2
    assert {row["row_id"] for row in prepared["preview_rows"]} == {2, 3}
    response = client.post(BASE + "/datasets/from-catalog", headers=h, json={"source_id": schema["id"],
        "name": "예시 등록", "purpose": "training", "source_hash": prepared["content_hash"]})
    assert response.status_code == 201 and response.json()["mode"] == "mock"
    assert store.list_assets("real", "dataset") == []
    assert client.post(BASE + "/datasets/prepare", headers=f["headers"], json={"source_id": schema["id"]}).status_code == 404
