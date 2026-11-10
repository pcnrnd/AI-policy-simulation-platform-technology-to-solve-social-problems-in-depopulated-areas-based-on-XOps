"""선택한 카탈로그의 전체 조건 결과를 검증하고 변경 불가능한 학습 자료로 등록한다.

준비는 읽기 전용이다. 등록 때 같은 조건을 재조회하여 준비 해시와 대조한다. 기존
realdata 스냅샷·학습 계약은 건드리지 않으며 일반 표형 등록과 모델 호환 검사를 구분한다.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from src.core.data_mode import source_mode
from src.core.db import _conn
from src.core.settings import get_settings
from src.dataops.adapters import SampleAdapter, get_adapter
from src.dataops.backends import SqlAdapter, UnsupportedTabularError
from src.dataops.catalog import get_catalog
from src.dataops.query_builder import build_query
from src.dataops.results import ExecutionRequest
from src.dataops.safety import assert_safe_filter, assert_safe_schema, assert_safe_sort, assert_safe_sql
from src.modelops import datasets, store

def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _read(mode: str, body: dict[str, Any]) -> dict[str, Any]:
    max_rows = get_settings().training_dataset_max_rows
    schema = get_catalog().get(body["source_id"])
    if source_mode(schema) != mode:
        raise HTTPException(404, "현재 모드에서 사용할 카탈로그 원본을 선택하세요.")
    assert_safe_schema(schema)
    names = [column["name"] for column in schema["columns"]]
    selected = body.get("columns") or names
    if len(names) != len(set(names)) or len(selected) != len(set(selected)) or not set(selected) <= set(names):
        raise HTTPException(400, "원본 스키마에 있는 컬럼을 중복 없이 선택하세요.")
    if schema.get("range") and schema["range"]["column"] not in names:
        raise HTTPException(400, "조회 범위 컬럼이 원본 스키마에 없습니다.")
    filter_expr = (body.get("filter") or "").strip()
    sort = body.get("sort") or ""
    assert_safe_filter(filter_expr, set(names))
    assert_safe_sort(sort, set(names))
    columns = [next(column for column in schema["columns"] if column["name"] == name) for name in selected]
    if any(any(marker in str(column.get("type", "")).upper() for marker in ("JSON", "ARRAY", "[]", "GEOMETRY", "GEOGRAPHY"))
           for column in columns):
        raise HTTPException(422, "중첩·공간 형식 컬럼은 그대로 등록할 수 없습니다. 표형 컬럼을 선택하거나 변환한 원본을 사용하세요.")
    projected = {**schema, "columns": columns}
    query = build_query(method="GET", schema=projected, filter_expr=filter_expr, sort=sort, page=1, page_size=max_rows + 1)
    if query.lang != "SQL":
        raise HTTPException(422, "이 원본 형식은 학습 자료 등록을 지원하지 않습니다. 표형 데이터로 변환한 원본을 선택하세요.")
    assert_safe_sql(query.text)
    adapter = get_adapter(projected)
    request = ExecutionRequest(schema=projected, method="GET", sql=query.text, filter_expr=filter_expr,
                               sort=sort, page=1, page_size=max_rows + 1)
    try:
        if isinstance(adapter, SqlAdapter) and mode == "real":
            result = adapter.read_complete(request, max_rows=max_rows)
        elif isinstance(adapter, SampleAdapter) and mode == "mock":
            result = adapter.execute(request)
        else:
            raise HTTPException(503, "원본 저장소에 연결되지 않아 실제 자료를 읽지 못했습니다. 연결 상태와 원본 적재 여부를 확인하세요.")
    except HTTPException:
        raise
    except UnsupportedTabularError as exc:
        raise HTTPException(422, "중첩·바이너리 값은 그대로 등록할 수 없습니다. 표형 데이터로 변환한 원본을 선택하세요.") from exc
    except Exception as exc:
        # 드라이버 오류에는 접속 정보가 포함될 수 있어 원문을 응답·감사 기록에 싣지 않는다.
        if getattr(exc, "sqlstate", None) == "42P01":
            raise HTTPException(422, "선택한 원본의 실제 자료가 아직 준비되지 않았습니다. 원본 적재 상태를 확인한 뒤 다시 준비하세요.") from exc
        if getattr(exc, "sqlstate", None) == "42703":
            raise HTTPException(422, "실제 원본과 카탈로그의 컬럼이 다릅니다. 스키마를 확인한 뒤 다시 준비하세요.") from exc
        if getattr(exc, "sqlstate", None) == "42501":
            raise HTTPException(403, "선택한 원본을 조회할 권한이 없습니다. 원본 접근 권한을 확인하세요.") from exc
        raise HTTPException(503, "원본을 조회하지 못했습니다. 저장소 연결·접근 권한과 원본 테이블·컬럼을 확인하세요.") from exc
    if not result.executed or result.total is None:
        raise HTTPException(503, "원본의 실제 조회 결과를 확인하지 못해 등록할 수 없습니다.")
    if result.total > max_rows or len(result.rows) > max_rows:
        raise HTTPException(413, f"조회 결과가 등록 상한 {max_rows:,}행을 넘습니다. 조회 조건을 좁힌 뒤 전체 결과를 다시 준비하세요.")
    if result.total != len(result.rows):
        raise HTTPException(409, "조회 결과가 일부만 반환되어 등록하지 않았습니다. 전체 결과를 다시 준비하세요.")
    if not result.rows:
        raise HTTPException(422, "선택한 원본과 조회 조건에 해당하는 데이터가 없습니다. 원본 적재 여부와 조건을 확인하세요.")
    rows = []
    for row in result.rows:
        if not isinstance(row, dict) or not set(selected) <= set(row):
            raise HTTPException(422, "실제 조회 컬럼이 원본 스키마와 다릅니다. 스키마를 확인하세요.")
        projected_row = {name: row[name] for name in selected}
        if any(value is not None and (not isinstance(value, (str, bool, int, float))
               or isinstance(value, float) and not math.isfinite(value)) for value in projected_row.values()):
            raise HTTPException(422, "중첩 자료 또는 지원하지 않는 값이 있습니다. 문자열·숫자·불리언·빈값으로 구성된 표형 원본을 선택하세요.")
        rows.append(projected_row)
    # 행 순서만 달라진 같은 결과는 같은 버전으로 본다. 중복 행은 그대로 보존한다.
    rows.sort(key=_json)
    source = {"id": schema["id"], "name": schema.get("label") or schema["id"],
              "storage": schema.get("source"), "object": schema["object"], "lineage": schema.get("lineage")}
    scope = {"filter": filter_expr, "sort": sort, "range": schema.get("range"), "columns": selected,
             "selection": "all_matching_rows"}
    document = {"format": "catalog-tabular-v1", "mode": mode, "source": source,
                "query": scope, "schema": columns, "rows": rows, "row_count": len(rows)}
    document["content_hash"] = _hash(document)
    return document


def prepare(mode: str, body: dict[str, Any]) -> dict[str, Any]:
    document = _read(mode, body)
    return {**{key: value for key, value in document.items() if key != "rows"},
            "preview_rows": document["rows"][:5], "preview_count": min(5, document["row_count"]),
            "complete": True, "prepared_at": store.now(), "max_rows": get_settings().training_dataset_max_rows}


def register(mode: str, body: dict[str, Any], actor: str) -> dict[str, Any]:
    document = _read(mode, body)
    if document["content_hash"] != body["source_hash"]:
        raise HTTPException(409, "준비한 뒤 원본이나 조회 조건이 바뀌었습니다. 현재 선택으로 다시 준비하세요.")
    with datasets._register_lock:
        conn = _conn()
        path: Path | None = None
        pending: Path | None = None
        pending_created = promoted = False
        try:
            # DB 쓰기 잠금은 프로세스 간 중복 등록도 직렬화한다. 검증 실패 시 여기 오지 않는다.
            conn.execute("BEGIN IMMEDIATE")
            existing = next((row for row in store.list_assets(mode, "dataset")
                             if row.get("dataset_kind") == "catalog_tabular"
                             and row.get("content_hash") == document["content_hash"]
                             and row.get("purpose") == body["purpose"] and datasets._intact(row)), None)
            if existing:
                conn.rollback()
                return existing
            store.check_write_guard()
            dataset_id = store.new_id()
            created_at = store.now()
            root = get_settings().model_artifact_dir / "modelops" / mode / "datasets"
            root.mkdir(parents=True, exist_ok=True)
            path = root / (dataset_id + ".json")
            pending = root / (dataset_id + ".pending")
            if path.exists():
                raise HTTPException(409, "등록본 식별자가 이미 있습니다. 다시 등록하세요.")
            raw = _json(document).encode("utf-8")
            with pending.open("xb") as handle:
                pending_created = True
                handle.write(raw)
            pending.replace(path)
            promoted = True
            registered = {"id": dataset_id, "dataset_id": dataset_id, "mode": mode, "source": mode,
                          "dataset_kind": "catalog_tabular", "name": body["name"].strip(), "purpose": body["purpose"],
                          "source_dataset_id": body["source_id"], "source_name": document["source"]["name"],
                          "provenance": document["source"], "query": document["query"], "schema": document["schema"],
                          "content_hash": document["content_hash"], "schema_version": document["format"],
                          "row_count": document["row_count"], "created_at": created_at, "status": "registered",
                          "training_compatible": False, "features": [], "region_id": None, "target": None,
                          "observed_from": None, "observed_to": None,
                          "private_path": str(path), "private_hash": hashlib.sha256(raw).hexdigest()}
            audit = {"id": store.new_id(), "mode": mode, "source": mode, "created_at": created_at,
                     "action": "dataset_registered", "asset_id": dataset_id, "actor": actor,
                     "source_id": body["source_id"], "dataset_hash": document["content_hash"]}
            for kind, value in (("dataset", registered), ("audit", audit)):
                store.check_write_guard()
                conn.execute("INSERT INTO mo_assets(mode,kind,id,body_json,created_at) VALUES (?,?,?,?,?)",
                             (mode, kind, value["id"], _json(value), created_at))
            conn.commit()
            return registered
        except Exception:
            conn.rollback()
            # 작업에서 만든 유일한 경로만 정리한다. 원본·기존 등록본은 수정하지 않는다.
            for owned, created in ((pending, pending_created), (path, promoted)):
                if owned is not None and created:
                    owned.unlink(missing_ok=True)
            raise


def detail(mode: str, dataset_id: str, include_rows: bool) -> dict[str, Any]:
    dataset, document = datasets.load(mode, dataset_id)
    data = {key: value for key, value in dataset.items() if key not in ("private_path", "private_hash")}
    if include_rows:
        rows = document.get("rows", [])
        data.update({"rows": rows[:100], "rows_truncated": len(rows) > 100})
    return data
