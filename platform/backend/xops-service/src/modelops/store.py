from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import HTTPException

from src.core.db import _conn

MODEL_TYPES = {"factor_analysis": "요인분석", "simulation": "시뮬레이션"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str = "modelops_") -> str:
    return prefix + uuid4().hex


def init_db() -> None:
    _conn().execute(
        "CREATE TABLE IF NOT EXISTS mo_assets ("
        "mode TEXT NOT NULL, kind TEXT NOT NULL, id TEXT NOT NULL, "
        "body_json TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY(mode, kind, id))"
    )
    _conn().commit()


def get(mode: str, kind: str, asset_id: str) -> dict[str, Any]:
    row = _conn().execute(
        "SELECT body_json FROM mo_assets WHERE mode=? AND kind=? AND id=?", (mode, kind, asset_id)
    ).fetchone()
    if row is None:
        raise HTTPException(404, "선택한 항목을 현재 모드에서 찾을 수 없습니다.")
    return json.loads(row[0])


def list_assets(mode: str, kind: str) -> list[dict[str, Any]]:
    return [json.loads(row[0]) for row in _conn().execute(
        "SELECT body_json FROM mo_assets WHERE mode=? AND kind=? ORDER BY created_at DESC, id DESC", (mode, kind)
    )]


def save(mode: str, kind: str, body: dict[str, Any], *, create: bool = False) -> dict[str, Any]:
    body = {**body, "mode": mode, "source": mode}
    args = (mode, kind, body["id"], json.dumps(body, ensure_ascii=False, allow_nan=False), body.get("created_at") or now())
    sql = "INSERT INTO mo_assets (mode,kind,id,body_json,created_at) VALUES (?,?,?,?,?)"
    if not create:
        sql += " ON CONFLICT(mode,kind,id) DO UPDATE SET body_json=excluded.body_json"
    try:
        _conn().execute(sql, args)
        _conn().commit()
    except sqlite3.IntegrityError as exc:
        raise HTTPException(409, "이미 등록된 ID입니다.") from exc
    return body


def audit(mode: str, action: str, asset_id: str, actor: str, **details: Any) -> None:
    save(mode, "audit", {"id": new_id(), "created_at": now(), "action": action,
                         "asset_id": asset_id, "actor": actor, **details}, create=True)


def delete_pipeline(mode: str, asset_id: str, actor: str) -> None:
    get(mode, "pipeline", asset_id)
    _conn().execute("DELETE FROM mo_assets WHERE mode=? AND kind='pipeline' AND id=?", (mode, asset_id))
    _conn().commit()
    audit(mode, "pipeline_deleted", asset_id, actor)
