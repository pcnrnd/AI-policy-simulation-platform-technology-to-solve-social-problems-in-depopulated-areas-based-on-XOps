from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from src.core.seed import get_seed
from src.core.settings import get_settings
from src.modelops import factor_panel, store
from src.realdata import snapshot
from src.realdata.features import feature_names


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def available(mode: str) -> list[dict[str, Any]]:
    if mode == "mock":
        regions = get_seed()["regions"]
        return [{"id": "mock_" + region["id"], "name": region["name"] + " 인구 추이",
                 "region_id": region["id"], "target": "population_change_pct", "source": "mock",
                 "content_hash": _hash(region), "observed_from": 0, "observed_to": len(region["history"]) - 1}
                for region in regions] + [factor_panel.source(region, _hash(factor_panel.generate(region))) for region in regions]
    return [{"id": ds["dataset_id"], "name": "남원 " + ("방문객" if ds["spec"]["target"] == "nonlocal_visitors" else "관측 소비"),
             "region_id": "namwon", "target": ds["spec"]["target"], "source": "real",
             "content_hash": ds["content_hash"], "observed_from": ds["observed_from"], "observed_to": ds["observed_to"]}
            for ds in snapshot.list_datasets() if ds.get("spec", {}).get("model_id") in snapshot.TARGETS.values()]


def register(mode: str, body: dict[str, Any], actor: str) -> dict[str, Any]:
    source = next((row for row in available(mode) if row["id"] == body["source_dataset_id"]), None)
    if source is None:
        raise HTTPException(400, "현재 모드에서 사용할 학습데이터를 선택하세요.")
    if source["content_hash"] != body["source_hash"]:
        raise HTTPException(409, "데이터가 갱신됐습니다. 목록을 다시 확인하세요.")
    if mode == "mock" and source["id"].startswith(factor_panel.SOURCE_PREFIX):
        region = next(region for region in get_seed()["regions"] if factor_panel.SOURCE_PREFIX + region["id"] == source["id"])
        rows = factor_panel.generate(region)
        if not all(math.isfinite(row[name]) for row in rows for name in [*factor_panel.FEATURES, "y"]):
            raise HTTPException(400, "학습에 필요한 기간과 수치를 확인하세요.")
        document = {"panel": {"rows": rows, "features": factor_panel.FEATURES, "target": factor_panel.TARGET}}
        features = list(factor_panel.FEATURES)
        row_count = len(rows)
    elif mode == "mock":
        region = next(region for region in get_seed()["regions"] if "mock_" + region["id"] == source["id"])
        if len(region["history"]) < 6 or not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in region["history"]):
            raise HTTPException(400, "학습에 필요한 기간과 수치를 확인하세요.")
        document = {"region": region}
        features = ["rate_t-1", "riskIndex", "birthRate", "agingIndex"]
        row_count = len(region["history"])
    else:
        record = snapshot.load_dataset(source["id"])
        if not record.rows or record.target != source["target"] or record.content_hash != source["content_hash"] or snapshot._content_hash(record.rows) != record.content_hash:
            raise HTTPException(400, "학습데이터의 내용과 등록 정보를 확인하세요.")
        keys = [(row.get("base_ym"), row.get("dong_code")) for row in record.rows]
        if len(keys) != len(set(keys)) or any(not month or not dong for month, dong in keys):
            raise HTTPException(400, "중복된 지역·기간 또는 누락된 항목을 확인하세요.")
        if record.row_count != len(record.rows) or record.observed_from != min(month for month, _ in keys) or record.observed_to != max(month for month, _ in keys):
            raise HTTPException(400, "학습데이터의 기간과 행 수를 확인하세요.")
        if any(row.get("y") is not None and (isinstance(row["y"], bool) or not isinstance(row["y"], (int, float)) or not math.isfinite(row["y"])) for row in record.rows):
            raise HTTPException(400, "학습 대상 수치를 확인하세요.")
        document = asdict(record)
        features = feature_names(record)
        row_count = record.row_count
    dataset_id = store.new_id(body.get("id_prefix", "modelops_"))
    path = get_settings().model_artifact_dir / "modelops" / mode / "datasets" / (dataset_id + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    registered = store.save(mode, "dataset", {
        **source, "id": dataset_id, "dataset_id": dataset_id, "name": body["name"],
        "source_dataset_id": source["id"], "features": features, "row_count": row_count,
        "private_path": str(path), "private_hash": hashlib.sha256(path.read_bytes()).hexdigest(),
        "created_at": store.now(),
    }, create=True)
    store.audit(mode, "dataset_registered", dataset_id, actor, dataset_hash=source["content_hash"])
    return registered


def load(mode: str, dataset_id: str, expected_hash: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset = store.get(mode, "dataset", dataset_id)
    if expected_hash is not None and expected_hash != dataset["content_hash"]:
        raise HTTPException(409, "선택한 학습데이터를 다시 확인하세요.")
    path = Path(dataset["private_path"])
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != dataset["private_hash"]:
        raise HTTPException(409, "등록한 학습데이터의 내용을 확인하세요.")
    return dataset, json.loads(path.read_text(encoding="utf-8"))
