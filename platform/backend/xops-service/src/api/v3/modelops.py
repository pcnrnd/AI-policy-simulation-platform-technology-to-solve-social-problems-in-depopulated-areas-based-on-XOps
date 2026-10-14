from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Header
from pydantic import BaseModel, Field

from src.api.dependencies import require_auth
from src.modelops import datasets, service, store

router = APIRouter(prefix="/modelops", tags=["modelops"])
Mode = Annotated[Literal["mock", "real"], Header(alias="X-Data-Mode")]
Read = Annotated[dict[str, Any], Depends(require_auth("data:read"))]
Write = Annotated[dict[str, Any], Depends(require_auth("data:write"))]


class DatasetRegistration(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    source_dataset_id: str = Field(min_length=1, max_length=128)
    source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    id_prefix: str = Field(default="modelops_", max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class InstanceRegistration(BaseModel):
    model_type: Literal["factor_analysis", "simulation"]
    dataset_id: str = Field(min_length=1, max_length=128)
    dataset_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    name: str = Field(default="", max_length=120)
    id_prefix: str = Field(default="modelops_", max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class PipelineRegistration(InstanceRegistration):
    id: str = Field(min_length=3, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    name: str = Field(min_length=1, max_length=120)
    trigger_policy: str = Field(default="수동", max_length=120)


def _actor(auth: dict[str, Any]) -> str:
    return str(auth.get("sub") or "api")


@router.on_event("startup")
def recover_jobs() -> None:
    service.recover_jobs()


@router.get("/catalog")
def catalog(auth: Read, mode: Mode = "real") -> dict[str, Any]:
    return {"mode": mode, "source": mode, "types": [{"model_type": key, "name": name} for key, name in store.MODEL_TYPES.items()],
            "instances": service.instances(mode), "datasets": store.list_assets(mode, "dataset"),
            "pipelines": store.list_assets(mode, "pipeline"), "jobs": store.list_assets(mode, "job"), "audit": store.list_assets(mode, "audit")}


@router.get("/available-datasets")
def available_datasets(auth: Read, mode: Mode = "real") -> dict[str, Any]:
    return {"mode": mode, "source": mode, "data": datasets.available(mode)}


@router.get("/datasets")
def list_datasets(auth: Read, mode: Mode = "real") -> dict[str, Any]:
    return {"mode": mode, "source": mode, "data": store.list_assets(mode, "dataset")}


@router.post("/datasets", status_code=201)
def register_dataset(body: DatasetRegistration, auth: Write, mode: Mode = "real") -> dict[str, Any]:
    return datasets.register(mode, body.model_dump(), _actor(auth))


@router.post("/instances", status_code=201)
def register_instance(body: InstanceRegistration, auth: Write, mode: Mode = "real") -> dict[str, Any]:
    return service.register_instance(mode, body.model_dump(), _actor(auth))


@router.get("/instances/{instance_id}/versions")
def list_versions(instance_id: str, auth: Read, mode: Mode = "real") -> dict[str, Any]:
    return {"mode": mode, "source": mode, "data": service.versions(mode, instance_id)}


@router.get("/instances/{instance_id}/versions/{version_id}/artifact")
def version_artifact(instance_id: str, version_id: str, auth: Read, mode: Mode = "real") -> dict[str, Any]:
    version = next((row for row in service.versions(mode, instance_id) if row["id"] == version_id), None)
    if version is None:
        store.get(mode, "version", "missing")
    return json.loads(Path(version["artifact_path"]).read_text(encoding="utf-8"))


@router.post("/instances/{instance_id}/versions/{version_id}/activate")
def activate_version(instance_id: str, version_id: str, auth: Write, mode: Mode = "real") -> dict[str, Any]:
    return service.activate(mode, instance_id, version_id, _actor(auth))


@router.post("/instances/{instance_id}/versions/{version_id}/rollback")
def rollback_version(instance_id: str, version_id: str, auth: Write, mode: Mode = "real") -> dict[str, Any]:
    return service.activate(mode, instance_id, version_id, _actor(auth), rollback=True)


@router.post("/pipelines", status_code=201)
def register_pipeline(body: PipelineRegistration, auth: Write, mode: Mode = "real") -> dict[str, Any]:
    return service.register_pipeline(mode, body.model_dump(), _actor(auth))


@router.delete("/pipelines/{pipeline_id}")
def delete_pipeline(pipeline_id: str, auth: Write, mode: Mode = "real") -> dict[str, Any]:
    store.delete_pipeline(mode, pipeline_id, _actor(auth))
    return {"deleted": pipeline_id, "mode": mode}


@router.post("/pipelines/{pipeline_id}/run", status_code=202)
def run_pipeline(pipeline_id: str, auth: Write, mode: Mode = "real") -> dict[str, Any]:
    return service.start_job(mode, pipeline_id, _actor(auth))


@router.get("/jobs/{job_id}")
def get_job(job_id: str, auth: Read, mode: Mode = "real") -> dict[str, Any]:
    return store.get(mode, "job", job_id)
