"""타 부서 연동 응답 계약 — OpenAPI에 실은 payload 타입이 실제 응답과 같은지 검사한다.

라우트는 dict를 그대로 돌려준다(응답 JSON 불변). 타입은 문서화용이라, 실제 응답을 타입으로
검증한 뒤 다시 직렬화해 원본과 같은지 본다 — 빠지거나 남는 필드가 있으면 계약이 어긋난 것이다.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from src.schemas.overview import OverviewSummary
from src.schemas.realdata import Envelope, RealdataEvaluation, RealdataModelEntry

_MODELS = "/api/v3/realdata/models"
_EVALUATION = "/api/v3/realdata/models/{model_id}/evaluation"
_OVERVIEW = "/api/v3/overview/summary"


def _assert_conforms(model: type[BaseModel], body: Any) -> None:
    parsed = model.model_validate(body)
    assert parsed.model_dump(mode="json", by_alias=True) == body


def _response_schema_name(spec: dict[str, Any], path: str) -> str:
    schema = spec["paths"][path]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    return str(schema["$ref"]).rsplit("/", 1)[-1]


# ── 실제 응답 ↔ 문서화 타입 ───────────────────────────────────
def test_realdata_models_response_matches_documented_type(client: TestClient, auth_headers: dict[str, str]) -> None:
    body = client.get(_MODELS, headers=auth_headers).json()
    assert {entry["model_id"] for entry in body["data"]} == {
        "namwon-nonlocal-visitors-next-month",
        "namwon-observed-sales-next-month",
    }
    _assert_conforms(Envelope[list[RealdataModelEntry]], body)


@pytest.mark.parametrize("include_seed", ["true", "false"])
def test_overview_summary_response_matches_documented_type(client: TestClient, include_seed: str) -> None:
    body = client.get(_OVERVIEW, params={"include_seed": include_seed}).json()
    if include_seed == "true":
        assert body["model"] is not None  # 모델 스냅샷 구조까지 검사되게
    _assert_conforms(OverviewSummary, body)


def test_evaluation_envelope_without_active_model_matches_documented_type(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    body = client.get(
        _EVALUATION.format(model_id="namwon-observed-sales-next-month"), headers=auth_headers
    ).json()
    assert body["status"] in ("model_required", "ok", "empty")
    _assert_conforms(Envelope[RealdataEvaluation], body)


# ── OpenAPI ──────────────────────────────────────────────────
def test_openapi_documents_the_three_payloads(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    components = spec["components"]["schemas"]

    models_schema = components[_response_schema_name(spec, _MODELS)]
    assert "#/components/schemas/RealdataModelEntry" in json.dumps(models_schema)
    evaluation_schema = components[_response_schema_name(spec, _EVALUATION)]
    assert "#/components/schemas/RealdataEvaluation" in json.dumps(evaluation_schema)
    assert _response_schema_name(spec, _OVERVIEW) == "OverviewSummary"
    for envelope in (models_schema, evaluation_schema):
        assert {"status", "message", "data", "provenance"} <= set(envelope["properties"])


@pytest.mark.parametrize(
    ("name", "model"),
    [
        ("RealdataModelEntry", RealdataModelEntry),
        ("RealdataEvaluation", RealdataEvaluation),
        ("OverviewSummary", OverviewSummary),
    ],
)
def test_openapi_examples_conform_to_their_types(client: TestClient, name: str, model: type[BaseModel]) -> None:
    examples = client.get("/openapi.json").json()["components"]["schemas"][name]["examples"]
    assert examples
    for example in examples:
        _assert_conforms(model, example)
