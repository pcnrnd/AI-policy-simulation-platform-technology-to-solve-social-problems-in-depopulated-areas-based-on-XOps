"""DataOps 요청/응답 DTO."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TokenResponse(BaseModel):
    """JWT 발급 응답."""

    access_token: str
    token_type: str = "Bearer"
    scope: str


class SourceSummary(BaseModel):
    """카탈로그 목록 항목."""

    id: str
    label: str | None = None
    source: str | None = None
    object: str | None = None
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    archive: dict[str, Any] | None = None
    range: dict[str, Any] | None = None


class WriteBody(BaseModel):
    """POST/PUT/PATCH 본문 — 컬럼 값(가상화 API라 저장소 추상화, 응답엔 미반영)."""

    data: dict[str, Any] = Field(default_factory=dict)
