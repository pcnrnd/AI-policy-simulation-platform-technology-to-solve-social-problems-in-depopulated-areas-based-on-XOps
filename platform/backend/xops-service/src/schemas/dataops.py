"""DataOps 요청/응답 DTO."""

from __future__ import annotations

from datetime import date
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
    lineage: dict[str, Any] | None = None
    range: dict[str, Any] | None = None
    columns: list[dict[str, Any]] = Field(default_factory=list)
    user_registered: bool = False


class WriteBody(BaseModel):
    """POST/PUT/PATCH 본문 — 컬럼 값(가상화 API라 저장소 추상화, 응답엔 미반영)."""

    data: dict[str, Any] = Field(default_factory=dict)


class ColumnDef(BaseModel):
    """컬럼 정의."""

    name: str = Field(min_length=1)
    type: str = Field(min_length=1)
    description: str = ""


class RangeDef(BaseModel):
    """메타데이터 적재 범위."""

    column: str
    from_: Any = Field(alias="from")
    to: Any

    model_config = {"populate_by_name": True}


class ArchiveRegisterRequest(BaseModel):
    """신규 아카이브 등록 요청 — metadata_schema 형태로 정규화된다."""

    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_\-]+$")
    label: str = ""
    source: str = Field(min_length=1, description="저장소 유형 문자열 (예: 'RDB · PostgreSQL')")
    object: str = Field(min_length=1, description="테이블/컬렉션명")
    description: str = ""
    tier: str = Field("Warm", pattern="^(Hot|Warm|Cold)$")
    retention: str = ""
    tags: list[str] = Field(default_factory=list)
    columns: list[ColumnDef] = Field(min_length=1)
    range: RangeDef | None = None

    def to_schema(self) -> dict[str, Any]:
        """카탈로그 저장 형태(metadata_schema)로 변환."""
        schema: dict[str, Any] = {
            "id": self.id,
            "label": self.label or self.object,
            "source": self.source,
            "object": self.object,
            "description": self.description,
            "tags": self.tags,
            "archive": {"tier": self.tier, "retention": self.retention, "loaded_at": date.today().isoformat()},
            "lineage": {"origin": "사용자 등록 (수동 메타데이터)", "version": "v1", "commit": self.id[-6:]},
            "columns": [c.model_dump() for c in self.columns],
        }
        if self.range is not None:
            schema["range"] = {"column": self.range.column, "from": self.range.from_, "to": self.range.to}
        return schema
