"""DataOps 요청/응답 DTO."""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.dataops.safety import MAX_IDENTIFIER_LENGTH, SQL_IDENTIFIER_RE, assert_safe_value

# pydantic Field(pattern=...) 에 넘길 문자열 — safety 의 정규식과 같은 출처를 쓴다.
_IDENTIFIER_PATTERN = SQL_IDENTIFIER_RE.pattern


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
    """컬럼 정의. name 은 생성 SQL에 식별자로 조립되므로 allowlist 패턴을 강제한다."""

    name: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH, pattern=_IDENTIFIER_PATTERN)
    type: str = Field(min_length=1, max_length=64)
    description: str = ""


class RangeDef(BaseModel):
    """메타데이터 적재 범위. column 은 식별자, from/to 는 SQL 리터럴로 들어간다."""

    column: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH, pattern=_IDENTIFIER_PATTERN)
    from_: Any = Field(alias="from")
    to: Any

    model_config = {"populate_by_name": True}

    @field_validator("from_", "to")
    @classmethod
    def _safe_boundary(cls, value: Any) -> Any:
        """따옴표·세미콜론·괄호가 섞인 경계값을 등록 단계에서 거부한다."""
        assert_safe_value(value, kind="range 경계값")
        return value


class ArchiveRegisterRequest(BaseModel):
    """신규 아카이브 등록 요청 — metadata_schema 형태로 정규화된다."""

    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$")
    label: str = ""
    source: str = Field(
        min_length=1, max_length=128, description="저장소 유형 문자열 (예: 'RDB · PostgreSQL')"
    )
    # object 는 생성 SQL의 FROM 절에 그대로 들어가므로 식별자 패턴을 강제한다.
    object: str = Field(
        min_length=1,
        max_length=MAX_IDENTIFIER_LENGTH,
        pattern=_IDENTIFIER_PATTERN,
        description="테이블/컬렉션명 (영문·숫자·밑줄, 숫자 시작 불가)",
    )
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
