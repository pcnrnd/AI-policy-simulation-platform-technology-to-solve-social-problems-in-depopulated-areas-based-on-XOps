"""메타데이터 카탈로그 — mock_data.json의 metadata_schemas를 로드/검색.

API가 데이터를 요청하면 메타데이터를 검색해 저장소 정보를 얻고 적합한 Adapter로 라우팅한다
(무분별한 저장소 직접 접근 차단). 실제 메타 스토어 연결은 후속.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from src.core import db
from src.core.exceptions import SourceNotFoundError, XopsError
from src.core.settings import get_settings


class DuplicateSourceError(XopsError):
    """이미 존재하는 소스 id로 등록 시도."""

    status_code = 409


class ProtectedSourceError(XopsError):
    """기본(시드) 소스는 삭제할 수 없음."""

    status_code = 403


class MetadataCatalog:
    """데이터 소스 스키마 카탈로그 (id·태그·객체명·설명 검색)."""

    def __init__(self, schemas: list[dict[str, Any]]) -> None:
        # 시드 스키마(읽기 전용). 사용자 등록 소스는 SQLite가 소스오브트루스.
        self._seed: dict[str, dict[str, Any]] = {s["id"]: s for s in schemas}

    @classmethod
    def from_file(cls, path: Any) -> "MetadataCatalog":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data.get("metadata_schemas", []))

    def list_sources(self) -> list[dict[str, Any]]:
        return [*self._seed.values(), *db.list_user_sources()]

    def get(self, source_id: str) -> dict[str, Any]:
        schema = self._seed.get(source_id) or db.get_user_source(source_id)
        if schema is None:
            raise SourceNotFoundError(f"데이터 소스를 찾을 수 없습니다: {source_id}")
        return schema

    def search(self, query: str) -> list[dict[str, Any]]:
        """소스명·태그·설명·객체명 부분 일치 검색."""
        q = query.strip().lower()
        sources = self.list_sources()
        return sources if not q else [s for s in sources if _matches(s, q)]

    def add(self, schema: dict[str, Any]) -> dict[str, Any]:
        """사용자 소스 등록 — user_registered 표식 부여, id 중복 거부 (SQLite 영속화)."""
        source_id = schema["id"]
        if source_id in self._seed or db.get_user_source(source_id) is not None:
            raise DuplicateSourceError(f"이미 존재하는 소스 id입니다: {source_id}")
        stored = {**schema, "user_registered": True}
        db.add_user_source(stored)
        return stored

    def remove(self, source_id: str) -> None:
        """사용자 등록 소스만 삭제 — 기본 시드 소스는 보호."""
        if source_id in self._seed:
            raise ProtectedSourceError(f"기본 소스는 삭제할 수 없습니다: {source_id}")
        if not db.delete_user_source(source_id):
            raise SourceNotFoundError(f"데이터 소스를 찾을 수 없습니다: {source_id}")


def _matches(schema: dict[str, Any], q: str) -> bool:
    haystack = " ".join(
        [
            str(schema.get("id", "")),
            str(schema.get("label", "")),
            str(schema.get("object", "")),
            str(schema.get("description", "")),
            " ".join(schema.get("tags", [])),
        ]
    ).lower()
    return q in haystack


@lru_cache
def get_catalog() -> MetadataCatalog:
    """카탈로그 싱글톤 (mock_data.json 시드)."""
    return MetadataCatalog.from_file(get_settings().mock_data_path)
