"""메타데이터 카탈로그 — mock_data.json의 metadata_schemas를 로드/검색.

API가 데이터를 요청하면 메타데이터를 검색해 저장소 정보를 얻고 적합한 Adapter로 라우팅한다
(무분별한 저장소 직접 접근 차단). 실제 메타 스토어 연결은 후속.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from src.core.exceptions import SourceNotFoundError
from src.core.settings import get_settings


class MetadataCatalog:
    """데이터 소스 스키마 카탈로그 (id·태그·객체명·설명 검색)."""

    def __init__(self, schemas: list[dict[str, Any]]) -> None:
        self._by_id: dict[str, dict[str, Any]] = {s["id"]: s for s in schemas}

    @classmethod
    def from_file(cls, path: Any) -> "MetadataCatalog":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data.get("metadata_schemas", []))

    def list_sources(self) -> list[dict[str, Any]]:
        return list(self._by_id.values())

    def get(self, source_id: str) -> dict[str, Any]:
        schema = self._by_id.get(source_id)
        if schema is None:
            raise SourceNotFoundError(f"데이터 소스를 찾을 수 없습니다: {source_id}")
        return schema

    def search(self, query: str) -> list[dict[str, Any]]:
        """소스명·태그·설명·객체명 부분 일치 검색."""
        q = query.strip().lower()
        if not q:
            return self.list_sources()
        return [s for s in self._by_id.values() if _matches(s, q)]


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
