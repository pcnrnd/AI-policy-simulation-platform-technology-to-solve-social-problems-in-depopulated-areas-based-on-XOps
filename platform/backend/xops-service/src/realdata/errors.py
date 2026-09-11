"""실데이터 연계(A) 전용 예외 계층 — snapshot·features·models·evaluate가 공유한다."""

from __future__ import annotations


class RealdataError(Exception):
    """A 소유 모듈(snapshot/features/models/evaluate)의 기본 예외."""


class DatasetNotFound(RealdataError):
    """`load_dataset`이 SQLite 행 또는 파일을 찾지 못했을 때."""


class InsufficientData(RealdataError):
    """평가에 필요한 선행 관측(최소 12개월)이 부족할 때 — 메시지에 이유를 담는다."""


class MappingError(RealdataError):
    """BC `dong_name` → KT `dong_code` 대응 실패 — 미매핑 이름 목록을 보존한다."""

    def __init__(self, message: str, unmapped: list[str]) -> None:
        super().__init__(message)
        self.unmapped = unmapped
