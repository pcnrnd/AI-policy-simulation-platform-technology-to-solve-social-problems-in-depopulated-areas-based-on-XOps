"""gwto 시트 공통 파싱 헬퍼: 빈 행 기준 블록 분리, 연도/월 헤더 정규식, 합계행 판정."""

from __future__ import annotations

import re

DROP_LABELS = {"합계", "누계", "평균", "소계"}

_YEAR_MONTH_RE = re.compile(r"(\d{4})\s*년\s*0?(\d{1,2})\s*월")
_YEAR_ONLY_RE = re.compile(r"(\d{4})\s*년\s*$")
_MONTH_ONLY_RE = re.compile(r"^0?(\d{1,2})\s*월$")
_YEAR2_MONTH_RE = re.compile(r"(\d{2})\s*년\s*0?(\d{1,2})\s*월")


def parse_year_month(label: str) -> tuple[int, int] | None:
    """'2020년 1월' / '2020년1월' / '25년 01월' → (year, month). 실패 시 None."""
    m = _YEAR_MONTH_RE.search(label)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = _YEAR2_MONTH_RE.search(label)
    if m:
        return 2000 + int(m.group(1)), int(m.group(2))
    return None


def parse_month_only(label: str) -> int | None:
    m = _MONTH_ONLY_RE.match(label.strip())
    return int(m.group(1)) if m else None


def parse_year_only(label: str) -> int | None:
    m = _YEAR_ONLY_RE.match(label.strip())
    return int(m.group(1)) if m else None


def is_blank_row(row: list[str], max_col: int | None = None) -> bool:
    cells = row[:max_col] if max_col is not None else row
    return all(not c.strip() for c in cells)


def split_paragraphs(
    rows: list[list[str]], max_col: int | None = None
) -> list[tuple[int, int]]:
    """빈 행(지정 열 범위 기준)으로 구분되는 (시작, 끝) 행 구간 목록. 끝은 배타적."""
    paragraphs: list[tuple[int, int]] = []
    start: int | None = None
    for i, row in enumerate(rows):
        blank = is_blank_row(row, max_col)
        if not blank and start is None:
            start = i
        elif blank and start is not None:
            paragraphs.append((start, i))
            start = None
    if start is not None:
        paragraphs.append((start, len(rows)))
    return paragraphs


def is_drop_label(cell: str) -> bool:
    return cell.strip() in DROP_LABELS


class MeltStats:
    """언피벗 과정에서 드롭한 행 수와 실제 소비한 값 셀 수를 누적한다."""

    def __init__(self) -> None:
        self.title_rows = 0
        self.total_rows = 0  # 합계/누계/평균 등 요약 행
        self.blank_rows = 0
        self.value_cells = 0  # 실제 언피벗에 사용한 non-blank 값 셀 수
        self.duplicate_cells = 0  # 중복 블록에서 스킵한 값 셀 수(값 일치 확인됨)

    def dropped_rows_dict(self) -> dict[str, int]:
        return {
            "title": self.title_rows,
            "total": self.total_rows,
            "blank": self.blank_rows,
            "side_table": 0,
        }


def melt_year_month_header_rows(
    rows: list[list[str]],
    max_col: int,
    header_detector=lambda row: row[0].strip() == "구분",
) -> tuple[list[tuple[str, int, int, str]], MeltStats]:
    """'구분'+연-월 헤더가 반복되는 세로 블록을 (category, year, month, value)로 언피벗한다.

    블록은 빈 행으로 구분된 단락(paragraph) 단위로 인식하고, 각 단락 안에서
    맨 처음 나오는 header_detector 만족 행을 헤더로, 그 앞 행은 title 로 드롭한다.
    같은 (category, year, month) 조합이 다른 블록에 중복 등장하면(원본 자체 중복이거나
    뒤쪽 블록이 앞쪽을 정정한 경우) 뒤에 나온 값으로 덮어쓰고 duplicate_cells 로 집계한다
    (시트 뒤쪽 블록이 최신/정정본이라는 관례를 따름 — 호출측이 notes 로 기록해야 한다).
    """
    stats = MeltStats()
    out: list[tuple[str, int, int, str]] = []
    seen: dict[tuple[str, int, int], int] = {}  # key -> out 인덱스

    for start, end in split_paragraphs(rows, max_col):
        header_idx = None
        for i in range(start, end):
            if header_detector(rows[i]):
                header_idx = i
                break
        if header_idx is None:
            stats.title_rows += end - start
            continue
        stats.title_rows += header_idx - start

        header = rows[header_idx][:max_col]
        col_time: dict[int, tuple[int, int]] = {}
        for idx in range(1, len(header)):
            ym = parse_year_month(header[idx])
            if ym:
                col_time[idx] = ym

        for r in range(header_idx + 1, end):
            row = rows[r][:max_col]
            category = row[0].strip()
            if not category:
                stats.blank_rows += 1
                continue
            if is_drop_label(category):
                stats.total_rows += 1
                continue
            for idx, (year, month) in col_time.items():
                value = row[idx].strip() if idx < len(row) else ""
                if not value:
                    continue
                key = (category, year, month)
                if key in seen:
                    stats.duplicate_cells += 1
                    out[seen[key]] = (category, year, month, value)
                    continue
                seen[key] = len(out)
                stats.value_cells += 1
                out.append((category, year, month, value))
    return out, stats


def find_time_pairs(
    label_row: list[str], pair_start_col: int, period: int, max_col: int | None = None
) -> list[tuple[int, int, int]]:
    """라벨 행에서 pair_start_col 부터 period 간격으로 연-월 라벨을 찾아 (col,year,month) 목록을 반환."""
    width = max_col or len(label_row)
    pairs: list[tuple[int, int, int]] = []
    for p in range(pair_start_col, width, period):
        label = label_row[p] if p < len(label_row) else ""
        ym = parse_year_month(label)
        if ym:
            pairs.append((p, ym[0], ym[1]))
    return pairs


def melt_grouped_wide_blocks(
    rows: list[list[str]], id_col: int = 0
) -> tuple[list[tuple[str, int, int, str, str]], MeltStats]:
    """16_s3_1_city_tourists: (연-월 슈퍼헤더 3열그룹 + 서브카테고리 헤더) 블록 반복.

    슈퍼헤더 행은 그룹 첫 열에만 값이 있고(병합셀), 서브헤더 행은 매 열이 채워져 있다.
    빈 행으로 갈리는 단락마다 첫 '구분' 행을 슈퍼헤더로 찾고, 그 다음 행을 서브헤더로
    쓴다. 같은 (entity, year, month, subcat) 이 중복 등장하면(뒤 블록이 정정본인 경우
    포함) 뒤에 나온 값으로 덮어쓴다. 반환: (entity, year, month, subcat, value).
    """
    stats = MeltStats()
    out: list[tuple[str, int, int, str, str]] = []
    seen: dict[tuple[str, int, int, str], int] = {}

    for start, end in split_paragraphs(rows):
        super_idx = None
        for i in range(start, end):
            if rows[i][id_col].strip() == "구분":
                super_idx = i
                break
        if super_idx is None:
            stats.title_rows += end - start
            continue
        stats.title_rows += super_idx - start
        if super_idx + 1 >= end:
            stats.title_rows += end - super_idx
            continue
        stats.title_rows += 1  # 서브헤더 행

        super_row = rows[super_idx]
        sub_row = rows[super_idx + 1]
        width = max(len(super_row), len(sub_row))
        filled = []
        last = ""
        for c in range(width):
            v = super_row[c].strip() if c < len(super_row) else ""
            if v:
                last = v
            filled.append(last)

        col_meta: dict[int, tuple[int, int, str]] = {}
        for c in range(1, width):
            ym = parse_year_month(filled[c])
            subcat = sub_row[c].strip() if c < len(sub_row) else ""
            if ym and subcat:
                col_meta[c] = (ym[0], ym[1], subcat)

        for r in range(super_idx + 2, end):
            row = rows[r]
            entity = row[id_col].strip() if id_col < len(row) else ""
            if not entity:
                stats.blank_rows += 1
                continue
            if is_drop_label(entity):
                stats.total_rows += 1
                continue
            for c, (year, month, subcat) in col_meta.items():
                value = row[c].strip() if c < len(row) else ""
                if not value:
                    continue
                key = (entity, year, month, subcat)
                if key in seen:
                    stats.duplicate_cells += 1
                    out[seen[key]] = (entity, year, month, subcat, value)
                    continue
                seen[key] = len(out)
                stats.value_cells += 1
                out.append((entity, year, month, subcat, value))
    return out, stats


def _is_labeled_county_header(row: list[str]) -> bool:
    nonempty = sum(1 for c in row if c.strip())
    if nonempty < 2:
        return False
    c0 = row[0].strip()
    return c0 == "구분" or parse_year_month(c0) is not None


def melt_label_above_header_blocks(
    rows: list[list[str]],
) -> tuple[list[tuple[int, int, str, str, str]], MeltStats, list[str]]:
    """17/18/22: 헤더 행 자체가 '구분' 이거나(17), 월 라벨이 헤더 행 col0 을 겸하는(18/22) 구조.

    헤더 후보 = col0 이 '구분' 이거나 연-월로 파싱되고, 2개 이상 열이 채워진 행.
    '구분' 헤더는 바로 앞에서 셀이 하나만 채워진 연-월 행을 거꾸로 찾아 라벨을 얻고,
    라벨형 헤더는 col0 자체가 라벨이다. 라벨을 못 찾으면(첫 블록이 라벨 없이 시작)
    해당 블록은 드롭한다. 반환: (year, month, category, county, value), stats, county_names.
    """
    header_positions = [i for i, row in enumerate(rows) if _is_labeled_county_header(row)]
    stats = MeltStats()
    out: list[tuple[int, int, str, str, str]] = []
    county_names: list[str] = []

    for hi, header_idx in enumerate(header_positions):
        header_row = rows[header_idx]
        c0 = header_row[0].strip()
        if c0 == "구분":
            limit = header_positions[hi - 1] + 1 if hi > 0 else 0
            label = None
            j = header_idx - 1
            while j >= limit:
                nonempty = [c.strip() for c in rows[j] if c.strip()]
                if len(nonempty) == 1:
                    ym = parse_year_month(nonempty[0])
                    if ym:
                        label = ym
                        break
                j -= 1
            stats.title_rows += header_idx - limit
        else:
            label = parse_year_month(c0)

        current_names = [c.strip() for c in header_row[1:]]
        if not county_names:
            county_names = current_names

        r = header_idx + 1
        next_header = header_positions[hi + 1] if hi + 1 < len(header_positions) else len(rows)
        while r < next_header:
            row = rows[r]
            nonempty = [c.strip() for c in row if c.strip()]
            if len(nonempty) <= 1:
                stats.blank_rows += 1
                r += 1
                continue
            category = row[0].strip()
            if is_drop_label(category):
                stats.total_rows += 1
                r += 1
                continue
            if label is not None:
                for ci, name in enumerate(current_names, start=1):
                    if not name:
                        continue
                    value = row[ci].strip() if ci < len(row) else ""
                    if not value:
                        continue
                    stats.value_cells += 1
                    out.append((label[0], label[1], category, name, value))
            else:
                stats.title_rows += 1  # 라벨 없는 블록(첫 블록 등)은 통째로 제외
            r += 1
    return out, stats, county_names


def melt_simple_wide(
    rows: list[list[str]],
    header_row_idx: int,
    id_col_count: int,
    county_names: list[str] | None = None,
) -> tuple[list[tuple[str, ...]], MeltStats]:
    """19/20/21: 블록 반복 없는 단일 와이드 표(헤더 1행 + 지역 열들)를 언피벗한다.

    id_col_count 개의 앞쪽 열은 그대로 유지하고, 그 뒤 county_names 개수만큼의 열을
    (지역명, 값) 쌍으로 풀어낸다. county_names 를 안 주면 헤더 행에서 그대로 읽는다.
    반환 튜플: (*id_vals, county_name, value).
    """
    header = rows[header_row_idx]
    if county_names is None:
        county_names = [c.strip() for c in header[id_col_count:]]

    stats = MeltStats()
    stats.title_rows += header_row_idx
    out: list[tuple[str, ...]] = []
    for r in range(header_row_idx + 1, len(rows)):
        row = rows[r]
        if is_blank_row(row):
            stats.blank_rows += 1
            continue
        id_vals = tuple(row[c].strip() if c < len(row) else "" for c in range(id_col_count))
        if id_vals and is_drop_label(id_vals[-1]):
            stats.total_rows += 1
            continue
        for ci, name in enumerate(county_names):
            if not name:
                continue
            col = id_col_count + ci
            value = row[col].strip() if col < len(row) else ""
            if not value:
                continue
            stats.value_cells += 1
            out.append(id_vals + (name, value))
    return out, stats


def melt_ranked_pairs(
    rows: list[list[str]], id_col: int = 0
) -> tuple[list[tuple[str, int, int, str, str]], MeltStats]:
    """09_s1_8_residence_city 처럼 블록마다 라벨 행의 (지역명,값) 쌍 시작열이 달라지는 구조.

    각 단락(빈 행으로 구분)에서 연-월 라벨이 처음 등장하는 행을 라벨 행으로 삼고,
    그 다음 행(서브헤더 '구분')을 드롭한 뒤, id_col 값이 있는 행만 데이터로 취급한다.
    반환: (rank_text, year, month, name, value) 목록.
    """
    stats = MeltStats()
    out: list[tuple[str, int, int, str, str]] = []
    seen: dict[tuple[str, int, int], int] = {}
    for start, end in split_paragraphs(rows):
        label_idx = None
        pair_start = None
        for i in range(start, end):
            row = rows[i]
            for c in range(1, len(row)):
                if parse_year_month(row[c]):
                    label_idx = i
                    pair_start = c
                    break
            if label_idx is not None:
                break
        if label_idx is None:
            stats.title_rows += end - start
            continue
        stats.title_rows += (label_idx - start) + 1  # 라벨 행 앞 title + 서브헤더 행

        pairs = find_time_pairs(rows[label_idx], pair_start, period=2, max_col=len(rows[label_idx]))
        for r in range(label_idx + 2, end):
            row = rows[r]
            rank = row[id_col].strip() if id_col < len(row) else ""
            if not rank:
                stats.blank_rows += 1
                continue
            for col, year, month in pairs:
                name = row[col].strip() if col < len(row) else ""
                value = row[col + 1].strip() if col + 1 < len(row) else ""
                if not name or not value:
                    continue
                key = (rank, year, month)
                entry = (rank, year, month, name, value)
                if key in seen:
                    stats.duplicate_cells += 1
                    out[seen[key]] = entry
                    continue
                seen[key] = len(out)
                stats.value_cells += 1
                out.append(entry)
    return out, stats


def melt_name_value_pairs(
    rows: list[list[str]],
    label_row_idx: int,
    data_start_idx: int,
    pair_start_col: int,
    period: int,
    max_col: int | None = None,
) -> tuple[list[tuple[int, int, str, str]], MeltStats]:
    """(라벨, 이름, 값) 3~n열 반복 블록을 (year, month, name, value)로 언피벗한다.

    06_datalab_foreigners, 09_s1_8_residence_city 처럼 (지역명, 값) 쌍이 period
    간격으로 가로 반복되고, 각 쌍의 첫 열(라벨 행 기준)에 연-월 라벨이 붙는 구조.
    """
    label_row = rows[label_row_idx]
    width = max_col or len(label_row)
    pairs: list[tuple[int, int, int]] = []  # (col, year, month)
    for p in range(pair_start_col, width, period):
        label = label_row[p] if p < len(label_row) else ""
        ym = parse_year_month(label)
        if ym:
            pairs.append((p, ym[0], ym[1]))

    stats = MeltStats()
    out: list[tuple[int, int, str, str]] = []
    seen: dict[tuple[int, int, str], int] = {}
    for r in range(data_start_idx, len(rows)):
        row = rows[r]
        if is_blank_row(row, width):
            stats.blank_rows += 1
            continue
        for p, year, month in pairs:
            name = row[p].strip() if p < len(row) else ""
            value = row[p + 1].strip() if p + 1 < len(row) else ""
            if not name or not value:
                continue
            key = (year, month, name)
            entry = (year, month, name, value)
            if key in seen:
                stats.duplicate_cells += 1
                out[seen[key]] = entry
                continue
            seen[key] = len(out)
            stats.value_cells += 1
            out.append(entry)
    return out, stats
