"""2026 공인인증용 일반음식점 데이터 전처리 (chart_speed / chart_type 공용).

v1(`cert/v1/chart_speed/prepro/prepro_data.ipynb`)과 동일한 방식으로 처리하되,
2026 원본(`data/식품_일반음식점.csv`)의 스키마 차이 2건만 흡수한다.
  - `개방서비스명` 컬럼 없음      → 상수 '일반음식점' 추가
  - `도로명전체주소` → `도로명주소`  → 이름 매핑

실행 (cwd 무관): python prepro_2026.py
출력: data_2026/restaurant_{2021..2026}.csv + data_2026/database.db (테이블 restaurant_YYYY)
"""
from __future__ import annotations

import random
from pathlib import Path

import duckdb
import pandas as pd

HERE = Path(__file__).resolve().parent  # cert/v2
RAW = HERE.parents[1] / "data" / "식품_일반음식점.csv"  # 리포 root data/
OUT = HERE / "data_2026"
YEARS = range(2021, 2027)
SAMPLE_RANGE = (10120, 10200)  # v1과 동일: 연도별 최신 N행 랜덤 슬라이스
RAW_USECOLS = [
    "인허가일자", "폐업일자", "영업상태명", "도로명주소", "사업장명",
    "업태구분명", "남성종사자수", "여성종사자수",
]
V1_INPUT_COLS = [
    "개방서비스명", "인허가일자", "폐업일자", "영업상태명", "도로명전체주소",
    "사업장명", "업태구분명", "남성종사자수", "여성종사자수",
]
V1_OUTPUT_COLS = [
    "개방서비스명", "인허가일자", "폐업일자", "영업상태명", "소재지", "시설명", "구분",
    "남성종사자수", "여성종사자수", "시도", "시군구", "읍면동", "도로명", "번지", "num",
]
ADDRESS_COLS = ["시도", "시군구", "읍면동", "도로명", "번지"]


def split_address(address: str) -> tuple[str | None, ...]:
    """'전라북도 남원시 대강면 대강월산길 37-16' → (시도, 시군구, 읍면동, 도로명, 번지)."""
    parts = address.split()
    if len(parts) < 4:
        return (None,) * 5
    return parts[0], parts[1], parts[2], " ".join(parts[3:-1]), parts[-1]


def load_raw() -> pd.DataFrame:
    """원본 로드 후 v1 입력 스키마로 맞추고 인허가일자 내림차순 정렬."""
    df = pd.read_csv(RAW, encoding="cp949", usecols=RAW_USECOLS)
    df = (
        df.rename(columns={"도로명주소": "도로명전체주소"})
        .assign(개방서비스명="일반음식점")[V1_INPUT_COLS]
        .sort_values("인허가일자", ascending=False)
    )
    parsed = pd.to_datetime(df["인허가일자"], format="%Y-%m-%d", errors="coerce")
    return df.assign(인허가일자=parsed.ffill())


def prepro_year(df: pd.DataFrame) -> pd.DataFrame:
    """v1 `prepro_df`: 최신 N행 샘플 → 주소 정제/분해 → num=1 → 컬럼명 변경."""
    n = random.randint(*SAMPLE_RANGE)
    df = df.iloc[:n].copy()
    df["도로명전체주소"] = df["도로명전체주소"].fillna("").str.split(",").str[0]
    df["폐업일자"] = df["폐업일자"].fillna("운영중")
    df[ADDRESS_COLS] = pd.DataFrame(
        [split_address(a) for a in df["도로명전체주소"]], index=df.index, columns=ADDRESS_COLS
    )
    df["num"] = 1
    renamed = df.rename(columns={"도로명전체주소": "소재지", "사업장명": "시설명", "업태구분명": "구분"})
    return renamed.reset_index(drop=True)


def check(frames: dict[int, pd.DataFrame]) -> None:
    for year, df in frames.items():
        assert list(df.columns) == V1_OUTPUT_COLS, (year, list(df.columns))
        assert SAMPLE_RANGE[0] <= len(df) <= SAMPLE_RANGE[1], (year, len(df))
        assert (df["인허가일자"].dt.year == year).all(), year
        assert (df["num"] == 1).all(), year
    print("check OK: columns == v1, rows in range, years match")


def main() -> None:
    random.seed(2026)  # v1은 비고정 랜덤 — 재실행 재현성을 위해 고정
    raw = load_raw()
    frames = {y: prepro_year(raw[raw["인허가일자"].dt.year == y]) for y in YEARS}

    OUT.mkdir(exist_ok=True)
    conn = duckdb.connect(str(OUT / "database.db"))
    for year, df in frames.items():
        df.to_csv(OUT / f"restaurant_{year}.csv", index=False)
        conn.execute(f"CREATE OR REPLACE TABLE restaurant_{year} AS SELECT * FROM df")
        print(f"restaurant_{year}: {len(df)} rows")
    conn.close()
    check(frames)


if __name__ == "__main__":
    main()
