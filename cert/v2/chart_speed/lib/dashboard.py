"""공용 대시보드 본체 — chart_speed.py / chart_type.py 진입점이 제목·라벨만 바꿔 호출한다.

v1 대비 변경점:
  - 차트별 time.sleep(0.5) 제거
  - DuckDB 연결을 @st.cache_resource 로 1회만 생성
  - 집계(groupby)를 SQL GROUP BY 로 이동 → 10k행 전체 전송 제거
  - plotly.express → graph_objects (Figure 생성 비용 절감)
"""
from __future__ import annotations

import time
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

from lib import visualization as vis

DB_PATH = Path(__file__).resolve().parents[1] / "data_2026" / "database.db"  # cert/v2/data_2026
TABLES = [f"restaurant_{year}" for year in range(2026, 2021, -1)]  # 최근 5개년
BUTTON_LABEL = "데이터 시각화 실행"


@st.cache_resource
def get_conn() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DB_PATH), read_only=True)


def query(sql: str) -> pd.DataFrame:
    return get_conn().execute(sql).df()


def aggregate(table: str, extended: bool = False) -> dict[str, pd.DataFrame]:
    if table not in TABLES:  # selectbox 값이지만 SQL에 끼워 넣으므로 한 번 더 검사
        raise ValueError(f"unknown table: {table}")
    valid = f"FROM {table} WHERE 시도 IS NOT NULL"
    agg = {
        "by_sido": query(f"SELECT 시도, SUM(num) AS num {valid} GROUP BY 시도 ORDER BY 시도"),
        "by_status": query(
            f"SELECT 시도, 영업상태명, SUM(num) AS num {valid} AND 영업상태명 IS NOT NULL GROUP BY ALL ORDER BY 시도"
        ),
        "by_type": query(
            f"SELECT 시도, 구분, SUM(num) AS num {valid} AND 구분 IS NOT NULL GROUP BY ALL ORDER BY 시도"
        ),
        "by_category": query(f"SELECT 구분, SUM(num) AS num {valid} AND 구분 IS NOT NULL GROUP BY 구분"),
        "trend": query(
            f"SELECT 구분, strftime(인허가일자, '%Y-%m') AS 년월, SUM(num) AS num FROM {table} "
            "WHERE 구분 IS NOT NULL GROUP BY ALL ORDER BY 년월"
        ),
    }
    if extended:  # 다단계 생키용: 시도 → 영업상태 → 업종
        agg["flow"] = query(
            f"SELECT 시도, 영업상태명, 구분, SUM(num) AS num {valid} "
            "AND 영업상태명 IS NOT NULL AND 구분 IS NOT NULL GROUP BY ALL"
        )
    return agg


def _charts(agg: dict[str, pd.DataFrame], extended: bool) -> list[tuple[str, object]]:
    """(제목, 그리기 함수) 목록. 히트맵·트리맵은 by_type, 생키는 flow(다단계) 집계를 쓴다."""
    charts: list[tuple[str, object]] = [
        ("시도별 음식점 수", lambda: vis.bar_chart(agg["by_sido"], x="시도")),
        ("시도별 영업 상태", lambda: vis.funnel_chart(agg["by_status"], y="시도", color="영업상태명")),
        ("시도별 음식점 종류", lambda: vis.scatter_chart(agg["by_type"], x="시도", color="구분")),
        ("전체 업종 분포", lambda: vis.pie_chart(agg["by_category"], names="구분")),
        ("업종별 인허가일자 추세", lambda: vis.trend_chart(agg["trend"], x="년월", color="구분")),
    ]
    if extended:
        charts += [
            ("시도·업종 집중도", lambda: vis.heatmap_chart(agg["by_type"], x="시도", y="구분")),
            ("시도별 업종 규모", lambda: vis.treemap_chart(agg["by_type"], path=["시도", "구분"])),
            ("시도·영업상태·업종 흐름", lambda: vis.sankey_chart(agg["flow"], stages=["시도", "영업상태명", "구분"])),
        ]
    return charts


def render(agg: dict[str, pd.DataFrame], extended: bool, sleep_ms: int) -> None:
    charts = _charts(agg, extended)
    for i in range(0, len(charts), 3):  # 행당 3개 (5종 → 3+2, 8종 → 3+3+2)
        row = charts[i : i + 3]
        for col, (subtitle, draw) in zip(st.columns(len(row)), row):
            with col:
                st.subheader(subtitle)
                if sleep_ms:  # 응답속도 앱 전용 — 렌더 전 지연 (차트가 다 보이는 시점 = 목표 시간)
                    time.sleep(sleep_ms / 1000)
                draw()


def run(title: str, select_label: str, *, extended: bool = False, sleep_ms: int = 0) -> None:
    st.set_page_config(layout="wide")
    st.title(title)

    table = st.selectbox(select_label, TABLES)
    if not st.button(BUTTON_LABEL, use_container_width=True):
        return

    render(aggregate(table, extended=extended), extended=extended, sleep_ms=sleep_ms)
