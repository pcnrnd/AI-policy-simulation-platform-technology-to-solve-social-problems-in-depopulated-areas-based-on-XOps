"""차트 5종 (plotly graph_objects).

v1은 plotly.express를 썼는데 Figure 생성에 차트당 ~50ms가 든다(검증 오버헤드).
같은 결과물을 go.*로 직접 구성해 서버 측 Figure 생성 비용을 줄인다.
입력은 이미 집계된 DataFrame(SQL GROUP BY 결과)이다.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import qualitative

PALETTE = qualitative.Plotly


def _show(fig: go.Figure, height: int = 450) -> None:
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig)


def _traces_by(df: pd.DataFrame, color: str, x: str, y: str, **kw) -> list[go.Scatter]:
    return [go.Scatter(name=str(k), x=g[x], y=g[y], **kw) for k, g in df.groupby(color)]


def bar_chart(df: pd.DataFrame, x: str, y: str = "num") -> None:
    """시도별 음식점 수 — 단일 trace, 막대별 색상."""
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(df))]
    fig = go.Figure(go.Bar(x=df[x], y=df[y], marker_color=colors))
    fig.update_layout(xaxis_tickangle=90, showlegend=False)
    _show(fig)


def funnel_chart(df: pd.DataFrame, y: str, color: str, x: str = "num") -> None:
    """시도별 영업 상태."""
    fig = go.Figure([go.Funnel(name=str(k), y=g[y], x=g[x]) for k, g in df.groupby(color)])
    _show(fig)


def scatter_chart(df: pd.DataFrame, x: str, color: str, y: str = "num") -> None:
    """시도별 음식점 종류."""
    fig = go.Figure(_traces_by(df, color, x, y, mode="markers"))
    fig.update_layout(xaxis_tickangle=90)
    _show(fig)


def pie_chart(df: pd.DataFrame, names: str, values: str = "num") -> None:
    """전체 업종 분포."""
    if df.empty:
        st.warning("표시할 데이터가 없습니다.")
        return
    fig = go.Figure(go.Pie(labels=df[names], values=df[values], marker_colors=qualitative.Set3))
    _show(fig, height=500)


def trend_chart(df: pd.DataFrame, x: str, color: str, y: str = "num") -> None:
    """업종별 인허가일자 월별 추세."""
    fig = go.Figure(_traces_by(df, color, x, y, mode="lines"))
    fig.update_layout(xaxis_tickangle=45, xaxis_title="인허가 년월", yaxis_title="음식점 수")
    _show(fig, height=400)


# --- 유형 확장 3종 (chart_type 전용). 모두 by_type 집계(시도·구분·num) 재사용 ---

def heatmap_chart(df: pd.DataFrame, x: str, y: str, values: str = "num") -> None:
    """시도 × 구분 교차 분포 (행렬)."""
    pivot = df.pivot_table(index=y, columns=x, values=values, aggfunc="sum", fill_value=0)
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale="Blues", labels={"color": "수"})
    fig.update_xaxes(tickangle=90)
    _show(fig, height=500)


def treemap_chart(df: pd.DataFrame, path: list[str], values: str = "num") -> None:
    """시도 → 구분 계층별 규모."""
    fig = px.treemap(df, path=path, values=values, color_discrete_sequence=qualitative.Set3)
    _show(fig, height=500)


def sankey_chart(df: pd.DataFrame, stages: list[str], values: str = "num", top: int = 8) -> None:
    """다단계 흐름 (예: 시도 → 영업상태 → 업종).

    stages: 왼→오 순서의 컬럼 리스트. 마지막 단계(업종)만 상위 top + '그외'로 축소.
    노드는 (단계, 값)으로 유일화 → 같은 문자열이 다른 단계에 있어도 안전.
    """
    d = df.copy()
    last = stages[-1]
    keep = d.groupby(last)[values].sum().nlargest(top).index
    d[last] = d[last].where(d[last].isin(keep), "그외")

    labels: list[str] = []
    idx: dict[tuple[str, object], int] = {}
    for stage in stages:
        for v in d[stage].drop_duplicates():
            key = (stage, v)
            if key not in idx:
                idx[key] = len(labels)
                labels.append(str(v))

    src, tgt, val = [], [], []
    for a, b in zip(stages, stages[1:]):
        link = d.groupby([a, b], as_index=False)[values].sum()
        src += [idx[(a, v)] for v in link[a]]
        tgt += [idx[(b, v)] for v in link[b]]
        val += link[values].tolist()

    fig = go.Figure(go.Sankey(
        arrangement="snap",  # 노드 라벨 겹침 완화
        node=dict(label=labels, pad=24, thickness=18,
                  line=dict(color="rgba(0,0,0,0.35)", width=0.6)),
        link=dict(source=src, target=tgt, value=val),
        textfont=dict(size=14, color="#111"),  # 기본보다 크고 진하게 → 가독성
    ))
    _show(fig, height=620)
