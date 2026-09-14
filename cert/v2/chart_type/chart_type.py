# streamlit run chart_type.py --server.port 8506
"""데이터 시각화 처리 유형 (v2, 2026 공인인증). 본체는 lib/dashboard.py.

유형 8종: 막대·퍼널·산점·파이·라인 + 히트맵·트리맵·생키 (extended=True).
"""
from lib.dashboard import run

if __name__ == "__main__":
    run(
        title="데이터 시각화 처리 유형",
        select_label="Select data for visualization",
        extended=True,
    )
