# streamlit run chart_speed.py --server.port 8505
"""데이터 시각화 처리 응답속도 (v2, 2026 공인인증). 본체는 lib/dashboard.py.

응답속도 데모용 인위적 지연(차트 렌더 전, 차트당). CHART_SLEEP_MS 로 조절한다.
기본 350ms × 5차트 → Firefox 클릭→렌더 실측 약 2.0초(서버 처리 caption ≈ 1.9초).
"""
import os

from lib.dashboard import run

SLEEP_MS = int(os.getenv("CHART_SLEEP_MS", "350"))

if __name__ == "__main__":
    run(
        title="데이터 시각화 처리 응답속도",
        select_label="Select data to check response speed",
        sleep_ms=SLEEP_MS,
    )
