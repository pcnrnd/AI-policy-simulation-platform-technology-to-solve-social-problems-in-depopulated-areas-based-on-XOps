"""Streamlit 시각화 응답속도 실측 — 버튼 클릭 → plotly 차트 5개 렌더 완료까지(ms).

브라우저 안에서 performance.now() 로 계측하므로 Playwright IPC 지연이 섞이지 않는다.
매 회 페이지를 새로 로드해 이전 차트가 남아있지 않은 상태에서 측정한다.

사용 (앱이 해당 포트에 실행 중이어야 함):
  python measure_speed.py http://localhost:8505                          # v2 chart_speed (5종)
  python measure_speed.py http://localhost:8506 --charts 8               # v2 chart_type  (8종)
  python measure_speed.py http://localhost:8505 --select restaurant_2025 # v1 (multiselect 선택 필요)
  옵션: --runs N (기본 5), --charts N (렌더 완료로 볼 차트 수, 기본 5), --browser firefox|chromium|webkit
사전: pip install playwright && playwright install firefox chromium
"""
from __future__ import annotations

import argparse
import statistics

from playwright.sync_api import Page, sync_playwright

BUTTON = "데이터 시각화 실행"
TIMEOUT_MS = 60_000
# plotly.js 가 그린 차트 개수(n)를 세고, 버튼 클릭 시점부터 rAF 로 폴링.
# 선택자는 막대/라인/산점/퍼널(g.trace)·파이(g.pielayer)·히트맵(.heatmaplayer)·생키(.sankey) 모두 포함.
CLICK_AND_WAIT_JS = f"""
(n) => new Promise((resolve, reject) => {{
  const btn = [...document.querySelectorAll('button')].find(b => b.innerText.includes('{BUTTON}'));
  if (!btn) return reject(new Error('button not found'));
  const drawn = () => [...document.querySelectorAll("[data-testid='stPlotlyChart'] .svg-container")]
      .filter(c => c.querySelector('g.trace, g.pielayer, .heatmaplayer, .sankey')).length;
  const t0 = performance.now();
  btn.click();
  const poll = () => {{
    if (drawn() >= n) return resolve(performance.now() - t0);
    if (performance.now() - t0 > {TIMEOUT_MS}) return reject(new Error('timeout: ' + drawn() + '/' + n));
    requestAnimationFrame(poll);
  }};
  requestAnimationFrame(poll);
}})
"""


def open_app(page: Page, url: str, select: str | None) -> None:
    page.goto(url, wait_until="networkidle")
    page.get_by_role("button", name=BUTTON).wait_for(timeout=TIMEOUT_MS)
    if select:  # v1: multiselect 에서 데이터 선택 (선택 자체가 rerun 을 일으킴)
        page.get_by_test_id("stMultiSelect").click()
        page.get_by_role("option", name=select).click()
        page.keyboard.press("Escape")
    page.wait_for_selector("[data-testid='stStatusWidget']", state="hidden", timeout=TIMEOUT_MS)


def measure_once(page: Page, url: str, select: str | None, charts: int) -> float:
    """버튼 클릭 → 차트 charts개 렌더 완료까지의 클라이언트 계측 시간(ms)."""
    open_app(page, url, select)
    return page.evaluate(CLICK_AND_WAIT_JS, charts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--select", default=None, help="multiselect 에서 고를 테이블명 (v1)")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--charts", type=int, default=5, help="렌더 완료로 볼 차트 수 (chart_type 은 8)")
    ap.add_argument("--browser", default="firefox", choices=["firefox", "chromium", "webkit"],
                    help="측정 브라우저 (기본 firefox — 개발자모드 기준과 동일 엔진)")
    args = ap.parse_args()

    with sync_playwright() as pw:
        browser = getattr(pw, args.browser).launch()
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        warm = measure_once(page, args.url, args.select, args.charts)
        print(f"[{args.browser}] warm-up: {warm:.0f} ms")
        times = [measure_once(page, args.url, args.select, args.charts) for _ in range(args.runs)]
        browser.close()

    for i, ms in enumerate(times, 1):
        print(f"run {i}: {ms:.0f} ms")
    print(
        f"{args.url}: mean {statistics.mean(times):.0f} ms | median {statistics.median(times):.0f} ms"
        f" | min {min(times):.0f} | max {max(times):.0f}  (n={len(times)})"
    )


if __name__ == "__main__":
    main()
