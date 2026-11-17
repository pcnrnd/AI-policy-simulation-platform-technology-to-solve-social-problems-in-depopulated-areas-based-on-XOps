import { REMOTE_FORECAST_GUIDE_TEXT } from "../lib/realSimulation.js";

/** ③ 예측 카드 아래 비교 줄 — 활성 모델 예측과 원격 시계열 모델 예측(범위)을 나란히. 원격 예측이 없으면 같은 자리에 행동 안내.
 * 화면에는 특정 모델 이름을 쓰지 않고 역할 이름만 쓴다(두 모드 같음). */
export default function ForecastCompare({ view, loading }) {
  if (!view && !loading) return null;
  return (
    <div className="sim-compare" data-slot="sim.sim.compare">
      <div className="sim-compare-cell">
        <span className="sim-compare-tag">활성 모델 예측{view ? ` · ${view.period}` : ""}</span>
        <strong>{view ? view.ridge : "…"}</strong>
      </div>
      <div className="sim-compare-cell">
        <span className="sim-compare-tag">시계열 모델 예측</span>
        {!view ? <strong>…</strong> : view.remote ? (
          <>
            <strong>{view.remote.median}</strong>
            <span className="sim-compare-range">범위 {view.remote.range}</span>
          </>
        ) : <p className="sim-compare-guide">{REMOTE_FORECAST_GUIDE_TEXT}</p>}
      </div>
    </div>
  );
}
