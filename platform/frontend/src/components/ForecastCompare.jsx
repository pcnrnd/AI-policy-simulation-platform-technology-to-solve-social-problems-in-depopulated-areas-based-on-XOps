/** ③ 예측 카드 아래 비교 줄 — 비교 대상 이름 한 줄 아래 활성 모델 예측과 원격 시계열 모델 예측(범위)을 나란히.
 * 원격 예측이 없으면(미설정·시간 초과·오류·계열 없음) 시계열 모델 값 자리에 "—"만 둔다. 화면에는 역할 이름만 쓴다(두 모드 같음). */
export default function ForecastCompare({ view, loading }) {
  if (!view && !loading) return null;
  return (
    <div className="sim-compare" data-slot="sim.sim.compare">
      {view && <p className="sim-compare-title">{view.title}</p>}
      <div className="sim-compare-cell">
        <span className="sim-compare-tag">활성 모델</span>
        <strong>{view ? view.ridge : "…"}</strong>
      </div>
      <div className="sim-compare-cell">
        <span className="sim-compare-tag">시계열 모델</span>
        <strong>{!view ? "…" : view.remote ? view.remote.median : "—"}</strong>
        {view?.remote && <span className="sim-compare-range">범위 {view.remote.range}</span>}
      </div>
    </div>
  );
}
