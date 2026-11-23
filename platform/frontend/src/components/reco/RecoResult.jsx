import Card from "../Card.jsx";
import { INDEX_MAX } from "../../lib/facilityRecommendation.js";
import { CHART_ITEMS, chosenOf, noReview, overviewRows, reasonRows, shortName } from "../../lib/recoText.js";

const TICKS = [0, 20, 40, 60, 80];
const pct = (v) => Math.max(0, Math.min(100, ((v ?? 0) / INDEX_MAX) * 100));

/** 방안 표시 — 방안 유형 태그 + 방안 문구("검토" 꼬리 없이). */
export function PlanLabel({ option }) {
  return <><span className="reco-plan-tag">{option.planType}</span>{noReview(option.text)}</>;
}

// 추천 지수 차트 — 1~5순위 가로 막대. 고른 막대 진하게, 1~3순위 연하게, 4·5순위 회색. 위아래 화살표로 순위 이동.
function IndexChart({ items, selected, onPick, regionName, guide }) {
  const shown = items.slice(0, CHART_ITEMS);
  const move = (e, k) => {
    if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
    e.preventDefault();
    const next = shown[k + (e.key === "ArrowDown" ? 1 : -1)];
    if (next) { onPick(next.id); e.currentTarget.parentElement.children[shown.indexOf(next)]?.focus(); }
  };
  return (
    <div className="reco-idx">
      <div className="reco-idx-head">추천 지수</div>
      <div className={`reco-bars${shown.length < 4 ? " few" : ""}`} role="group" aria-label="추천 지수 순위">
        {shown.length ? shown.map((it, k) => {
          const c = chosenOf(it);
          return (
            <button key={it.id} type="button" className="reco-bar" aria-pressed={it.id === selected?.id} onClick={() => onPick(it.id)} onKeyDown={(e) => move(e, k)}
              aria-label={`${it.rank}순위 ${shortName(it, regionName)} ${c?.planType ?? ""} 추천 지수 ${it.index ?? ""}`}>
              <span className="reco-bar-label"><span className="reco-bar-name">{shortName(it, regionName)}</span><span className="reco-bar-plan">{it.planType}</span></span>
              <span className="reco-bar-track">
                <span className={`reco-bar-fill${it.rank <= 3 ? " top" : ""}`} style={{ width: `${pct(it.index)}%` }}></span>
                <span className="reco-bar-val" style={{ left: `${pct(it.index)}%` }}>{it.index ?? ""}</span>
              </span>
            </button>
          );
        }) : <p className="reco-guide">{guide}</p>}
      </div>
      <div className="reco-idx-axis" aria-hidden="true">{TICKS.map((t) => <span key={t}>{t}</span>)}</div>
    </div>
  );
}

const Fields = ({ rows }) => (
  <dl className="reco-dfields">
    {rows.map(([name, value, strong]) => [<dt key={`${name}-t`}>{name}</dt>, <dd key={`${name}-d`} className={strong ? "strong" : undefined}>{value}</dd>])}
  </dl>
);

// 고른 사업 상세 — 머리(순위·시설·방안) + 사업 개요 | 선정 사유 두 칸. 사업이 없으면 같은 자리에 행동 안내.
function Detail({ item, regionName, inputs, guide }) {
  const c = chosenOf(item);
  return (
    <div className="reco-detail" aria-live="polite">
      <div className="reco-detail-rank">{item ? `${item.rank}순위` : "추천 사업"}</div>
      <h4 className={`reco-detail-name${item ? "" : " guide"}`}>{item ? shortName(item, regionName) : guide}</h4>
      <div className="reco-detail-plan">{c ? <PlanLabel option={c} /> : ""}</div>
      <div className="reco-dcols">
        <section className="reco-dcol"><h4>사업 개요</h4><Fields rows={overviewRows(item)} /></section>
        <section className="reco-dcol"><h4>선정 사유</h4><Fields rows={reasonRows(item, inputs)} /></section>
      </div>
    </div>
  );
}

/** 추천 결과 — 왼쪽 추천 지수 차트, 오른쪽 고른 사업 상세. */
export default function RecoResult({ items, selected, onPick, regionName, inputs, guide, source, headerRight }) {
  return (
    <Card title="추천 결과" className="reco-zone" dataSource={source} headerRight={headerRight} data-slot="sim.report.result">
      <div className="reco-result">
        <IndexChart items={items} selected={selected} onPick={onPick} regionName={regionName} guide={guide} />
        <Detail item={selected} regionName={regionName} inputs={inputs} guide={guide} />
      </div>
    </Card>
  );
}
