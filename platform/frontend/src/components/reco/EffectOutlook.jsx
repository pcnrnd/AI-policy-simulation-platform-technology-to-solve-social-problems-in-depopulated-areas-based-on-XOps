import { useMemo } from "react";
import Card from "../Card.jsx";
import { ChartEmptyNote } from "../PendingData.jsx";
import OutlookChart from "./OutlookChart.jsx";
import { outlookChart } from "../../lib/recoOutlook.js";
import { GUIDE, businessName, chipItems, chosenOf, effectHeads, pctRange, shortName } from "../../lib/recoText.js";

function Head({ title, head, guide }) {
  return (
    <figcaption>
      <strong>{title}</strong>
      <span className={`reco-fig-value${head ? "" : " guide"}`}>{head ? head.value : guide}</span>
      <span className="reco-fig-sub">{head?.sub ?? ""}</span>
    </figcaption>
  );
}

// 사용률 전·후 — 현재 막대 하나, 전환 후 막대 하나(범위 가운데까지 채우고 글씨는 범위).
function UtilBars({ item, guide }) {
  const c = chosenOf(item);
  if (!c) return <div className="reco-util"><ChartEmptyNote>{guide}</ChartEmptyNote></div>;
  const a = c.utilAfter;
  return (
    <div className="reco-util" role="img" aria-label={`사용률 현재 ${item.util}% 전환 후 ${pctRange(a)}`}>
      <div className="reco-util-row"><span className="reco-util-name">현재</span><span className="reco-util-track"><span className="reco-util-fill base" style={{ width: `${item.util}%` }}></span></span><span className="reco-util-val">{item.util}%</span></div>
      <div className="reco-util-row"><span className="reco-util-name">전환 후</span><span className="reco-util-track"><span className="reco-util-fill after" style={{ width: `${(a.low + a.high) / 2}%` }}></span></span><span className="reco-util-val">{pctRange(a)}</span></div>
      <div className="reco-util-axis" aria-hidden="true"><span>0</span><span>25</span><span>50</span><span>75</span><span>100%</span></div>
    </div>
  );
}

/**
 * 정책 효과 전망 — 반영 정책 칩(추천 지수 차트·표와 같은 선택) + 사용률 전·후 + 생활인구·정주인구 전망.
 * 효과 숫자는 각 그래프 머리에만 둔다(추천 결과와 같은 값을 두 번 보이지 않는다).
 */
export default function EffectOutlook({ items, selected, onPick, regionName, base, guide, source }) {
  const heads = effectHeads(selected);
  const charts = useMemo(() => ["population", "settle"].map((id) => outlookChart(base, id, selected)), [base, selected]);
  const chips = chipItems(items, selected);
  const pending = selected || guide === GUIDE.registry ? GUIDE.pending : guide;
  return (
    <Card title="정책 효과 전망" className="reco-zone" dataSource={source} data-slot="sim.report.outlook">
      <div className="reco-chips" role="group" aria-label="반영 정책">
        <span>반영 정책</span>
        {chips.length ? chips.map((it) => (
          <button key={it.id} type="button" className="reco-chip" aria-pressed={it.id === selected?.id} aria-label={businessName(it, regionName)} onClick={() => onPick(it.id)}>
            <b>{it.rank}</b>{shortName(it, regionName)}
          </button>
        )) : <span className="reco-guide">{guide}</span>}
      </div>
      <div className="reco-figs" data-slot="sim.report.charts">
        <figure className="reco-fig">
          <Head title="사용률 전·후" head={heads.util} guide={pending} />
          <UtilBars item={selected} guide={GUIDE.usage} />
        </figure>
        <figure className="reco-fig">
          <Head title="생활인구 전망" head={heads.population} guide={pending} />
          <OutlookChart chart={charts[0]} guide={GUIDE.trends} />
        </figure>
        <figure className="reco-fig">
          <Head title="정주인구 전망" head={heads.settle} guide={pending} />
          <OutlookChart chart={charts[1]} guide={GUIDE.trends} />
        </figure>
      </div>
    </Card>
  );
}
