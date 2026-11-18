import Card from "../Card.jsx";
import { PlanLabel } from "./RecoResult.jsx";
import { livingText, settleText } from "../../lib/facilityRecommendation.js";
import { CHART_ITEMS, budgetText, pctRange, sensText, shortName, stageWord } from "../../lib/recoText.js";

export const PLAN_COLUMNS = ["순위", "시설", "전환 방안", "사용률 변화", "생활인구 변화", "정주인구 변화", "예산 ±20%", "소요 예산", "판단"];
const BUDGET_COL = 7;

/**
 * 세부 비교 — 시설별 전환 방안을 시설마다 묶은 평탄한 표 하나(tbody 하나 = 시설 하나).
 * 순위·시설 칸은 세로로 합치고 추천 방안 행을 위에 둔다. 1~5순위 묶음을 누르면 차트·칩·상세·지도가 함께 바뀐다.
 */
export default function FacilityPlanTable({ items, selected, onPick, regionName, guide, source }) {
  const pickable = new Set(items.slice(0, CHART_ITEMS).map((it) => it.id));
  const rows = items.filter((it) => it.options);
  const pickProps = (it) => (pickable.has(it.id) ? {
    tabIndex: 0,
    "aria-current": it.id === selected?.id,
    onClick: () => onPick(it.id),
    onKeyDown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onPick(it.id); } }
  } : {});
  return (
    <Card title="세부 비교" className="reco-zone" dataSource={source} data-slot="sim.report.compare">
      <div className="reco-sub">시설별 전환 방안</div>
      <div className="table-container reco-table">
        <table>
          <caption className="sr-only">시설별 전환 방안</caption>
          <thead>
            <tr>{PLAN_COLUMNS.map((c, i) => <th key={c} scope="col" className={i === BUDGET_COL ? "cell-num" : undefined}>{c}</th>)}</tr>
          </thead>
          {rows.length ? rows.map((it) => {
            const opts = [...it.options].sort((a, b) => b.chosen - a.chosen);
            return (
              <tbody key={it.id} className="grp" {...pickProps(it)}>
                {opts.map((o, k) => (
                  <tr key={`${o.planType}-${o.kind}-${k}`} className={o.chosen ? "rec" : undefined}>
                    {k === 0 && <td rowSpan={opts.length} className="g-rank">{it.rank}</td>}
                    {k === 0 && <td rowSpan={opts.length} className="g-name">{shortName(it, regionName)}<span className="g-sub">{stageWord(it.facility)}</span></td>}
                    <td><PlanLabel option={o} /></td>
                    <td className="nw">{`${it.util}% → ${pctRange(o.utilAfter)}`}</td>
                    <td className="nw">{livingText(o.living)}</td>
                    <td className="nw">{settleText(o)}</td>
                    <td className="nw">{sensText(o)}</td>
                    <td className="cell-num nw">{budgetText(o.budget)}</td>
                    <td className="nw"><span className="reco-verdict" data-verdict={o.verdict}>{o.verdict}</span></td>
                  </tr>
                ))}
              </tbody>
            );
          }) : (
            <tbody><tr><td colSpan={PLAN_COLUMNS.length} className="reco-guide">{guide}</td></tr></tbody>
          )}
        </table>
      </div>
    </Card>
  );
}
