import Card from "../Card.jsx";
import { PlanLabel } from "./RecoResult.jsx";
import { GUIDE, chosenOf, shortName } from "../../lib/recoText.js";

/** 보고서로 보내기 — 고른 사업을 시행계획 중점과제·기금 투자계획 사업으로 리포팅에 넘긴다. 사업이 없으면 같은 자리에 행동 안내. */
export default function ReportHandoff({ selected, regionName, onSend, guide, source }) {
  const c = chosenOf(selected);
  return (
    <Card title="보고서로 보내기" className="reco-zone" dataSource={source} data-slot="sim.report.handoff">
      <div className="reco-handoff">
        <div>
          <strong>{c ? shortName(selected, regionName) : guide}</strong>
          <p>{c ? <PlanLabel option={c} /> : GUIDE.handoff}</p>
        </div>
        <button type="button" className="btn btn-primary" onClick={() => { if (c) onSend(selected); }} aria-disabled={!c || undefined}
          title={c ? undefined : GUIDE.handoff}>
          보고서로 보내기 <i className="fa-solid fa-arrow-right" aria-hidden="true"></i>
        </button>
      </div>
    </Card>
  );
}
