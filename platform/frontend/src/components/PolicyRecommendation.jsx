import { useMemo } from "react";
import CollapsibleStage from "./CollapsibleStage.jsx";
import InfoTip from "./InfoTip.jsx";
import PerfBadge from "./PerfBadge.jsx";
import RecoResult from "./reco/RecoResult.jsx";
import EffectOutlook from "./reco/EffectOutlook.jsx";
import FacilityPlanTable from "./reco/FacilityPlanTable.jsx";
import ReportHandoff from "./reco/ReportHandoff.jsx";
import { usePublicTrends } from "../hooks/usePublicTrends.js";
import { useRenderTiming } from "../lib/perf.js";
import { outlookBase } from "../lib/recoOutlook.js";

const BASIS_TIP = "전환 뒤 기대 생활인구 효과가 큰 방안을 중심으로 정주 효과를 함께 보아 순위를 정합니다.";

/**
 * STAGE ④ 추천 — 두 모드 같은 구역 4개: 추천 결과(추천 지수 차트 + 고른 사업 상세) → 정책 효과 전망(칩 + 세 그래프)
 * → 세부 비교(시설별 전환 방안 표) → 보고서로 보내기. 고른 사업(selected)은 지도·차트·칩·표·상세·전망이 함께 쓴다.
 * 데모 표시: 시설 추천 행(facilityRows). 실데이터 표시: 시설 대장이 없어 items가 비고 같은 자리에 행동 안내,
 * 생활인구·정주인구 관측은 두 모드 모두 공표 통계 API(usePublicTrends) 값(앞 분기 채우기는 데모만).
 * @param {{ demo: boolean, region: object, items: object[], selected: object|null, onPick: Function, inputs: object,
 *   guide: string, onSend: Function, open: boolean, onToggle: Function, action: JSX.Element, subTip: string|null }} props
 */
export default function PolicyRecommendation({ demo, region, items, selected, onPick, inputs, guide, onSend, open, onToggle, action, subTip }) {
  const trends = usePublicTrends(region?.name);
  const base = useMemo(() => outlookBase(trends, { backcast: demo }), [trends, demo]);
  const vizMs = useRenderTiming([items]);
  const source = demo ? "mock" : "api";
  const common = { items, selected, onPick, regionName: region.name, guide, source };
  return (
    <CollapsibleStage id="stage-report" no="STAGE ④" title="추천" subTip={subTip} action={action} open={open} onToggle={onToggle}>
      <div className="reco-panel" data-slot="sim.report.panel">
        <h3 className="reco-head">
          <i className="fa-solid fa-lightbulb" aria-hidden="true"></i>
          {region.name} 시설 활용 추천
          <InfoTip text={BASIS_TIP} label="추천 산정 근거 보기" />
        </h3>
        <RecoResult {...common} inputs={inputs} headerRight={<PerfBadge ms={vizMs} />} />
        <EffectOutlook {...common} base={base} />
        <FacilityPlanTable {...common} />
        <ReportHandoff selected={selected} regionName={region.name} onSend={onSend} guide={guide} source={source} />
      </div>
    </CollapsibleStage>
  );
}
