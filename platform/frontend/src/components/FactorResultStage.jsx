import Card from "./Card.jsx";
import CollapsibleStage from "./CollapsibleStage.jsx";
import PendingData from "./PendingData.jsx";

// STAGE ② 요인분석 결과
// 데이터 기반 사회문제: +/- 상관 → 문제 유형 진단 → 파라미터 도출.
// UI(카드 구조)는 항상 노출하고, 데이터는 STAGE ① 요인분석 완료(locked=false) 후 공개한다.
export default function FactorResultStage({ region, open, onToggle, locked = false, running = false }) {
  const c = region.case;
  const pos = c.correlations?.positive ?? [];
  const neg = c.correlations?.negative ?? [];

  return (
    <CollapsibleStage
      id="stage-result"
      no="STAGE ②"
      title="요인분석 결과"
      sub="데이터 기반 지자체 사회문제 진단"
      open={open}
      onToggle={onToggle}
    >
      <div className="pl-flow-grid pl-flow-2">
        <Card title="상관관계 분석" icon="fa-scale-balanced">
          {!locked ? (
            <div className="pl-corr-grid">
              <div className="pl-corr pl-corr-pos">
                <div className="pl-corr-head">
                  <i className="fa-solid fa-arrow-trend-up" aria-hidden="true"></i> Positive correlation
                </div>
                <ul>
                  {pos.map((p) => (
                    <li key={p}>{p}</li>
                  ))}
                </ul>
              </div>
              <div className="pl-corr pl-corr-neg">
                <div className="pl-corr-head">
                  <i className="fa-solid fa-arrow-trend-down" aria-hidden="true"></i> Negative correlation
                </div>
                <ul>
                  {neg.map((n) => (
                    <li key={n}>{n}</li>
                  ))}
                </ul>
              </div>
            </div>
          ) : (
            <PendingData running={running} text="[요인분석 실행] 후 상관관계 분석 결과가 표시됩니다." />
          )}
        </Card>

        <Card title="문제 유형 진단 결과" icon="fa-stethoscope">
          {!locked ? (
            <>
              <ul className="pl-diag-list">
                {c.problemDiagnosis.map((d) => (
                  <li key={d}>
                    <i className="fa-solid fa-circle-exclamation" aria-hidden="true"></i>
                    {d}
                  </li>
                ))}
              </ul>
              <div className="pl-param-derive">
                <i className="fa-solid fa-arrow-down-long" aria-hidden="true"></i> 파라미터 도출 → 시뮬레이션 입력
              </div>
            </>
          ) : (
            <PendingData running={running} text="[요인분석 실행] 후 문제 유형 진단 결과가 표시됩니다." />
          )}
        </Card>
      </div>
    </CollapsibleStage>
  );
}
