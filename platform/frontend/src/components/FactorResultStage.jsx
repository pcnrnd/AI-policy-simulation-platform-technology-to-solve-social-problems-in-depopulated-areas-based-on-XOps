import Card from "./Card.jsx";
import CollapsibleStage from "./CollapsibleStage.jsx";
import PendingData from "./PendingData.jsx";

// STAGE ② 요인분석 결과
// 데이터 기반 사회문제: +/- 상관 → 문제 유형 진단 → 파라미터 도출.
// UI(카드 구조)는 항상 노출하고, 데이터는 STAGE ① 요인분석 완료(locked=false) 후 공개한다.
//
// unavailableText: 표시할 데이터 자체가 없을 때의 사유(데모 표시 OFF 등). 주어지면 잠금 여부와
// 무관하게 결과 카드를 그 문구로 채운다.
export default function FactorResultStage({
  region,
  open,
  onToggle,
  locked = false,
  running = false,
  unavailableText = null
}) {
  const c = region.case;
  const pos = c.correlations?.positive ?? [];
  const neg = c.correlations?.negative ?? [];
  const blocked = locked || Boolean(unavailableText);

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
          {!blocked ? (
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
            <PendingData
              running={running && !unavailableText}
              text={unavailableText ?? "[요인분석 실행] 후 상관관계 분석 결과가 표시됩니다."}
            />
          )}
        </Card>

        <Card title="문제 유형 진단 결과" icon="fa-stethoscope">
          {!blocked ? (
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
            <PendingData
              running={running && !unavailableText}
              text={unavailableText ?? "[요인분석 실행] 후 문제 유형 진단 결과가 표시됩니다."}
            />
          )}
        </Card>
      </div>
    </CollapsibleStage>
  );
}
