import Card from "./Card.jsx";
import CollapsibleStage from "./CollapsibleStage.jsx";
import PendingData from "./PendingData.jsx";
import { CORRELATION_BASIS_TEXT } from "../lib/realSimulation.js";

// STAGE ② 요인분석 결과
// 상관관계: STAGE ① 요인 기여도의 +/− 방향(두 모드 같은 함수). 문제 유형 진단: 데모는 예시 서사, 실데이터는 ① 결과에서 읽은 신호.
// UI(카드 구조)는 항상 노출하고, 데이터는 STAGE ① 요인분석 완료(locked=false) 후 공개한다. 두 카드는 같은 높이(pl-flow-grid stretch).
// split: { positive, negative } — ① 카드에 보인 요인 행. diagnosis: 진단 문구 배열(빈 배열이면 읽을 신호 없음 안내).
// pendingText: 잠금 중 대신 보일 문구(예: 모델 준비 안내).
export default function FactorResultStage({ open, onToggle, locked = false, running = false, split = null, diagnosis = null,
  pendingText = null, showDerive = false }) {
  const list = (rows) => rows.length ? rows.map((row) => <li key={row.feature}>{row.label}</li>) : <li>해당 요인 없음</li>;
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
        <Card title="상관관계 분석" icon="fa-scale-balanced" dataSource="api" data-slot="sim.result.card-1">
          {!locked && split ? (
            <>
              <div className="pl-corr-grid">
                <div className="pl-corr pl-corr-pos">
                  <div className="pl-corr-head">
                    <i className="fa-solid fa-arrow-trend-up" aria-hidden="true"></i> Positive correlation
                  </div>
                  <ul>{list(split.positive)}</ul>
                </div>
                <div className="pl-corr pl-corr-neg">
                  <div className="pl-corr-head">
                    <i className="fa-solid fa-arrow-trend-down" aria-hidden="true"></i> Negative correlation
                  </div>
                  <ul>{list(split.negative)}</ul>
                </div>
              </div>
              <p className="pl-stat-baseline">{CORRELATION_BASIS_TEXT}</p>
            </>
          ) : (
            <PendingData running={running} text={pendingText ?? "[요인분석 실행] 후 상관관계 분석 결과가 표시됩니다."} />
          )}
        </Card>

        <Card title="문제 유형 진단 결과" icon="fa-stethoscope" dataSource="api" data-slot="sim.result.card-2">
          {!locked && diagnosis ? (
            diagnosis.length ? (
              <ul className="pl-diag-list">
                {diagnosis.map((d) => (
                  <li key={d}>
                    <i className="fa-solid fa-circle-exclamation" aria-hidden="true"></i>
                    {d}
                  </li>
                ))}
              </ul>
            ) : (
              <PendingData text="진단할 신호가 없습니다. 행정동이나 학습 대상을 바꿔 다시 실행하세요." />
            )
          ) : (
            <PendingData running={running} text={pendingText ?? "[요인분석 실행] 후 문제 유형 진단 결과가 표시됩니다."} />
          )}
          {showDerive && !locked && (
            <div className="pl-param-derive">
              <i className="fa-solid fa-arrow-down-long" aria-hidden="true"></i> 파라미터 도출 → 시뮬레이션 입력
            </div>
          )}
        </Card>
      </div>
    </CollapsibleStage>
  );
}
