import Card from "./Card.jsx";
import CollapsibleStage from "./CollapsibleStage.jsx";
import PendingData from "./PendingData.jsx";

// STAGE ② 요인분석 결과
// 상관관계: STAGE ① 요인 기여도의 +/− 방향(두 모드 같은 함수). 문제 유형 진단: 데모는 예시 서사, 실데이터는 ① 결과에서 읽은 신호.
// UI(카드 구조)는 항상 노출하고, 데이터는 STAGE ① 요인분석 완료(locked=false) 후 공개한다. 두 카드는 같은 높이(pl-flow-grid stretch).
// split: { positive, negative } — ① 카드에 보인 요인 행.
// diagnosis: 두 모드 같은 diagnosisView 결과({ kind, change, rows, items, indicators }). 아래 ③ 연결 줄도 두 모드 같다.
// pendingText: 잠금 중 대신 보일 문구(예: 모델 준비 안내). diagnosisSource: 진단 값의 출처("mock" 데모 서사 | "api").
export default function FactorResultStage({ open, onToggle, locked = false, running = false, split = null, diagnosis = null,
  diagnosisSource = "api", pendingText = null }) {
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
                    <i className="fa-solid fa-arrow-trend-up" aria-hidden="true"></i> 양의 상관
                  </div>
                  <ul>{list(split.positive)}</ul>
                </div>
                <div className="pl-corr pl-corr-neg">
                  <div className="pl-corr-head">
                    <i className="fa-solid fa-arrow-trend-down" aria-hidden="true"></i> 음의 상관
                  </div>
                  <ul>{list(split.negative)}</ul>
                </div>
              </div>
            </>
          ) : (
            <PendingData running={running} text={pendingText ?? "[요인분석 실행] 후 상관관계 분석 결과가 표시됩니다."} />
          )}
        </Card>

        <Card title="문제 유형 진단 결과" icon="fa-stethoscope" dataSource={diagnosisSource} data-slot="sim.result.card-2">
          {!locked && diagnosis ? (
            <DiagnosisBody diagnosis={diagnosis} />
          ) : (
            <PendingData running={running} text={pendingText ?? "[요인분석 실행] 후 문제 유형 진단 결과가 표시됩니다."} />
          )}
          {!locked && diagnosis && (
            <div className="pl-param-derive">
              <i className="fa-solid fa-arrow-down-long" aria-hidden="true"></i> ③ 시뮬레이션으로 연결
            </div>
          )}
        </Card>
      </div>
    </CollapsibleStage>
  );
}

const KIND_TONE = { 감소형: "down", 증가형: "up", 정체형: "flat" };

/** 문제 유형 진단 본문 — 두 모드 같은 구조: 유형·변화율(머리) → 요인 행(이름·기여도) → 점검 항목 → 지표별 상태. */
function DiagnosisBody({ diagnosis: d }) {
  return (
    <div className="pl-diag">
      <div className={`pl-diag-head is-${KIND_TONE[d.kind] ?? "none"}`}>
        {d.kind && <span className="pl-diag-kind">{d.kind}</span>}
        <strong className="pl-diag-change">{d.change}</strong>
        <span className="pl-diag-basis">전년 같은 달 대비</span>
      </div>
      {d.rows?.length > 0 && (
        <dl className="pl-diag-rows">
          {d.rows.map((r) => (
            <div key={r.name} className={`pl-diag-row is-${r.dir}`}>
              <dt>{r.name}</dt>
              <dd className="pl-diag-label">{r.label}</dd>
              <dd className="pl-diag-value">{r.value}</dd>
            </div>
          ))}
        </dl>
      )}
      <div className="pl-diag-checks">
        <span className="pl-diag-tag">점검 항목</span>
        {d.items?.length ? d.items.map((text) => <span key={text} className="pl-diag-check">{text}</span>)
          : <span className="pl-diag-none">해당 항목 없음</span>}
      </div>
      {d.indicators?.length > 0 && (
        <div className="pl-diag-indicators">
          <span className="pl-diag-tag">지표별</span>
          {d.indicators.map((e) => (
            <span key={e.target} className={`pl-diag-chip is-${KIND_TONE[e.kind] ?? "none"}`}>
              {e.name}{e.kind && <b>{e.kind}</b>}<span>{e.change}</span>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
