import Card from "./Card.jsx";
import CollapsibleStage from "./CollapsibleStage.jsx";

// STAGE ② 요인분석 결과
// 데이터 기반 사회문제: +/- 상관 → 문제 유형 진단 → 파라미터 도출.
// STAGE ①의 요인분석이 완료(locked=false)되어야 결과가 공개된다.
export default function FactorResultStage({ region, open, onToggle, locked = false, running = false, onRun }) {
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
      <div className="pl-result-zone">
      <div className={"pl-flow-grid pl-flow-2" + (locked ? " pl-locked" : "")}>
        <Card title="상관관계 분석" icon="fa-scale-balanced">
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
        </Card>

        <Card title="문제 유형 진단 결과" icon="fa-stethoscope">
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
        </Card>
      </div>

      {locked && (
        <div className="pl-result-overlay" role="status">
          <i
            className={"fa-solid " + (running ? "fa-spinner fa-spin" : "fa-lock")}
            aria-hidden="true"
          ></i>
          <strong>{running ? "요인분석 진행 중..." : "요인분석 결과 대기"}</strong>
          <p>
            {running
              ? "분석이 완료되면 상관관계·문제진단 결과가 표시됩니다."
              : "STAGE ①에서 요인분석을 실행하면 결과가 공개됩니다."}
          </p>
          {!running && (
            <button type="button" className="btn btn-primary" onClick={onRun}>
              <i className="fa-solid fa-play"></i> 요인분석 실행
            </button>
          )}
        </div>
      )}
      </div>
    </CollapsibleStage>
  );
}
