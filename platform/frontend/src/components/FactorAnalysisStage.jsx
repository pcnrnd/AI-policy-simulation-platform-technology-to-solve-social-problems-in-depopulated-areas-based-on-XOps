import Card from "./Card.jsx";
import CollapsibleStage from "./CollapsibleStage.jsx";
import PendingData from "./PendingData.jsx";

// STAGE ① 사회문제 요인분석
// region.case 기반: AI+인구학적 요인 → XAI(SHAP)+딥러닝 모델 → 통계 분석 융합.
// UI(카드 구조)는 항상 노출하고, 데이터는 [요인분석 실행] 완료 후에만 공개한다.
// 아래 시드 사례(region.case)는 데모 표시 전용이고, 실데이터 표시는 view 로 같은 자리를 채운다.
//
// 두 모드가 같은 실행 바·카드 3개(제목 동일)·근거 자리를 쓴다.
// view(실데이터 표시): { sub, runInfo, cards: [body] × 3 } — 주어지면 카드 내용을 그것으로 채운다(제목은 그대로).
// conditions·evidence: 두 모드 공통 컨트롤(학습 대상·행정동 select, 계산 근거)을 호출부가 넘긴다.
// runLockTitle: 실행할 수 없을 때의 사유 — 버튼은 aria-disabled 로 남고 클릭은 호출부가 막는다.
export default function FactorAnalysisStage({
  region,
  open,
  onToggle,
  status = "idle",
  stepIndex = 0,
  steps = [],
  onRun,
  view = null,
  conditions = null,
  evidence = null,
  runLockTitle = null
}) {
  const c = region.case;
  const factors = c.factorAnalysis ?? [];
  const ai = c.aiModel ?? {};
  const running = status === "running";
  const done = status === "done";
  const card = (index, title, demoBody) => (
    <Card title={title} icon={DEMO_CARD_ICONS[index]} dataSource={view ? "api" : "mock"} data-slot={`sim.factor.card-${index + 1}`}>
      {view ? view.cards[index] : done ? demoBody : <PendingData running={running} />}
    </Card>
  );

  return (
    <CollapsibleStage
      id="stage-factor"
      no="STAGE ①"
      title="사회문제 요인분석"
      sub={view ? view.sub : `${region.name} · ${region.theme}`}
      open={open}
      onToggle={onToggle}
    >
      {/* 분석 실행 바: 분석 조건 → 입력 요약 → 단계 진행 → 실행 버튼 */}
      {conditions}
      <div className="pl-run-bar">
        <div className="pl-run-info">
          <i className="fa-solid fa-flask" aria-hidden="true"></i>
          <div>
            <strong>요인분석 실행</strong>
            <p>
              {view
                ? view.runInfo
                : `연계 소스 ${c.dataSources.length}개 · ${ai.xai ?? "XAI"} · ${(ai.deepLearning ?? []).join(" / ")}`}
            </p>
          </div>
        </div>

        <div className="pl-run-steps" role="status" aria-live="polite" aria-label="요인분석 진행 단계">
          {steps.map((s, i) => {
            const stepNo = i + 1;
            const isDone = done || (running && stepIndex > stepNo);
            const isActive = running && stepIndex === stepNo;
            return (
              <span
                key={s.key}
                className={
                  "pl-run-step" + (isDone ? " is-done" : "") + (isActive ? " is-active" : "")
                }
                aria-current={isActive ? "step" : undefined}
              >
                <i
                  className={
                    "fa-solid " +
                    (isDone ? "fa-circle-check" : isActive ? "fa-circle-dot" : s.icon)
                  }
                  aria-hidden="true"
                ></i>
                {s.label}
              </span>
            );
          })}
        </div>

        <button
          type="button"
          className={"btn " + (done ? "btn-secondary" : "btn-primary")}
          onClick={onRun}
          disabled={running}
          aria-disabled={runLockTitle ? true : undefined}
          title={runLockTitle ?? undefined}
        >
          {running ? (
            <>
              <i className="fa-solid fa-spinner fa-spin" aria-hidden="true"></i> 분석 중...
            </>
          ) : done ? (
            <>
              <i className="fa-solid fa-rotate-right" aria-hidden="true"></i> 재실행
            </>
          ) : (
            <>
              <i className="fa-solid fa-play" aria-hidden="true"></i> 요인분석 실행
            </>
          )}
        </button>
      </div>

      {/* 결과 영역: 카드 구조는 상시 노출, 데이터는 분석 완료 후 표시 */}
      <div className="pl-flow-grid pl-flow-3">
        {/* AI + 인구학적 요인 분석 결과 */}
        {card(0, "AI + 인구학적 요인 분석 결과", (
            <>
              <ul className="pl-factor-list">
                {factors.map((f) => (
                  <li key={f.label} className={f.direction === "-" ? "neg" : "pos"}>
                    <span className="pl-factor-sign">{f.direction}</span>
                    {f.label}
                  </li>
                ))}
              </ul>
              <div className="pl-source-note">
                <i className="fa-solid fa-database" aria-hidden="true"></i> 연계 데이터:{" "}
                {c.dataSources.length}개 소스 (주민등록·통신·카드 등)
              </div>
            </>
        ))}

        {/* XAI + 딥러닝 예측 모델 */}
        {card(1, "XAI · 딥러닝 예측 모델", (
            <>
              <div className="pl-model-box pl-model-xai">
                <span className="pl-model-tag">XAI 모듈</span>
                <strong>{ai.xai}</strong>
                <p>주요 요인 기여도(Shapley value) 분해</p>
              </div>
              <div className="pl-model-box pl-model-dl">
                <span className="pl-model-tag">딥러닝 예측 모델</span>
                <div className="pl-model-chips">
                  {(ai.deepLearning ?? []).map((m) => (
                    <span className="pl-chip" key={m}>{m}</span>
                  ))}
                </div>
              </div>
            </>
        ))}

        {/* 통계적 모델 분석 결과 (결과 융합) */}
        {card(2, "통계적 모델 분석 결과", (
            <>
              <div className="pl-fusion-mark">
                <i className="fa-solid fa-plus" aria-hidden="true"></i> 결과 융합
              </div>
              <p className="pl-stat-baseline">{ai.statBaseline}</p>
              <div className="pl-stage-label">분석 대상</div>
              <ul className="pl-mini-list">
                {c.analysisTargets.map((t) => (
                  <li key={t}>{t}</li>
                ))}
              </ul>
            </>
        ))}
      </div>
      {evidence}
    </CollapsibleStage>
  );
}

const DEMO_CARD_ICONS = ["fa-list-check", "fa-microchip", "fa-chart-column"];
