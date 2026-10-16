import Card from "./Card.jsx";
import CollapsibleStage from "./CollapsibleStage.jsx";
import PendingData from "./PendingData.jsx";
import { formatValue, FACTOR_EXPLAIN_TEXT, FACTOR_FOOTNOTE } from "../lib/realSimulation.js";

// STAGE ① 사회문제 요인분석
// 두 모드가 같은 실행 바·단계 칩·카드 3개(기준선 제목)·근거 자리를 쓰고, 내용은 활성 요인분석 모델의 결과다.
// view: factorAnalysisView(결과) — 있으면 카드 3개를 채운다. 없으면 placeholders[i](실행 전·실행 중·안내)를 둔다.
// conditions·evidence: 두 모드 공통 컨트롤(학습 대상·행정동 select, 계산 근거)을 호출부가 넘긴다.
// runLockTitle: 실행할 수 없을 때의 사유 — 버튼은 aria-disabled 로 남고 클릭은 호출부가 막는다.
export default function FactorAnalysisStage({
  open,
  onToggle,
  sub,
  runInfo,
  status = "idle",
  stepIndex = 0,
  steps = [],
  onRun,
  view = null,
  placeholders = [],
  sourceLabel,
  conditions = null,
  evidence = null,
  runLockTitle = null
}) {
  const running = status === "running";
  const done = status === "done";
  const card = (index, title, body) => (
    <Card title={title} icon={CARD_ICONS[index]} dataSource="api" sourceLabel={sourceLabel} data-slot={`sim.factor.card-${index + 1}`}>
      {view ? body : placeholders[index] ?? <PendingData running={running} />}
    </Card>
  );
  const signed = (value) => `${value > 0 ? "+" : ""}${formatValue(value)}`;

  return (
    <CollapsibleStage id="stage-factor" no="STAGE ①" title="사회문제 요인분석" sub={sub} open={open} onToggle={onToggle}>
      {/* 분석 실행 바: 분석 조건 → 입력 요약 → 단계 진행 → 실행 버튼 */}
      {conditions}
      <div className="pl-run-bar">
        <div className="pl-run-info">
          <i className="fa-solid fa-flask" aria-hidden="true"></i>
          <div>
            <strong>요인분석 실행</strong>
            <p>{runInfo}</p>
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
                className={"pl-run-step" + (isDone ? " is-done" : "") + (isActive ? " is-active" : "")}
                aria-current={isActive ? "step" : undefined}
              >
                <i
                  className={"fa-solid " + (isDone ? "fa-circle-check" : isActive ? "fa-circle-dot" : s.icon)}
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
        {/* AI + 인구학적 요인 분석 결과 — 요인 기여도(±, |값| 내림차순) */}
        {card(0, "AI + 인구학적 요인 분석 결과", view && (
            <>
              <ul className="pl-factor-list">
                {view.factors.map((f) => (
                  <li key={f.feature} className={f.direction === "-" ? "neg" : "pos"}>
                    <span className="pl-factor-sign">{f.direction}</span>
                    {f.label} · 기여도 {signed(f.value)} {view.unit} · 관측 {formatValue(f.input)}{f.inputUnit ? ` ${f.inputUnit}` : ""}
                  </li>
                ))}
              </ul>
              {view.more > 0 && <p className="pl-stat-baseline">외 {view.more}개 요인은 계산 근거에서 확인</p>}
              <div className="pl-source-note">
                <i className="fa-solid fa-database" aria-hidden="true"></i> {view.basis}
              </div>
              <p className="pl-stat-baseline">{FACTOR_EXPLAIN_TEXT}</p>
              <p className="pl-stat-baseline">{FACTOR_FOOTNOTE}</p>
            </>
        ))}

        {/* 모델 정보 — 기여도 분해식과 활성 버전 */}
        {card(1, "XAI · 딥러닝 예측 모델", view && (
            <>
              <div className="pl-model-box pl-model-xai">
                <span className="pl-model-tag">기여도 분해</span>
                <strong>요인별 기여도 합 + 기준값 {formatValue(view.model.intercept)} {view.unit}</strong>
                <p>= 모델 추정값 {formatValue(view.model.prediction)} {view.unit}</p>
              </div>
              <div className="pl-model-box pl-model-dl">
                <span className="pl-model-tag">학습 모델</span>
                <strong>요인분석 · 활성 버전 {view.model.version}</strong>
                <div className="pl-model-chips">
                  {view.model.features.map((m) => (
                    <span className="pl-chip" key={m}>{m}</span>
                  ))}
                </div>
              </div>
            </>
        ))}

        {/* 검증 결과 (결과 융합) */}
        {card(2, "통계적 모델 분석 결과", view && (
            <>
              <div className="pl-fusion-mark">
                <i className="fa-solid fa-plus" aria-hidden="true"></i> 결과 융합
              </div>
              <p className="pl-stat-baseline">
                검증 평균 절대 오차 {formatValue(view.validation.mae)} {view.unit} · 기준선 {formatValue(view.validation.baselineMae)} {view.unit}
                {view.validation.baselineName ? `(${view.validation.baselineName})` : ""}
              </p>
              <div className="pl-stage-label">분석 대상</div>
              <ul className="pl-mini-list">
                {view.validation.targets.map((t) => (
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

const CARD_ICONS = ["fa-list-check", "fa-microchip", "fa-chart-column"];
