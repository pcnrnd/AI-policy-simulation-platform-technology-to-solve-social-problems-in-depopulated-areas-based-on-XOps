import CollapsibleStage from "./CollapsibleStage.jsx";

// STAGE ① 사회문제 요인분석 = 분석 설정·실행만(결정 26). 결과 카드는 모두 STAGE ②(FactorResultStage)에 있다.
// 두 모드가 같은 분석 조건 줄·실행 바(진행 단계·[요인분석 실행])를 쓴다.
// conditions: 두 모드 공통 컨트롤(학습 대상·모델·요인 범위·학습 조건 연결)을 호출부가 넘긴다.
// notice: 실행 바 아래 안내(활성 모델 없음 등) — 없으면 비운다.
// runLockTitle: 실행할 수 없을 때의 사유 — 버튼은 aria-disabled 로 남고 클릭은 호출부가 막는다.
// runTitle: 실행 가능한 상태의 보조 설명(예: 조회 실패 뒤 "다시 확인" — 클릭이 목록을 다시 읽는다).
export default function FactorAnalysisStage({
  open,
  onToggle,
  sub,
  runInfo,
  status = "idle",
  stepIndex = 0,
  steps = [],
  onRun,
  conditions = null,
  notice = null,
  runLockTitle = null,
  runTitle = null,
  versionTitle = undefined
}) {
  const running = status === "running";
  const done = status === "done";
  return (
    <CollapsibleStage id="stage-factor" no="STAGE ①" title="사회문제 요인분석" sub={sub} open={open} onToggle={onToggle}>
      {/* 분석 실행 바: 분석 조건 → 입력 요약 → 단계 진행 → 실행 버튼 */}
      {conditions}
      <div className="pl-run-bar">
        <div className="pl-run-info">
          <i className="fa-solid fa-flask" aria-hidden="true"></i>
          <div>
            <strong>요인분석 실행</strong>
            <p title={versionTitle}>{runInfo}</p>
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
          title={runLockTitle ?? runTitle ?? undefined}
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

      {notice}
    </CollapsibleStage>
  );
}

