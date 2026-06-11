import { useCallback } from "react";

// 단계 흐름 네비게이터 — 클릭 시 해당 섹션(anchor)으로 부드럽게 스크롤한다.
// stages 미지정 시 정책 시뮬레이터의 4단계가 기본값(기존 호출부 호환).
const DEFAULT_STAGES = [
  { id: "stage-factor", no: "①", label: "사회문제 요인분석", icon: "fa-magnifying-glass-chart" },
  { id: "stage-result", no: "②", label: "요인분석 결과", icon: "fa-diagram-next" },
  { id: "stage-sim", no: "③", label: "시뮬레이션", icon: "fa-wand-magic-sparkles" },
  { id: "stage-report", no: "④", label: "리포팅", icon: "fa-file-invoice" }
];

export default function PipelineStepper({
  stages = DEFAULT_STAGES,
  activeId,
  doneIds = [],
  onJump,
  ariaLabel = "분석 파이프라인 단계"
}) {
  const jump = useCallback(
    (id) => {
      if (onJump) {
        onJump(id);
        return;
      }
      const el = document.getElementById(id);
      if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
    },
    [onJump]
  );

  return (
    <nav className="pipeline-stepper" aria-label={ariaLabel}>
      {stages.map((s, i) => {
        const isDone = doneIds.includes(s.id);
        return (
        <div className="pipeline-step-wrap" key={s.id}>
          <button
            type="button"
            className={
              "pipeline-step" + (activeId === s.id ? " active" : "") + (isDone ? " done" : "")
            }
            onClick={() => jump(s.id)}
            aria-current={activeId === s.id ? "step" : undefined}
          >
            <span className="pipeline-step-no">{s.no}</span>
            <span className="pipeline-step-body">
              <i className={`fa-solid ${s.icon}`} aria-hidden="true"></i>
              <span className="pipeline-step-label">{s.label}</span>
            </span>
            {isDone && (
              <i className="fa-solid fa-circle-check pipeline-step-check" aria-label="완료"></i>
            )}
          </button>
          {i < stages.length - 1 && (
            <i className="fa-solid fa-chevron-right pipeline-step-arrow" aria-hidden="true"></i>
          )}
        </div>
        );
      })}
    </nav>
  );
}
