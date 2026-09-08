import { useEffect, useState } from "react";
import { useAppState } from "../context/AppStateContext.jsx";
import { PIPELINE_STEPS } from "../constants/pipeline.js";
import { apiGet } from "../lib/api.js";

// XOps 4단계 연속성 리본 — 데이터 준비 → 모니터링 → 재학습·배포 → 시뮬레이션.
// 단계 배지는 전역 상태(AppStateContext)에서 파생만 한다. 새 상태를 저장하지 않으므로
// 어느 탭에서 보든 같은 값이 나오고, 목업 표시 스위치(.mock-values-hidden)의
// 가림 대상도 아니다(전용 xops- 접두사 클래스 사용).
export const FLOW_TAB_IDS = ["tab-dataops", "tab-mlops-monitor", "tab-mlops-orch", "tab-simulator"];

const CATALOG_URL = "/api/v3/dataops/catalog";
const DEFAULT_MODEL_ID = "population-forecast";

// 파이프라인 종료 상태(승급 성공 제외) → 배지 문구
const TERMINAL_BADGE = {
  rejected: "승급 반려",
  rolled_back: "자동 롤백",
  debounced: "실행 조정",
  failed: "실행 실패"
};

const TONE_ICON = {
  done: "fa-circle-check",
  active: "fa-spinner fa-spin",
  alert: "fa-triangle-exclamation"
};

// 파생 규칙 한 곳 — 상태 조합에서 4단계의 색조(tone)와 배지 문구를 만든다.
function deriveFlowSteps({
  catalogCount,
  driftInjected,
  metricOverrides,
  modelStore,
  pipelineRunning,
  pipelineScheduled,
  pipelineStep,
  pipelineRun,
  pipelineResult
}) {
  const pipelineBusy = pipelineRunning || pipelineScheduled;
  // 실행/예약 중에는 직전 실행의 종료 상태를 흐름에 노출하지 않는다(오케스트레이터 화면과 동일 규칙).
  const terminalState = pipelineBusy ? null : pipelineResult?.state ?? null;
  const promoted = terminalState === "succeeded";
  const metricsRefreshed = Object.keys(metricOverrides).length > 0;
  const servingVersion =
    modelStore.find(
      (model) => model.modelId === (pipelineRun?.model ?? DEFAULT_MODEL_ID) && model.status === "운영"
    )?.version ?? null;
  const runningStep = Math.min(Math.max(pipelineStep, 1), PIPELINE_STEPS.length);

  return [
    {
      id: "tab-dataops",
      no: "①",
      label: "데이터 준비",
      icon: "fa-database",
      tone: catalogCount > 0 ? "done" : "idle",
      // 카탈로그 조회 실패 시 배지 생략 (에러 아님)
      badge: catalogCount === null ? null : `카탈로그 ${catalogCount}건`
    },
    {
      id: "tab-mlops-monitor",
      no: "②",
      label: "모니터링",
      icon: "fa-gauge-high",
      tone: driftInjected ? "alert" : metricsRefreshed ? "done" : "idle",
      badge: driftInjected ? "드리프트 감지" : metricsRefreshed ? "지표 갱신" : "감시 중"
    },
    {
      id: "tab-mlops-orch",
      no: "③",
      label: "재학습·배포",
      icon: "fa-diagram-project",
      tone: pipelineBusy ? "active" : promoted ? "done" : terminalState ? "alert" : "idle",
      badge: pipelineScheduled
        ? "재학습 예약"
        : pipelineRunning
          ? `재학습 중 (${runningStep}/${PIPELINE_STEPS.length})`
          : promoted
            ? `승급 ${pipelineResult?.active_version ?? pipelineRun?.candidateVersion ?? "완료"}`
            : TERMINAL_BADGE[terminalState] ?? "대기"
    },
    {
      id: "tab-simulator",
      no: "④",
      label: "시뮬레이션",
      icon: "fa-map-location-dot",
      tone: promoted ? "done" : "idle",
      badge: servingVersion ? `모델 ${servingVersion} ${promoted ? "적용" : "기준"}` : null
    }
  ];
}

export default function XopsFlowRibbon() {
  const {
    activeTab,
    navigateToTab,
    addConsoleLog,
    driftInjected,
    metricOverrides,
    modelStore,
    pipelineRunning,
    pipelineScheduled,
    pipelineStep,
    pipelineRun,
    pipelineResult
  } = useAppState();

  // 카탈로그 건수만 마운트 시 1회 조회한다. 실패하면 배지 없이 진행하고 콘솔 로그만 남긴다.
  const [catalogCount, setCatalogCount] = useState(null);
  useEffect(() => {
    let alive = true;
    apiGet(CATALOG_URL)
      .then((list) => {
        if (alive && Array.isArray(list)) setCatalogCount(list.length);
      })
      .catch((err) => {
        if (alive) {
          addConsoleLog(
            `WARN: XOps 흐름 리본 카탈로그 배지 생략 — ${err?.message ?? "알 수 없는 오류"}`
          );
        }
      });
    return () => {
      alive = false;
    };
  }, [addConsoleLog]);

  const steps = deriveFlowSteps({
    catalogCount,
    driftInjected,
    metricOverrides,
    modelStore,
    pipelineRunning,
    pipelineScheduled,
    pipelineStep,
    pipelineRun,
    pipelineResult
  });

  return (
    <nav className="xops-ribbon" aria-label="XOps 진행 흐름">
      <span className="xops-ribbon-cap">
        <i className="fa-solid fa-route" aria-hidden="true"></i> XOps 흐름
      </span>
      <ol className="xops-ribbon-steps">
        {steps.map((step, idx) => {
          const isCurrent = step.id === activeTab;
          return (
            <li className="xops-ribbon-item" key={step.id}>
              <button
                type="button"
                className={`xops-ribbon-step is-${step.tone}${isCurrent ? " is-current" : ""}`}
                onClick={() => navigateToTab(step.id)}
                aria-current={isCurrent ? "true" : undefined}
                aria-label={`${step.no} ${step.label}${step.badge ? `, ${step.badge}` : ""}${isCurrent ? ", 현재 탭" : ""}`}
              >
                <span className="xops-ribbon-no" aria-hidden="true">
                  {step.no}
                </span>
                <span className="xops-ribbon-text">
                  <span className="xops-ribbon-label">
                    <i className={`fa-solid ${step.icon}`} aria-hidden="true"></i> {step.label}
                  </span>
                  {step.badge && (
                    <span className="xops-ribbon-badge">
                      {TONE_ICON[step.tone] && (
                        <i className={`fa-solid ${TONE_ICON[step.tone]}`} aria-hidden="true"></i>
                      )}
                      {step.badge}
                    </span>
                  )}
                </span>
              </button>
              {idx < steps.length - 1 && (
                <i className="fa-solid fa-chevron-right xops-ribbon-arrow" aria-hidden="true"></i>
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
