import { useEffect, useState } from "react";
import { useAppState } from "../context/AppStateContext.jsx";
import { PIPELINE_STEPS } from "../constants/pipeline.js";
import { apiGet } from "../lib/api.js";

// XOps 4단계 연속성 리본 — 데이터 준비 → 모니터링 → 재학습·배포 → 시뮬레이션.
// 단계 배지는 전역 상태(AppStateContext)에서 파생만 한다. 새 상태를 저장하지 않으므로
// 어느 탭에서 보든 같은 값이 나온다. 데모 표시 OFF에서는 배지와 상태 색을 숨긴다.
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
  ready: "fa-circle-dot",
  active: "fa-spinner fa-spin",
  alert: "fa-triangle-exclamation"
};

// 파생 규칙 한 곳 — 상태 조합에서 4단계의 색조(tone)와 배지 문구를 만든다.
function deriveFlowSteps({
  catalogCount,
  mockDataVisible,
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
      // 카탈로그 조회 실패 시 배지 생략 (에러 아님).
      // 목업 표시 OFF일 때도 숫자를 감춘다 — 이 건수는 mock_data.json 시드가 섞인 값이다(단계 상태는 유지).
      badge: catalogCount === null || !mockDataVisible ? null : `카탈로그 ${catalogCount}건`
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
      // 승급만으로 done·'적용'이라 부르지 않는다. SimulatorPage는 아직 modelStore/pipelineResult를 읽지 않아
      // 승급 버전이 시뮬레이션에 반영됐다는 근거가 없다 — 확인 가능(ready)까지만 표시한다.
      // 설계 §5(모델 버전 전파)가 구현되면 시뮬레이터가 쓰는 버전으로 done 판정을 연결한다.
      tone: promoted ? "ready" : "idle",
      // 목업 표시 OFF면 버전 배지도 렌더하지 않는다 — ① 카탈로그 건수와 같은 기준(단계 상태·tone·CTA는 유지).
      // modelStore는 MODEL_STORE 시드로 시작해 /api/v3/orchestration/models 응답으로만 부분 갱신되고,
      // 행 단위 출처 표기가 없어 렌더 시점에 시드와 백엔드 값을 구분할 수 없다 → 시드를 실측처럼 보이지 않게 감춘다.
      badge: servingVersion && mockDataVisible ? `모델 ${servingVersion} ${promoted ? "확인 가능" : "기준"}` : null
    }
  ];
}

export default function XopsFlowRibbon() {
  const {
    activeTab,
    navigateToTab,
    addConsoleLog,
    mockDataVisible,
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
    mockDataVisible,
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
          const badge = mockDataVisible ? step.badge : null;
          const tone = mockDataVisible ? step.tone : "idle";
          return (
            <li className="xops-ribbon-item" key={step.id}>
              <button
                type="button"
                className={`xops-ribbon-step is-${tone}${isCurrent ? " is-current" : ""}`}
                onClick={() => navigateToTab(step.id)}
                aria-current={isCurrent ? "true" : undefined}
                aria-label={`${step.no} ${step.label}${badge ? `, ${badge}` : ""}${isCurrent ? ", 현재 탭" : ""}`}
              >
                <span className="xops-ribbon-no" aria-hidden="true">
                  {step.no}
                </span>
                <span className="xops-ribbon-text">
                  <span className="xops-ribbon-label">
                    <i className={`fa-solid ${step.icon}`} aria-hidden="true"></i> {step.label}
                  </span>
                  {badge && (
                    <span className="xops-ribbon-badge">
                      {TONE_ICON[tone] && (
                        <i className={`fa-solid ${TONE_ICON[tone]}`} aria-hidden="true"></i>
                      )}
                      {badge}
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
