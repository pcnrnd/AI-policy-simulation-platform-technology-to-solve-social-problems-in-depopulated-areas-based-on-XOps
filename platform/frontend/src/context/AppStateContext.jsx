import { createContext, useContext, useState, useCallback, useRef, useEffect } from "react";
import mockData from "../assets/mock_data.json";
import { PIPELINE_STEPS } from "../constants/pipeline.js";
import { RETRAIN_PIPELINES, MODEL_STORE } from "../constants/models.js";
import { apiGet, apiSend } from "../lib/api.js";

const AppStateContext = createContext(null);

const STEP_DELAY_MS = 2500;
const ALERT_AUTO_DISMISS_MS = 5000;
const MOCK_DATA_STORAGE_KEY = "decline-poc-mock-data";

// 목업 데이터 전역 표시 여부. 저장값이 없거나 손상됐거나 스토리지를 못 쓰면 기본 노출(true).
function readMockDataVisible() {
  try {
    return localStorage.getItem(MOCK_DATA_STORAGE_KEY) !== "false";
  } catch {
    return true;
  }
}

const LOG_LEVEL_CLASS = {
  INFO: "log-info",
  WARN: "log-warn",
  ERROR: "log-err",
  SUCCESS: "log-success"
};

// 호출부에 남아 있는 위치 인자와 메시지 접두사를 함께 지원한다.
// 접두사가 있으면 이를 우선해 `ERROR: ...`, false, true 같은 기존 호출도
// 오류로 정확히 분류하고, 화면에서는 레벨을 한 번만 표시하도록 본문에서 제거한다.
function normalizeConsoleMessage(message, isSystem = false, isWarning = false) {
  let normalizedMessage = String(message ?? "").trim();
  const prefixMatch = normalizedMessage.match(/^(INFO|WARN(?:ING)?|ERROR|SUCCESS|ALERT)\s*:\s*/i);
  const explicitLevel = prefixMatch?.[1]?.toUpperCase();
  const level =
    explicitLevel === "WARNING" || explicitLevel === "ALERT"
      ? "WARN"
      : explicitLevel || (isSystem ? "ERROR" : isWarning ? "WARN" : "INFO");

  if (prefixMatch) normalizedMessage = normalizedMessage.slice(prefixMatch[0].length).trimStart();

  return {
    message: normalizedMessage,
    level,
    type: LOG_LEVEL_CLASS[level]
  };
}

const formatYmd = (date) =>
  `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;

// Model Store 승급 규칙 — 신규 버전을 운영으로 맨 앞에 등록하고 직전 운영 버전은 '이전'으로 강등한다.
// 원본 배열은 건드리지 않고 새 배열을 반환한다. accuracy를 넘기지 않으면 직전 운영 지표에서 파생한다.
function promoteVersion(store, modelId, version, accuracy, registeredAt) {
  const prevServing = store.find((m) => m.modelId === modelId && m.status === "운영");
  const acc = accuracy ?? Number(((prevServing?.accuracy ?? 0.88) + 0.033).toFixed(3));
  const demoted = store
    .map((m) => (m.modelId === modelId && m.status === "운영" ? { ...m, status: "이전" } : m))
    .filter((m) => !(m.modelId === modelId && m.version === version));
  return [
    {
      modelId,
      version,
      dataVersion: "ds-v13",
      params: "auto-tuned (재학습 파이프라인)",
      accuracy: acc,
      status: "운영",
      registeredAt: registeredAt ?? formatYmd(new Date())
    },
    ...demoted
  ];
}

export function AppStateProvider({ children }) {
  const [ready] = useState(true);
  const [appData] = useState(mockData);

  // 목업 데이터 노출 스위치 — 끄면 화면에서 숨기기만 하고 mock_data.json 원본은 그대로 둔다.
  const [mockDataVisible, setMockDataVisible] = useState(readMockDataVisible);

  const [activeTab, setActiveTab] = useState("tab-overview");
  const [tabFocusRequest, setTabFocusRequest] = useState(0);
  const [currentRegion, setCurrentRegion] = useState(mockData.regions[0]);

  const [welfareWeight, setWelfareWeight] = useState(50);
  const [industryWeight, setIndustryWeight] = useState(50);
  const [housingWeight, setHousingWeight] = useState(50);
  const [budgetTotal, setBudgetTotal] = useState(600); // 총 예산(억) — 시뮬레이션 제약요소

  const [driftInjected, setDriftInjected] = useState(false);
  const [pipelineRunning, setPipelineRunning] = useState(false);
  const [pipelineScheduled, setPipelineScheduled] = useState(false);
  const [pipelineStep, setPipelineStep] = useState(0);
  // 현재(또는 마지막) 파이프라인 실행 식별 정보 — 실행 ID·파이프라인·대상 모델·실험·트리거
  const [pipelineRun, setPipelineRun] = useState(null);
  // 파이프라인별 마지막 실행 결과 (pipelineId → { runId, finishedAt, result })
  const [pipelineHistory, setPipelineHistory] = useState({});
  // Model Store — 모델·실험 버전 이력 (승급 완료 시 신규 운영 버전 추가)
  const [modelStore, setModelStore] = useState(MODEL_STORE);
  // 백엔드 오케스트레이션 이벤트 결과(PipelineRun) — 애니메이션 완료 시 실제 승급/롤백 반영
  const [pipelineResult, setPipelineResult] = useState(null);

  // 승급 지표는 모델별로 보관한다. 다른 모델 실행이 인구예측 대시보드 값을 덮지 않게 한다.
  const [metricOverrides, setMetricOverrides] = useState({});
  const f1Override = metricOverrides["population-forecast"]?.f1 ?? null;

  const [consoleLogs, setConsoleLogs] = useState([
    {
      time: "13:00:00",
      ...normalizeConsoleMessage("INFO: MLOps 오케스트레이션 대기 중...")
    }
  ]);
  const [alerts, setAlerts] = useState([]);
  // 헤더 벨 알림 이력 — 팝업과 달리 사라지지 않고 최근 N건 보존
  const [notifications, setNotifications] = useState([]);
  const [unreadCount, setUnreadCount] = useState(0);

  const pipelineTimerRef = useRef(null);
  const pipelineRequestRef = useRef(0);
  const driftStartTimerRef = useRef(null);

  const navigateToTab = useCallback((tabId) => {
    setActiveTab(tabId);
    setTabFocusRequest((request) => request + 1);
  }, []);

  const toggleMockDataVisible = useCallback(() => setMockDataVisible((prev) => !prev), []);

  useEffect(() => {
    try {
      localStorage.setItem(MOCK_DATA_STORAGE_KEY, String(mockDataVisible));
    } catch {
      // 저장공간 초과·프라이빗 모드 등 — 이번 세션 동안은 메모리 상태로 계속 동작
    }
  }, [mockDataVisible]);

  const addConsoleLog = useCallback((message, isSystem = false, isWarning = false) => {
    const now = new Date();
    const time = `${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}:${String(now.getSeconds()).padStart(2, "0")}`;
    const normalizedLog = normalizeConsoleMessage(message, isSystem, isWarning);
    setConsoleLogs((prev) => [...prev, { time, ...normalizedLog }]);
  }, []);

  // 새로고침 시 modelStore는 상수(MODEL_STORE)로 리셋되므로, 백엔드가 보관 중인 현재 운영 버전에 맞춘다.
  // 동기화에 실패하면 상수 이력을 그대로 유지한다(화면은 계속 동작).
  useEffect(() => {
    let alive = true;
    apiGet("/api/v3/orchestration/models")
      .then((models) => {
        if (!alive || !Array.isArray(models)) return;
        setModelStore((prev) =>
          models.reduce((store, m) => {
            if (typeof m?.model_id !== "string" || typeof m?.version !== "string") return store;
            const alreadyServing = store.some(
              (row) => row.modelId === m.model_id && row.status === "운영" && row.version === m.version
            );
            // 지표는 넘기지 않는다 — 백엔드 metrics는 승급 후보가 아닌 현재 모델 지표라 파생 로직에 맡긴다.
            return alreadyServing ? store : promoteVersion(store, m.model_id, m.version, null);
          }, prev)
        );
      })
      .catch((err) => {
        if (!alive) return;
        addConsoleLog(`WARN: 모델 레지스트리 동기화 실패 — ${err?.message ?? "알 수 없는 오류"}`);
      });
    return () => {
      alive = false;
    };
  }, [addConsoleLog]);

  const dismissAlert = useCallback((id) => {
    setAlerts((prev) => prev.filter((a) => a.id !== id));
  }, []);

  const MAX_NOTIFICATIONS = 8;
  const pushNotification = useCallback((n) => {
    const now = new Date();
    const time = `${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}:${String(now.getSeconds()).padStart(2, "0")}`;
    setNotifications((prev) =>
      [{ id: Date.now() + Math.random(), time, severity: "info", ...n }, ...prev].slice(0, MAX_NOTIFICATIONS)
    );
    setUnreadCount((c) => c + 1);
  }, []);

  const markNotificationsRead = useCallback(() => setUnreadCount(0), []);

  const showAlert = useCallback(
    (alert) => {
      const id = Date.now() + Math.random();
      const entry = { id, ...alert };
      setAlerts((prev) => [...prev, entry]);
      setTimeout(() => dismissAlert(id), ALERT_AUTO_DISMISS_MS);
      // 팝업으로 띄운 경보는 벨 알림 이력에도 적재
      pushNotification({ severity: "warn", title: alert.title, message: alert.message });
    },
    [dismissAlert, pushNotification]
  );

  const resetPipeline = useCallback(() => {
    // 이전 실행의 늦은 API 응답이 새 실행/초기화 상태를 덮어쓰지 못하게 무효화한다.
    pipelineRequestRef.current += 1;
    if (driftStartTimerRef.current) {
      clearTimeout(driftStartTimerRef.current);
      driftStartTimerRef.current = null;
    }
    if (pipelineTimerRef.current) {
      clearTimeout(pipelineTimerRef.current);
      pipelineTimerRef.current = null;
    }
    setPipelineRunning(false);
    setPipelineScheduled(false);
    setPipelineStep(0);
    setPipelineRun(null); // 실행 상태 카드를 유휴(대기 중)로 복귀
    setPipelineResult(null);
  }, []);

  // trigger: 실행 사유 문자열(예: "드리프트 자동 감지 (PSI 0.384)"). 미지정 시 수동 실행으로 기록.
  // pipelineDef: RETRAIN_PIPELINES 항목. 미지정 시 기본(인구이동 예측) 파이프라인.
  const startPipeline = useCallback(
    async (trigger, pipelineDef) => {
      const pl = pipelineDef ?? RETRAIN_PIPELINES[0];
      const triggerLabel = typeof trigger === "string" ? trigger : "수동 실행";
      if (pipelineRunning) {
        addConsoleLog("WARN: 이미 실행 중인 재학습 파이프라인이 있어 새 실행을 시작하지 않았습니다.");
        return;
      }
      if (driftStartTimerRef.current && !triggerLabel.includes("드리프트 자동 감지")) {
        addConsoleLog("WARN: 드리프트 대응 재학습이 예약되어 있어 수동 실행을 시작하지 않았습니다.");
        return;
      }
      const backendTrigger = triggerLabel.includes("드리프트") ? "drift" : "manual";
      const currentServing = modelStore.find((model) => model.modelId === pl.model && model.status === "운영");
      if (currentServing?.version === pl.candidateVersion) {
        const message = `${pl.name}은 현재 운영 버전(${currentServing.version})보다 새로운 후보가 등록되지 않아 실행할 수 없습니다.`;
        addConsoleLog(`WARN: ${message}`);
        pushNotification({ severity: "warn", title: "재학습 후보 없음", message });
        return;
      }
      resetPipeline();
      const now = new Date();
      const ymd = `${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, "0")}${String(now.getDate()).padStart(2, "0")}`;
      const run = {
        runId: `RUN-${ymd}-${now.getTime().toString(36).slice(-4).toUpperCase()}`,
        pipelineId: pl.id,
        pipelineName: pl.name,
        model: pl.model,
        baseVersion: currentServing?.version ?? pl.baseVersion,
        candidateVersion: pl.candidateVersion,
        experiment: pl.experiment,
        trigger: triggerLabel,
        startedAt: `${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}:${String(now.getSeconds()).padStart(2, "0")}`
      };
      const requestId = pipelineRequestRef.current;
      setPipelineRun(run);
      setPipelineRunning(true);
      addConsoleLog(
        `INFO: 재학습 파이프라인 ${pl.id}(${pl.name}) 실행 시작. (${run.runId} · 트리거: ${triggerLabel})`
      );
      addConsoleLog(
        `INFO: 대상 모델 ${run.model} ${run.baseVersion} → 후보 ${run.candidateVersion} · 실험 ${run.experiment}`
      );

      // 백엔드 오케스트레이션 이벤트 발생 — 실제 승급/롤백 결정을 수신 (애니메이션은 UX)
      try {
        const backend = await apiSend("POST", "/api/v3/orchestration/events", {
          body: { model_id: pl.model, trigger: backendTrigger, candidate_latency_ms: 120 }
        });
        if (requestId !== pipelineRequestRef.current) return;
        setPipelineResult(backend);
        addConsoleLog(
          `INFO: 오케스트레이터 이벤트 접수 — ${backend.run_id} · 평가: ${backend.evaluation?.reason ?? backend.state}`
        );
        if (backend.state !== "succeeded") {
          const stageCount = backend.state === "debounced" ? 0 : Math.min(5, backend.stages?.length ?? 4);
          setPipelineStep(stageCount);
          setPipelineRunning(false);
          const finishedAt = `${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}:${String(now.getSeconds()).padStart(2, "0")}`;
          const resultLabel = backend.state === "rejected"
            ? "승급 반려"
            : backend.state === "debounced"
              ? "실행 조정"
              : backend.state === "rolled_back"
                ? "자동 롤백"
                : `미승급(${backend.state})`;
          setPipelineHistory((prev) => ({
            ...prev,
            [run.pipelineId]: { runId: run.runId, finishedAt, result: resultLabel }
          }));
          const message = backend.state === "rejected"
            ? "후보 모델이 승급 기준 미달로 반려되었습니다."
            : backend.state === "rolled_back"
              ? `카나리 헬스체크 실패로 자동 롤백했습니다. 직전 버전(${backend.active_version ?? run.baseVersion})을 유지합니다.`
              : "최근 재학습과 간격이 짧아 실행이 조정되었습니다.";
          addConsoleLog(`${backend.state === "rolled_back" ? "WARN" : "INFO"}: ${message}`);
          pushNotification({
            severity: backend.state === "rolled_back" ? "warn" : "info",
            title: backend.state === "rolled_back" ? "자동 롤백 수행" : "재학습 결과",
            message
          });
          return;
        }
      } catch (err) {
        if (requestId !== pipelineRequestRef.current) return;
        const reason = err?.message ?? "백엔드 응답을 받지 못했습니다.";
        addConsoleLog(`ERROR: 오케스트레이션 이벤트 실패 — ${reason}`);
        setPipelineResult({ state: "failed", reason });
        setPipelineStep(1);
        setPipelineRunning(false);
        const failedAt = new Date();
        const finishedAt = `${String(failedAt.getHours()).padStart(2, "0")}:${String(failedAt.getMinutes()).padStart(2, "0")}:${String(failedAt.getSeconds()).padStart(2, "0")}`;
        setPipelineHistory((prev) => ({
          ...prev,
          [run.pipelineId]: { runId: run.runId, finishedAt, result: "실패" }
        }));
        pushNotification({
          severity: "warn",
          title: "재학습 실행 실패",
          message: `${run.pipelineName} — ${reason}`
        });
        return;
      }
      if (requestId !== pipelineRequestRef.current) return;
      setPipelineStep(1);
    },
    [pipelineRunning, modelStore, resetPipeline, addConsoleLog, pushNotification]
  );

  useEffect(() => {
    if (!pipelineRunning) return undefined;
    if (pipelineStep === 0) return undefined;

    if (pipelineStep > PIPELINE_STEPS.length) {
      setPipelineRunning(false);
      const result = pipelineResult; // 백엔드 PipelineRun
      const state = result?.state ?? "failed";

      if (state !== "succeeded") {
        // 실패/롤백/반려/debounce — 승급 없음. 상태별 안내만 기록.
        const msg =
          state === "failed"
            ? `ERROR: 오케스트레이션 실행 실패 — ${result?.reason ?? "결과를 확인할 수 없습니다."}`
            : state === "rolled_back"
            ? `WARN: 헬스체크 실패 → 자동 롤백. 직전 버전(${result.active_version}) 유지 — ${result.deploy?.reason ?? ""}`
            : state === "rejected"
              ? "INFO: 후보 모델이 승급 기준 미달로 반려되었습니다."
              : state === "debounced"
                ? "INFO: 최근 재학습과 간격이 짧아 이벤트가 조정(debounce)되었습니다."
                : `ERROR: 알 수 없는 오케스트레이션 상태(${state}) — 승급을 중단했습니다.`;
        addConsoleLog(msg);
        pushNotification({
          severity: state === "failed" || state === "rolled_back" ? "warn" : "info",
          title:
            state === "failed"
              ? "재학습 실행 실패"
              : state === "rolled_back"
                ? "자동 롤백 수행"
                : "재학습 결과",
          message: msg.replace(/^\w+: /, "")
        });
        if (pipelineRun) {
          const now = new Date();
          const finishedAt = `${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}:${String(now.getSeconds()).padStart(2, "0")}`;
          const resultLabel =
            state === "failed"
              ? "실패"
              : state === "rolled_back"
                ? "자동 롤백"
                : state === "rejected"
                  ? "승급 반려"
                  : state === "debounced"
                    ? "실행 조정"
                    : `미승급(${state})`;
          setPipelineHistory((prev) => ({
            ...prev,
            [pipelineRun.pipelineId]: { runId: pipelineRun.runId, finishedAt, result: resultLabel }
          }));
        }
        return undefined;
      }

      // 승급 성공 — 백엔드 후보 지표를 실제 반영 (하드코딩 제거)
      const cm = result?.candidate_metrics ?? {};
      const newAcc = cm.accuracy ?? null;
      const newF1 = cm.f1 ?? null;
      if (pipelineRun?.model && (newAcc !== null || newF1 !== null)) {
        setMetricOverrides((prev) => ({
          ...prev,
          [pipelineRun.model]: {
            ...prev[pipelineRun.model],
            ...(newAcc !== null ? { accuracy: newAcc } : {}),
            ...(newF1 !== null ? { f1: newF1 } : {})
          }
        }));
      }
      if (pipelineRun?.model === "population-forecast") setDriftInjected(false);
      addConsoleLog(
        `SUCCESS: 승급·배포 완료 — ${pipelineRun?.model ?? "모델"} ${result?.active_version ?? pipelineRun?.candidateVersion} (primary ${result?.evaluation?.primary_metric ?? "-"})`
      );
      pushNotification({
        severity: "success",
        title: "모델 승급·배포 완료",
        message: `${pipelineRun?.pipelineName ?? "재학습"} — 신규 모델(Accuracy ${newAcc ?? "-"})이 SOTA로 승급되어 배포되었습니다.`
      });
      if (pipelineRun) {
        const now = new Date();
        const finishedAt = `${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}:${String(now.getSeconds()).padStart(2, "0")}`;
        setPipelineHistory((prev) => ({
          ...prev,
          [pipelineRun.pipelineId]: { runId: pipelineRun.runId, finishedAt, result: "SOTA 승급" }
        }));
        // Model Store 갱신 — 신규 버전을 운영으로 등록, 직전 운영 버전은 '이전'으로 강등
        setModelStore((prev) =>
          promoteVersion(
            prev,
            pipelineRun.model,
            result?.active_version ?? pipelineRun.candidateVersion,
            newAcc,
            formatYmd(now)
          )
        );
      }
      return undefined;
    }

    const step = PIPELINE_STEPS[pipelineStep - 1];
    // 카나리 배포 단계에는 실행 중인 파이프라인의 Docker 이미지 태그를 동봉 (2차년도 컨테이너 배포)
    const dockerSuffix =
      pipelineStep === 5 && pipelineRun
        ? ` [Docker 이미지: ${pipelineRun.model}:${pipelineRun.candidateVersion} 컨테이너 배포]`
        : "";
    addConsoleLog(step.log + dockerSuffix, false, step.warn || false);

    pipelineTimerRef.current = setTimeout(() => {
      setPipelineStep((s) => s + 1);
    }, STEP_DELAY_MS);

    return () => {
      if (pipelineTimerRef.current) clearTimeout(pipelineTimerRef.current);
    };
  }, [pipelineStep, pipelineRunning, addConsoleLog, pushNotification, pipelineRun, pipelineResult]);

  const injectDrift = useCallback(() => {
    if (pipelineRunning || driftStartTimerRef.current) return;
    setDriftInjected(true);
    setPipelineScheduled(true);
    addConsoleLog("WARN: 데이터 드리프트 심각수준 감지! (PSI 임계 0.20 초과)", false, true);
    addConsoleLog("INFO: MLOps 오케스트레이터가 재학습 자동 스케줄링을 시작합니다...");
    showAlert({
      title: "[경보] 데이터 드리프트 발생",
      message: "실시간 수집 분포 불안정 (PSI 임계 초과). MLOps 오케스트레이터 가동."
    });
    driftStartTimerRef.current = setTimeout(() => {
      driftStartTimerRef.current = null;
      setPipelineScheduled(false);
      startPipeline("드리프트 자동 감지 (PSI 임계 0.20 초과)");
    }, 1500);
  }, [pipelineRunning, addConsoleLog, showAlert, startPipeline]);

  // 지자체를 선택하고 지정 탭으로 이동(예: 현황 테이블 → 정책 시뮬레이터).
  const focusRegion = useCallback((region, tabId = "tab-simulator") => {
    setCurrentRegion(region);
    navigateToTab(tabId);
  }, [navigateToTab]);

  const value = {
    ready,
    appData,
    mockDataVisible,
    toggleMockDataVisible,
    activeTab,
    setActiveTab,
    navigateToTab,
    tabFocusRequest,
    currentRegion,
    setCurrentRegion,
    focusRegion,
    welfareWeight,
    setWelfareWeight,
    industryWeight,
    setIndustryWeight,
    housingWeight,
    setHousingWeight,
    budgetTotal,
    setBudgetTotal,
    driftInjected,
    pipelineRunning,
    pipelineScheduled,
    pipelineStep,
    pipelineRun,
    pipelineResult,
    pipelineHistory,
    modelStore,
    f1Override,
    metricOverrides,
    consoleLogs,
    addConsoleLog,
    alerts,
    showAlert,
    dismissAlert,
    notifications,
    unreadCount,
    pushNotification,
    markNotificationsRead,
    startPipeline,
    resetPipeline,
    injectDrift
  };

  return <AppStateContext.Provider value={value}>{children}</AppStateContext.Provider>;
}

export function useAppState() {
  const ctx = useContext(AppStateContext);
  if (!ctx) throw new Error("useAppState must be used within AppStateProvider");
  return ctx;
}
