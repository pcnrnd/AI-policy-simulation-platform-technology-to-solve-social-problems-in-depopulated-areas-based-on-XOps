import { createContext, useContext, useState, useCallback, useRef, useEffect } from "react";
import mockData from "../assets/mock_data.json";
import { PIPELINE_STEPS } from "../constants/pipeline.js";
import { RETRAIN_PIPELINES } from "../constants/models.js";

const AppStateContext = createContext(null);

const STEP_DELAY_MS = 2500;
const ALERT_AUTO_DISMISS_MS = 5000;

export function AppStateProvider({ children }) {
  const [ready] = useState(true);
  const [appData] = useState(mockData);

  const [activeTab, setActiveTab] = useState("tab-overview");
  const [currentRegion, setCurrentRegion] = useState(mockData.regions[0]);

  const [welfareWeight, setWelfareWeight] = useState(50);
  const [industryWeight, setIndustryWeight] = useState(50);
  const [housingWeight, setHousingWeight] = useState(50);
  const [budgetTotal, setBudgetTotal] = useState(600); // 총 예산(억) — 시뮬레이션 제약요소

  const [driftInjected, setDriftInjected] = useState(false);
  const [pipelineRunning, setPipelineRunning] = useState(false);
  const [pipelineStep, setPipelineStep] = useState(0);
  // 현재(또는 마지막) 파이프라인 실행 식별 정보 — 실행 ID·파이프라인·대상 모델·실험·트리거
  const [pipelineRun, setPipelineRun] = useState(null);
  // 파이프라인별 마지막 실행 결과 (pipelineId → { runId, finishedAt, result })
  const [pipelineHistory, setPipelineHistory] = useState({});

  const [accuracyOverride, setAccuracyOverride] = useState(null);
  const [f1Override, setF1Override] = useState(null);

  const [consoleLogs, setConsoleLogs] = useState([
    { time: "13:00:00", message: "INFO: MLOps 오케스트레이션 대기 중...", type: "" }
  ]);
  const [alerts, setAlerts] = useState([]);
  // 헤더 벨 알림 이력 — 팝업과 달리 사라지지 않고 최근 N건 보존
  const [notifications, setNotifications] = useState([]);
  const [unreadCount, setUnreadCount] = useState(0);

  const pipelineTimerRef = useRef(null);

  const addConsoleLog = useCallback((message, isError = false, isWarn = false) => {
    const now = new Date();
    const time = `${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}:${String(now.getSeconds()).padStart(2, "0")}`;
    const type = isError ? "log-err" : isWarn ? "log-warn" : "";
    setConsoleLogs((prev) => [...prev, { time, message, type }]);
  }, []);

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
    if (pipelineTimerRef.current) {
      clearTimeout(pipelineTimerRef.current);
      pipelineTimerRef.current = null;
    }
    setPipelineRunning(false);
    setPipelineStep(0);
  }, []);

  // trigger: 실행 사유 문자열(예: "드리프트 자동 감지 (PSI 0.384)"). 미지정 시 수동 실행으로 기록.
  // pipelineDef: RETRAIN_PIPELINES 항목. 미지정 시 기본(인구이동 예측) 파이프라인.
  const startPipeline = useCallback(
    (trigger, pipelineDef) => {
      resetPipeline();
      const pl = pipelineDef ?? RETRAIN_PIPELINES[0];
      const triggerLabel = typeof trigger === "string" ? trigger : "수동 실행";
      const now = new Date();
      const ymd = `${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, "0")}${String(now.getDate()).padStart(2, "0")}`;
      const run = {
        runId: `RUN-${ymd}-${now.getTime().toString(36).slice(-4).toUpperCase()}`,
        pipelineId: pl.id,
        pipelineName: pl.name,
        model: pl.model,
        baseVersion: pl.baseVersion,
        candidateVersion: pl.candidateVersion,
        experiment: pl.experiment,
        trigger: triggerLabel,
        startedAt: `${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}:${String(now.getSeconds()).padStart(2, "0")}`
      };
      setPipelineRun(run);
      setPipelineRunning(true);
      addConsoleLog(
        `INFO: 재학습 파이프라인 ${pl.id}(${pl.name}) 실행 시작. (${run.runId} · 트리거: ${triggerLabel})`
      );
      addConsoleLog(
        `INFO: 대상 모델 ${run.model} ${run.baseVersion} → 후보 ${run.candidateVersion} · 실험 ${run.experiment}`
      );
      setPipelineStep(1);
    },
    [resetPipeline, addConsoleLog]
  );

  useEffect(() => {
    if (!pipelineRunning) return undefined;
    if (pipelineStep === 0) return undefined;

    if (pipelineStep > PIPELINE_STEPS.length) {
      setPipelineRunning(false);
      addConsoleLog("SUCCESS: 모든 모델 성능 모니터링 & 복원 오케스트레이션 수행이 완료되었습니다.");
      setAccuracyOverride(0.925);
      setF1Override(0.916);
      setDriftInjected(false);
      pushNotification({
        severity: "success",
        title: "모델 승급·배포 완료",
        message: `${pipelineRun?.pipelineName ?? "재학습"} — 신규 모델(Accuracy 0.925)이 SOTA로 승급되어 배포되었습니다.`
      });
      // 파이프라인 카탈로그의 "마지막 실행" 기록 갱신
      if (pipelineRun) {
        const now = new Date();
        const finishedAt = `${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}:${String(now.getSeconds()).padStart(2, "0")}`;
        setPipelineHistory((prev) => ({
          ...prev,
          [pipelineRun.pipelineId]: { runId: pipelineRun.runId, finishedAt, result: "SOTA 승급" }
        }));
      }
      return undefined;
    }

    const step = PIPELINE_STEPS[pipelineStep - 1];
    addConsoleLog(step.log, false, step.warn || false);

    pipelineTimerRef.current = setTimeout(() => {
      setPipelineStep((s) => s + 1);
    }, STEP_DELAY_MS);

    return () => {
      if (pipelineTimerRef.current) clearTimeout(pipelineTimerRef.current);
    };
  }, [pipelineStep, pipelineRunning, addConsoleLog, pushNotification, pipelineRun]);

  const injectNormal = useCallback(() => {
    if (pipelineRunning) {
      alert("파이프라인이 이미 동작 중입니다. 완료 후 초기화해 주세요!");
      return;
    }
    setDriftInjected(false);
    setAccuracyOverride(null);
    setF1Override(null);
    addConsoleLog("INFO: 정상 시나리오 데이터가 실시간 API에 입력되었습니다.");
  }, [pipelineRunning, addConsoleLog]);

  const injectDrift = useCallback(() => {
    if (pipelineRunning) return;
    setDriftInjected(true);
    addConsoleLog("WARN: 데이터 드리프트 심각수준 감지! (PSI: 0.384 > 0.20)", false, true);
    addConsoleLog("INFO: MLOps 오케스트레이터가 재학습 자동 스케줄링을 시작합니다...");
    showAlert({
      title: "[경보] 데이터 드리프트 발생",
      message: "실시간 수집 분포 불안정 (PSI: 0.384). MLOps 오케스트레이터 가동."
    });
    setTimeout(() => startPipeline("드리프트 자동 감지 (PSI 0.384 > 0.20)"), 1500);
  }, [pipelineRunning, addConsoleLog, showAlert, startPipeline]);

  // 지자체를 선택하고 지정 탭으로 이동(예: 현황 테이블 → 정책 시뮬레이터).
  const focusRegion = useCallback((region, tabId = "tab-simulator") => {
    setCurrentRegion(region);
    setActiveTab(tabId);
  }, []);

  const value = {
    ready,
    appData,
    activeTab,
    setActiveTab,
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
    pipelineStep,
    pipelineRun,
    pipelineHistory,
    accuracyOverride,
    f1Override,
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
    injectNormal,
    injectDrift
  };

  return <AppStateContext.Provider value={value}>{children}</AppStateContext.Provider>;
}

export function useAppState() {
  const ctx = useContext(AppStateContext);
  if (!ctx) throw new Error("useAppState must be used within AppStateProvider");
  return ctx;
}
