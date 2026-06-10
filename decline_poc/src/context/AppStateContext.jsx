import { createContext, useContext, useState, useCallback, useRef, useEffect } from "react";
import mockData from "../assets/mock_data.json";
import { PIPELINE_STEPS } from "../constants/pipeline.js";

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

  const [driftInjected, setDriftInjected] = useState(false);
  const [pipelineRunning, setPipelineRunning] = useState(false);
  const [pipelineStep, setPipelineStep] = useState(0);

  const [accuracyOverride, setAccuracyOverride] = useState(null);
  const [f1Override, setF1Override] = useState(null);

  const [consoleLogs, setConsoleLogs] = useState([
    { time: "13:00:00", message: "INFO: MLOps 오케스트레이션 대기 중...", type: "" }
  ]);
  const [alerts, setAlerts] = useState([]);

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

  const showAlert = useCallback(
    (alert) => {
      const id = Date.now() + Math.random();
      const entry = { id, ...alert };
      setAlerts((prev) => [...prev, entry]);
      setTimeout(() => dismissAlert(id), ALERT_AUTO_DISMISS_MS);
    },
    [dismissAlert]
  );

  const resetPipeline = useCallback(() => {
    if (pipelineTimerRef.current) {
      clearTimeout(pipelineTimerRef.current);
      pipelineTimerRef.current = null;
    }
    setPipelineRunning(false);
    setPipelineStep(0);
  }, []);

  const startPipeline = useCallback(() => {
    resetPipeline();
    setPipelineRunning(true);
    addConsoleLog("INFO: 이벤트 기반 MLOps 오케스트레이션 자동 파이프라인 실행 시작.");
    setPipelineStep(1);
  }, [resetPipeline, addConsoleLog]);

  useEffect(() => {
    if (!pipelineRunning) return undefined;
    if (pipelineStep === 0) return undefined;

    if (pipelineStep > PIPELINE_STEPS.length) {
      setPipelineRunning(false);
      addConsoleLog("SUCCESS: 모든 모델 성능 모니터링 & 복원 오케스트레이션 수행이 완료되었습니다.");
      setAccuracyOverride(0.925);
      setF1Override(0.916);
      setDriftInjected(false);
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
  }, [pipelineStep, pipelineRunning, addConsoleLog]);

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
    setTimeout(() => startPipeline(), 1500);
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
    driftInjected,
    pipelineRunning,
    pipelineStep,
    accuracyOverride,
    f1Override,
    consoleLogs,
    addConsoleLog,
    alerts,
    showAlert,
    dismissAlert,
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
