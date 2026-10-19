import { useEffect, useMemo, useRef, useState } from "react";
import { Line, Scatter } from "react-chartjs-2";
import Card from "../components/Card.jsx";
import PolicyRecommendation from "../components/PolicyRecommendation.jsx";
import PipelineStepper from "../components/PipelineStepper.jsx";
import CollapsibleStage from "../components/CollapsibleStage.jsx";
import FactorAnalysisStage from "../components/FactorAnalysisStage.jsx";
import FactorResultStage from "../components/FactorResultStage.jsx";
import PendingData from "../components/PendingData.jsx";
import ScenarioCompare from "../components/ScenarioCompare.jsx";
import ModelEvidence from "../components/ModelEvidence.jsx";
import SimulatorMap, { SimulatorMapDetails, riskGrade } from "../components/SimulatorMap.jsx";
import SimulatorDock from "../components/SimulatorDock.jsx";
import RealForecastResult from "../components/RealForecastResult.jsx";
import { useAppState } from "../context/AppStateContext.jsx";
import { useChartTheme } from "../hooks/useChartTheme.js";
import { useRealSimulation } from "../hooks/useRealSimulation.js";
import { useFactorAnalysis } from "../hooks/useFactorAnalysis.js";
import { getScrollBehavior } from "../lib/motion.js";
import { rankStrategies } from "../constants/policyStrategies.js";
import { YEAR_LABELS, computeTrends, budgetToFactor, controlBoostOf, ANALYSIS_STEPS } from "../lib/simulation.js";
import {
  REAL_TARGETS, REAL_GUIDE_TEXT, REAL_CONTROLS_LOCK_TEXT, FACTOR_GUIDE_TEXT, FACTOR_PENDING_TEXT, FACTOR_RUN_LOCK_TEXT, FACTOR_IDLE_TEXT,
  FACTOR_DIAGNOSIS_TEXT, DEMO_FACTOR_TARGET, realUnavailableText, selectedFactorModel, inferenceMatches, inferenceContextKey, factorAnalysisView,
  observedFeatureLabel, rankSensitivity, targetUnit, formatUnitValue, formatMonth,
  realSliderControls, forecastSeries, sensitivityRecommendation, regionRealStatus
} from "../lib/realSimulation.js";

// 실데이터 표시에서 관측 입력을 바꾼 뒤 시뮬레이션 추론(읽기 전용 계산)을 다시 부르기까지의 대기.
const SIM_DEBOUNCE_MS = 500;
const STAGE_IDS = ["stage-factor", "stage-result", "stage-sim", "stage-report"];

// 정책 시뮬레이터 — 기준선 4단계 구조 하나를 두 모드가 함께 쓴다(모드는 데이터 출처만 바꾼다).
// ①·②: 두 모드 모두 선택한 기존 활성 요인분석 모델의 결과(useFactorAnalysis).
// ③·④: 데모는 지역 사례(mock_data.json)와 화면 예시 계산, 실데이터는 같은 지역·학습 대상의 활성 시뮬레이션 버전
// 추론(useRealSimulation) — 실데이터 ③·④는 ① 실행과 무관하다. 실데이터 대응이 없는 요소는 같은 자리에 행동 안내를 둔다.
export default function SimulatorPage() {
  const {
    appData, currentRegion, setCurrentRegion,
    welfareWeight, setWelfareWeight, industryWeight, setIndustryWeight, housingWeight, setHousingWeight,
    budgetTotal, setBudgetTotal, addConsoleLog, navigateToTab, mockDataVisible
  } = useAppState();
  const ct = useChartTheme();
  const demo = mockDataVisible;
  const [searchText, setSearchText] = useState("");

  // 실데이터 분석 조건 — 두 모드에 같은 select 를 두고 데모는 선택지 1개.
  const [target, setTarget] = useState("observed_sales_krw");
  // 행정동은 고른 지역과 함께 기억한다 — 지역이 바뀌면 "전체"(첫 옵션)로 돌아가 이전 지역 코드를 요청에 싣지 않는다.
  const [dongPick, setDongPick] = useState({ regionId: currentRegion.id, code: "" });
  const dong = dongPick.regionId === currentRegion.id ? dongPick.code : "";
  const setDong = (code) => setDongPick({ regionId: currentRegion.id, code });
  const real = useRealSimulation({ enabled: !demo, region: currentRegion.id, target, dong, log: (message) => addConsoleLog(message, false, true) });
  // STAGE ① 요인분석 — 두 모드 같은 흐름. 데모는 합성 요인 패널(생활인구), 실데이터는 선택한 학습 대상의 요인분석 모델.
  const factorTarget = demo ? DEMO_FACTOR_TARGET : target;
  const mode = demo ? "mock" : "real";
  const factorContext = `${mode}|${currentRegion.id}|${factorTarget}`;
  const [modelPick, setModelPick] = useState(null);
  const picked = modelPick?.context === factorContext ? modelPick : null;
  const factor = useFactorAnalysis({ demo, region: currentRegion.id, target: factorTarget, dong: demo ? "" : dong,
    modelId: picked?.id, versionId: picked?.version,
    log: (message) => addConsoleLog(message, false, message.startsWith("WARN")) });
  const factorModels = factor.catalog?.mode === mode && factor.catalog?.source === mode
    ? factor.catalog.instances.filter((row) => appData.regions.some((r) => r.id === row.region_id)
      && (demo ? row.target === DEMO_FACTOR_TARGET : Object.hasOwn(REAL_TARGETS, row.target))
      && selectedFactorModel(factor.catalog, row.id, row.region_id, row.target, mode)) : [];
  const factorModel = selectedFactorModel(factor.catalog, picked?.id, currentRegion.id, factorTarget, mode);
  useEffect(() => {
    if (factor.loading || factor.loadFailed) return;
    const chosen = factorModels.find((row) => row.id === picked?.id);
    if (picked?.explicit && !chosen) return;
    const model = chosen ?? factorModels.find((row) => row.region_id === currentRegion.id && row.target === factorTarget);
    const next = model ? { context: factorContext, id: model.id, version: model.active_version, explicit: picked?.explicit ?? false } : null;
    setModelPick((previous) => previous?.context === next?.context && previous?.id === next?.id
      && previous?.version === next?.version ? previous : next);
  }, [factor.catalog, factor.loading, factor.loadFailed, factorContext]);
  // 실데이터 모델이 준비되지 않아 잠근 컨트롤의 사유 — 조회 중·조회 실패·모델 없음을 구분한다.
  const realLockTitle = demo || real.ready ? null
    : real.loading ? "모델과 관측 입력을 확인하는 중입니다." : real.loadFailed ? "모델 정보를 불러오지 못했습니다. [정책 추천 도출]을 눌러 다시 확인하세요." : "시뮬레이션 활성 모델을 먼저 준비하세요.";
  const evidenceState = real.loading ? "loading" : real.loadFailed ? "error" : undefined;
  const unit = targetUnit(target);
  const targetName = REAL_TARGETS[target];
  const dongOptions = real.context?.dong_options ?? [];
  const label = (feature) => observedFeatureLabel(feature, target, dongOptions);
  const regions = appData.regions.map(({ id, name }) => ({ id, name }));
  const simProof = real.context?.provenance;
  const dongName = dongOptions.find((row) => row.code === simProof?.dong_code)?.name ?? simProof?.dong_code ?? "—";
  const realSourceLabel = real.ready ? `실데이터 · ${real.datasetName(simProof.dataset_id)}` : undefined;

  // 조절변수(x) — 데모는 지역 사례의 시설 변수, 실데이터는 시뮬레이션 모델의 조정 가능한 관측 입력. 바뀌면 기본값으로.
  const controls = demo ? currentRegion.case.simulation.controls ?? [] : realSliderControls(real.context, label, unit);
  const controlsKey = [mode, currentRegion.id, demo ? factorModel?.active_version : simProof?.version_id,
    controls.map((c) => `${c.key}:${c.default}`).join(",")].join("|");
  const defaultValues = () => Object.fromEntries(controls.map((c) => [c.key, c.default]));
  const [controlValues, setControlValues] = useState(defaultValues);
  useEffect(() => {
    setControlValues(defaultValues());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [controlsKey]);

  // 총 예산(제약요소) → 체감수익 곡선. 시설 조절변수 → 정규화×weight 합산 부스트(데모 계산).
  const budgetFactor = budgetToFactor(budgetTotal);
  const controlBoost = controlBoostOf(controls, controlValues);

  // 정책 추천(실행 버튼으로 도출) 상태 — 데모는 RICE 랭킹 배열, 실데이터는 민감도 패널 view.
  const [recommendation, setRecommendation] = useState(null);
  const [recoSig, setRecoSig] = useState("");
  const [recoSim, setRecoSim] = useState(null);
  const controlSig = controls.map((c) => `${c.key}:${controlValues[c.key] ?? c.default}`).join(",");
  const contextKey = inferenceContextKey({ mode, region: currentRegion.id, target: factorTarget, dong: demo ? "" : dong,
    factorModel, simulationProof: demo ? null : simProof });
  const currentSig = `${contextKey}|${welfareWeight}|${industryWeight}|${housingWeight}|${budgetTotal}|${controlSig}`;
  const recoStale = recommendation !== null && recoSig !== currentSig;

  // 단계 접기/펼치기 상태 (기본 전부 펼침)
  const [openStages, setOpenStages] = useState(Object.fromEntries(STAGE_IDS.map((id) => [id, true])));
  const revealStage = (id, delay = 60) => {
    setOpenStages((s) => ({ ...s, [id]: true }));
    setTimeout(() => {
      document.getElementById(id)?.scrollIntoView({ behavior: getScrollBehavior(), block: "start" });
    }, delay);
  };

  // STAGE① 요인분석 실행 상태 — idle(미실행) → running(단계 진행) → done(결과 공개). 두 모드 같은 훅(useFactorAnalysis).
  const factorResult = inferenceMatches(factor.result?.analysis, { mode, region: currentRegion.id, target: factorTarget,
    modelId: factorModel?.id, versionId: factorModel?.active_version, dong: demo ? "" : dong }) ? factor.result : null;
  const analysisStatus = factor.status === "done" && !factorResult ? "idle" : factor.status;
  const [simResult, setSimResult] = useState(null); // 실데이터 시뮬레이션 추론 결과
  const [simSig, setSimSig] = useState("");
  // 실데이터는 같은 지역·학습 대상의 요인분석 활성 모델이 있어야 실행된다(데모는 자동 준비). 조회 중·실패·모델 없음을 구분한다.
  // 조회 실패면 잠그지 않고 [요인분석 실행]이 목록·입력을 다시 읽는다(재시도).
  const factorRetry = factor.loadFailed && !factorModel;
  const factorLockTitle = factorModel || factorRetry ? null
    : factor.loading ? "모델 목록을 확인하는 중입니다." : FACTOR_RUN_LOCK_TEXT;
  const factorMissing = !factor.loading && !factor.loadFailed && (!factorModel || factor.missing);

  // 지역·학습 대상·행정동 변경 시 결과 초기화 (지역별 요인분석은 별도 실행 — 요인분석 상태는 훅이 비운다)
  useEffect(() => {
    // 다른 지역에 갔다가 돌아와도 이전 행정동이 되살아나지 않게 기억을 지운다(화면 값은 이미 "전체").
    setDongPick((pick) => (pick.regionId === currentRegion.id ? pick : { regionId: currentRegion.id, code: "" }));
    setSimResult(null);
    setRecommendation(null);
    setRecoSig("");
    setRecoSim(null);
    // real.revision: 활성 버전이 바뀌거나(409) 다시 확인해 입력을 다시 읽으면 이전 결과를 비운다.
  }, [demo, currentRegion.id, target, dong, real.revision]);

  const handleRunAnalysis = async () => {
    if (factorRetry) { factor.reload(); return; }
    // 실행할 수 없으면(버튼 aria-disabled + 사유 title) 클릭을 무시한다.
    if (analysisStatus === "running" || factorLockTitle) return;
    setOpenStages((s) => ({ ...s, "stage-factor": true }));
    if (demo) {
      // 데모 ③·④는 ① 결과 뒤에 열린다(기준선 흐름) — 다시 실행하면 이전 추천을 비운다.
      setRecommendation(null);
      setRecoSig("");
      setRecoSim(null);
    }
    addConsoleLog(`INFO: ${currentRegion.name} 요인분석 실행`);
    const analysis = await factor.run();
    if (!analysis) return;
    addConsoleLog(`SUCCESS: ${currentRegion.name} 요인분석 완료 — 요인 ${analysis.contributions.length}개 기여도 표시`);
    setOpenStages((s) => ({ ...s, "stage-result": true, "stage-sim": true }));
    // 분석 결과가 갱신된 STAGE ②로 이동해 완료가 즉시 보이게 한다
    revealStage("stage-result", 120);
  };

  // 실데이터 시뮬레이션 추론 — 현재 조절변수 값으로. 결과는 그때의 서명과 함께 저장해 낡음을 판정한다.
  const runSimulation = async () => {
    const sig = currentSig;
    const changes = Object.fromEntries(controls.map((c) => [c.key, Number(controlValues[c.key] ?? c.default)]));
    const result = await real.simulate(changes);
    if (result) {
      setSimResult(result);
      setSimSig(sig);
    }
    return result;
  };
  const analysisDone = analysisStatus === "done";
  const analysisRunning = analysisStatus === "running";
  // 시뮬레이션 모델이 준비되면, 그리고 관측 입력을 바꾸면 잠시 뒤 다음 달 예측을 계산한다
  // (데모의 실시간 추이 갱신과 같은 자리, STAGE ① 실행과 무관).
  const simTimerRef = useRef(null);
  useEffect(() => {
    if (demo || !real.ready) return undefined;
    simTimerRef.current = setTimeout(runSimulation, SIM_DEBOUNCE_MS);
    return () => clearTimeout(simTimerRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [demo, real.ready, contextKey, controlSig]);

  const handleRecommend = async () => {
    // 조회 실패면 [정책 추천 도출]이 목록·입력을 다시 읽는다(재시도).
    if (!demo && !real.ready) {
      if (real.loadFailed) real.reload();
      return;
    }
    if (demo) {
      const ranked = rankStrategies(
        currentRegion,
        { welfare: welfareWeight, industry: industryWeight, housing: housingWeight },
        budgetFactor
      );
      setRecommendation(ranked);
      setRecoSig(currentSig);
      addConsoleLog(`INFO: ${currentRegion.name} 정책 추천 도출 완료 — 1순위 '${ranked[0].name}' (점수 ${formatUnitValue(ranked[0].score, "점")}).`);
    } else {
      // 대기 중인 자동 계산이 이 요청을 밀어내지 않게 먼저 끈다.
      clearTimeout(simTimerRef.current);
      const sim = simResult && simSig === currentSig ? simResult : await runSimulation();
      if (!sim) return;
      const ranking = rankSensitivity(sim);
      setRecommendation(sensitivityRecommendation(ranking, label, unit, targetName));
      setRecoSim(sim);
      setRecoSig(currentSig);
      addConsoleLog(ranking.length
        ? `INFO: ${currentRegion.name} 정책 추천 도출 완료 — 1순위 '${label(ranking[0].feature)} ${ranking[0].to > ranking[0].from ? "증가" : "감소"}' (모델 예측상 변화 +${formatUnitValue(ranking[0].delta, unit)}${unit}).`
        : `WARN: ${currentRegion.name} 정책 추천 도출 — 관측 범위 안에서 증가 예측을 만든 조정 항목이 없습니다.`);
    }
    // 결과는 STAGE ④에 렌더되므로 — 펼친 뒤 화면으로 이동해 즉시 보이게 한다
    revealStage("stage-report");
  };

  const toggleStage = (id) => setOpenStages((s) => ({ ...s, [id]: !s[id] }));
  const allOpen = Object.values(openStages).every(Boolean);
  const setAllStages = (open) => setOpenStages(Object.fromEntries(STAGE_IDS.map((id) => [id, open])));

  // 스크롤 스파이 — 화면 상단 밴드에 걸린 단계를 스텝퍼에 하이라이트
  const [activeStageId, setActiveStageId] = useState("stage-factor");
  useEffect(() => {
    const visibility = new Map();
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => visibility.set(e.target.id, e.isIntersecting));
        const active = STAGE_IDS.find((id) => visibility.get(id));
        if (active) setActiveStageId(active);
      },
      { rootMargin: "-10% 0px -60% 0px" }
    );
    STAGE_IDS.forEach((id) => {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    });
    return () => observer.disconnect();
  }, []);

  const { baseTrend, simTrend } = useMemo(
    () => computeTrends(currentRegion, welfareWeight, industryWeight, housingWeight, budgetFactor, controlBoost),
    [currentRegion, welfareWeight, industryWeight, housingWeight, budgetFactor, controlBoost]
  );
  const currentSimulation = !demo && inferenceMatches(simResult, { mode: "real", region: currentRegion.id, target,
    modelId: simProof?.instance_id, versionId: simProof?.version_id, dong }) ? simResult : null;
  const series = demo ? null : forecastSeries(real.context, currentSimulation);
  const chartData = useMemo(
    () => ({
      labels: demo ? YEAR_LABELS : series.labels,
      datasets: [
        {
          label: demo ? "자연 감소 예측 추이 (Base Model)" : "기준 입력 예측",
          data: demo ? baseTrend : series.baseline,
          borderColor: "rgba(239, 68, 68, 0.65)",
          borderDash: [5, 5],
          fill: false,
          tension: 0.2
        },
        {
          label: demo ? "정책 시뮬레이션 적용 예측 추이" : "조정 입력 예측",
          data: demo ? simTrend : series.adjusted,
          borderColor: "rgba(59, 130, 246, 1)",
          backgroundColor: "rgba(59, 130, 246, 0.05)",
          fill: true,
          tension: 0.2
        }
      ]
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [demo, baseTrend, simTrend, simResult, real.context]
  );

  const chartOpts = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: { grid: { color: ct.grid }, ticks: { color: ct.tick } },
      x: { grid: { display: false }, ticks: { color: ct.tick } }
    },
    plugins: { legend: { labels: { color: ct.legend, boxWidth: 12 } } }
  };

  const finalPop = simTrend[9] ?? currentRegion.population;
  const baseFinalPop = baseTrend[9] ?? currentRegion.population;
  const policyDifference = finalPop - baseFinalPop;
  const growthPercent = (((finalPop - currentRegion.population) / currentRegion.population) * 100).toFixed(1);
  const growthClass = parseFloat(growthPercent) >= 0 ? "trend-up" : "trend-down";

  const scatterData = useMemo(
    () => ({
      datasets: [
        {
          // 전국 인구감소 지역 분포 속에서 분석 대상의 상대 위치를 보여주는 배경 점
          label: "전국 비교 지자체",
          data: (appData.benchmark_regions ?? []).map((r) => ({ x: r.riskIndex, y: r.population, name: r.name })),
          backgroundColor: "rgba(148, 163, 184, 0.55)",
          pointRadius: 5,
          pointHoverRadius: 8
        },
        {
          label: "분석 대상 지자체",
          data: appData.regions.map((r) => ({ x: r.riskIndex, y: r.population, name: r.name })),
          backgroundColor: appData.regions.map((r) =>
            r.id === currentRegion.id ? "rgba(59, 130, 246, 1)" : "rgba(245, 158, 11, 0.7)"
          ),
          pointRadius: appData.regions.map((r) => (r.id === currentRegion.id ? 10 : 6)),
          pointHoverRadius: 12
        }
      ]
    }),
    [appData, currentRegion]
  );

  const scatterOpts = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        title: { display: true, text: "인구소멸 위험지수 (낮을수록 위험)", color: ct.tick },
        grid: { color: ct.grid },
        ticks: { color: ct.tick }
      },
      y: {
        title: { display: true, text: "인구수(명)", color: ct.tick },
        grid: { color: ct.grid },
        ticks: { color: ct.tick }
      }
    },
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: (cx) => `${cx.raw.name}: 위험 ${cx.raw.x}, 인구 ${cx.raw.y.toLocaleString()}명`
        }
      }
    }
  };

  const filteredRegions = appData.regions.filter((r) =>
    r.name.toLowerCase().includes(searchText.trim().toLowerCase())
  );
  const distributionRegions = [...(appData.benchmark_regions ?? []), ...appData.regions];
  const riskRank = distributionRegions.filter((region) => region.riskIndex < currentRegion.riskIndex).length + 1;
  const populationRank = distributionRegions.filter((region) => region.population > currentRegion.population).length + 1;

  const sim = currentRegion.case.simulation;
  // 결과 표시 조건 — 실행 상태(조작)와 데이터 유무를 분리한다. 실행 버튼·단계 표시는 그대로 돈다.
  // 데모는 ① 요인분석 완료 후, 실데이터 ③·④는 ① 실행과 무관하게 시뮬레이션 결과로 판단한다.
  const resultsReady = demo && analysisDone;
  const recoReady = (!demo || analysisDone) && recommendation && !recoStale;
  // 관측 입력을 바꾼 뒤 재계산이 실패하면 이전 입력의 예측을 현재 값처럼 두지 않는다(계산 대기 중에는 그대로 둔다).
  const simShown = currentSimulation && (simSig === currentSig || !real.predictFailed) ? currentSimulation : null;

  // 스텝퍼 완료 표시: ①② 요인분석 완료, ③④ 정책 추천 도출 완료 기준
  const doneStages = [
    ...(analysisDone ? ["stage-factor", "stage-result"] : []),
    ...(recoReady ? ["stage-sim", "stage-report"] : [])
  ];

  // ── 실데이터 표시의 같은 자리 내용 ──
  const guide = (text) => (
    <div className="real-flow-guide" data-guide>
      <p>{text}</p>
    </div>
  );
  // ③ 자리의 시뮬레이션 모델 대기 문구 — 모델 없음(전문)·조회 중·조회 실패를 구분한다.
  const realPending = <PendingData running={real.loading} text={real.missing ? REAL_GUIDE_TEXT : realLockTitle} />;
  // ①·② 요인분석 — 두 모드 같은 매핑(factorAnalysisView). 실데이터 모델 없음이면 안내(전문은 ① 첫 카드 한 곳, 나머지는 한 줄).
  const factorLabel = (feature) => observedFeatureLabel(feature, factorTarget, factorResult?.analysis.dong_options ?? dongOptions);
  const factorView = factorResult ? factorAnalysisView(factorResult.analysis, { demo, dataset: factorResult.dataset, label: factorLabel }) : null;
  const factorLine = <PendingData text={FACTOR_PENDING_TEXT} />;
  const factorIdle = <PendingData running={analysisRunning} text={FACTOR_IDLE_TEXT} />;
  const factorPlaceholders = factorMissing ? [guide(FACTOR_GUIDE_TEXT), factorLine, factorLine] : [factorIdle, factorIdle, factorIdle];
  const factorRunInfo = factorView ? factorView.runInfo
    : demo ? "인구 구조·시설 지표 9개 요인 · 학습 대상 생활인구" : `등록 학습데이터 기준 · 학습 대상 ${targetName}`;
  const factorSourceLabel = factorView?.sourceLabel ?? (demo ? "데모" : undefined);
  const rangeProof = simProof?.range_provenance;
  const frame = demo
    ? { objective: sim.objective, demand: sim.factors.demand, supply: sim.factors.supply, adjust: sim.factors.adjust, constraint: `${sim.constraint} · ${budgetTotal.toLocaleString()}억` }
    : real.ready
      ? {
          objective: `다음 달 ${targetName} 예측 (예측월 ${formatMonth(simProof.prediction_period)})`,
          demand: `${targetName} 관측 · ${dongName}`,
          supply: simResult ? `기준 예측 ${formatUnitValue(simResult.baseline_prediction, unit)} ${unit}` : "–",
          adjust: `관측 입력 ${controls.length}개 (${controls.map((c) => c.label).join("·")})`,
          constraint: `학습 구간 관측 범위 안 조정 · ${formatMonth(rangeProof?.from)}~${formatMonth(rangeProof?.to)}`
        }
      : { objective: "–", demand: "–", supply: "–", adjust: "–", constraint: "–" };
  const cardSource = demo ? "mock" : "api";

  const conditions = (
    <div className="real-flow-controls" data-slot="sim.factor.conditions">
      <label>
        학습 대상
        <select aria-label="학습 대상" value={demo ? "demo" : target} disabled={analysisRunning} onChange={(e) => { if (!demo) { setDong(""); setTarget(e.target.value); } }}>
          {demo ? <option value="demo">인구 (데모 사례)</option> : Object.entries(REAL_TARGETS).map(([id, name]) => <option key={id} value={id}>{name}</option>)}
        </select>
      </label>
      <label>
        요인분석 모델
        <select
          aria-label="요인분석 모델"
          value={factorModel?.id ?? ""}
          disabled={analysisRunning || factor.loading || !factorModels.length}
          title={factor.loadFailed ? "모델 목록 조회 실패: 요인분석 실행을 눌러 다시 확인하세요." : factorLockTitle ?? undefined}
          onChange={(event) => {
            const model = factorModels.find((row) => row.id === event.target.value);
            const region = appData.regions.find((row) => row.id === model?.region_id);
            if (!model || !region) return;
            setModelPick({ context: [mode, model.region_id, model.target].join("|"), id: model.id, version: model.active_version, explicit: true });
            setDongPick({ regionId: region.id, code: "" });
            if (!demo) setTarget(model.target);
            setCurrentRegion(region);
          }}
        >
          {!factorModel && <option value="">{factor.loading ? "모델 확인 중…" : factor.loadFailed ? "모델 목록 조회 실패" : "이 지역·대상의 활성 모델 없음"}</option>}
          {factorModels.map((row) => <option key={row.id} value={row.id}>{row.name} · {appData.regions.find((r) => r.id === row.region_id)?.name} · {demo ? "생활인구" : REAL_TARGETS[row.target]}</option>)}
        </select>
      </label>
      {!demo && real.loading && <p role="status">모델과 관측 입력 확인 중…</p>}
      {!demo && real.feedback && <p className="async-feedback is-error" role="alert">{real.feedback}</p>}
      {factor.feedback && <p className="async-feedback is-error" role="alert">{factor.feedback}</p>}
      {factor.loadFailed && <p role="alert">모델 목록을 불러오지 못했습니다. 요인분석 실행을 눌러 다시 확인하세요.</p>}
    </div>
  );

  const regionInfo = (r) => {
    if (!demo) return { chip: null, meta: regionRealStatus(real.catalog, r.id, target) };
    const g = riskGrade(r.riskIndex);
    return {
      chip: { label: g.label, style: { color: g.color, backgroundColor: `rgba(${g.rgb}, 0.12)`, borderColor: `rgba(${g.rgb}, 0.3)` } },
      meta: `${r.theme} · ${r.population.toLocaleString()}명 · 위험 ${r.riskIndex}`
    };
  };

  return (
    <>
      <div className="pl-toolbar" data-slot="sim.toolbar">
        <PipelineStepper activeId={activeStageId} doneIds={doneStages} onJump={revealStage} />
        <button type="button" className="pl-collapse-all" onClick={() => setAllStages(!allOpen)}>
          <i className={`fa-solid ${allOpen ? "fa-compress" : "fa-expand"}`} aria-hidden="true"></i>
          {allOpen ? "모두 접기" : "모두 펼치기"}
        </button>
      </div>

      {/* ── STAGE ① 사회문제 요인분석 ── */}
      <FactorAnalysisStage
        open={openStages["stage-factor"]}
        onToggle={() => toggleStage("stage-factor")}
        sub={`${currentRegion.name} · ${demo ? "데모 요인 패널" : "실데이터"}`}
        runInfo={factorRunInfo}
        status={analysisStatus}
        stepIndex={factor.step}
        steps={ANALYSIS_STEPS}
        onRun={handleRunAnalysis}
        view={factorView}
        placeholders={factorPlaceholders}
        sourceLabel={factorSourceLabel}
        conditions={conditions}
        runLockTitle={factorLockTitle}
        runTitle={factorRetry ? "다시 확인" : null}
        evidence={<ModelEvidence slot="sim.factor.evidence" result={factorResult?.analysis} regions={regions} label={factorLabel}
          datasetName={() => factorResult?.dataset?.name ?? factorResult?.analysis.provenance.dataset_id}
          emptyText={factorMissing ? FACTOR_PENDING_TEXT : "[요인분석 실행] 후 모델·학습데이터 근거가 표시됩니다."} />}
      />

      {/* ── STAGE ② 요인분석 결과 ── */}
      <FactorResultStage
        open={openStages["stage-result"]}
        onToggle={() => toggleStage("stage-result")}
        locked={analysisStatus !== "done"}
        running={analysisRunning}
        split={factorView?.split}
        sourceLabel={factorSourceLabel}
        diagnosis={demo ? currentRegion.case.problemDiagnosis : null}
        diagnosisNote={demo ? null : FACTOR_DIAGNOSIS_TEXT}
        pendingText={factorMissing ? FACTOR_PENDING_TEXT : null}
        showDerive={demo}
      />

      {/* ── STAGE ③ 시뮬레이션: 공간정보 탐색 + 강화학습 변수 + 예측 ── */}
      <CollapsibleStage
        id="stage-sim"
        no="STAGE ③"
        title="시뮬레이션"
        sub="복잡계 기반 강화학습 — 조절변수(x)·상관변수(y)·제약요소(예산)"
        open={openStages["stage-sim"]}
        onToggle={() => toggleStage("stage-sim")}
      >
        {/* 강화학습 변수 프레임 */}
        <div className="pl-sim-frame" data-slot="sim.sim.frame">
          <div className="pl-sim-cell pl-sim-obj">
            <span className="pl-sim-tag">목적함수</span>
            <span className="mock-data-output">{frame.objective}</span>
            {analysisDone && (
              <span className="pl-param-chip">
                <i className="fa-solid fa-circle-check" aria-hidden="true"></i> STAGE ② 도출 파라미터 적용
              </span>
            )}
          </div>
          <div className="pl-sim-vars">
            <div className="pl-sim-cell">
              <span className="pl-sim-tag">수요 (x)</span>
              <span className="mock-data-output">{frame.demand}</span>
            </div>
            <div className="pl-sim-cell">
              <span className="pl-sim-tag">공급 (y)</span>
              <span className="mock-data-output">{frame.supply}</span>
            </div>
            <div className="pl-sim-cell">
              <span className="pl-sim-tag">조절변수</span>
              <span className="mock-data-output">{frame.adjust}</span>
            </div>
            <div className="pl-sim-cell pl-sim-constraint">
              <span className="pl-sim-tag">제약요소</span>
              <span className="mock-data-output">{frame.constraint}</span>
            </div>
          </div>
        </div>

        {/* 공간정보 탐색: 전체 지도 + 플로팅 패널 + 우측 지자체 독 */}
        <div className="sim-map-zone">
          <SimulatorMap
            appData={appData}
            currentRegion={currentRegion}
            setCurrentRegion={setCurrentRegion}
            allowSeed={demo}
            searchText={searchText}
            onSearch={setSearchText}
            overlayNote={demo ? null : `${realUnavailableText("위험등급·인구밀도·시설물 지도는")} 지역은 오른쪽 목록에서 선택하세요.`}
            resizeKey={`${openStages["stage-sim"]}|${analysisStatus}`}
          />
          <SimulatorDock
            regions={filteredRegions}
            currentRegion={currentRegion}
            onSelectRegion={setCurrentRegion}
            regionInfo={regionInfo}
            searchText={searchText}
            onClearSearch={() => setSearchText("")}
            districtControl={<label className="sim-dock-hint">행정동
              <select aria-label="행정동" value={demo ? "all" : dong || simProof?.dong_code || ""}
                disabled={analysisRunning || (!demo && !real.context)}
                title={!demo && !real.context ? realLockTitle ?? undefined : undefined}
                onChange={(event) => { if (!demo && real.context) setDong(event.target.value); }}>
                {demo ? <option value="all">전체</option> : !real.context ? <option value="">모델 준비 후 선택</option>
                  : dongOptions.map((row) => <option key={row.code} value={row.code}>{row.name}</option>)}
              </select>
            </label>}
            hint={demo ? "예산·정책 강조·시설 변수를 조정하면 10년 인구 예측이 실시간 갱신됩니다." : "관측 입력을 조정하면 다음 달 예측이 갱신됩니다."}
            fixed={{
              budgetTotal, setBudgetTotal, welfare: welfareWeight, setWelfare: setWelfareWeight,
              industry: industryWeight, setIndustry: setIndustryWeight, housing: housingWeight, setHousing: setHousingWeight,
              lockText: demo ? null : REAL_CONTROLS_LOCK_TEXT
            }}
            controls={controls}
            controlValues={controlValues}
            onControlChange={(key, value) => setControlValues((prev) => ({ ...prev, [key]: value }))}
            controlsLabel={demo ? "시설 조절변수 (x)" : "관측 입력 조정 (x)"}
            controlsEmptyText={demo ? null : real.missing ? "시뮬레이션 모델 준비 후 조정할 수 있습니다." : real.loading ? "관측 입력 확인 중…" : null}
            controlsLockTitle={realLockTitle ?? "조정할 조절변수가 없습니다."}
            onResetControls={() => { if (controls.length) setControlValues(defaultValues()); }}
            onRecommend={handleRecommend}
            // 실데이터는 ① 실행 없이 시뮬레이션 모델만으로 추천한다. 조회 실패면 눌러서 다시 확인(재시도).
            recommendDisabled={demo ? !analysisDone : !real.ready && !real.loadFailed}
            notReadyText={demo ? (analysisDone ? null : "STAGE ① 요인분석 실행 후 추천을 도출할 수 있습니다.") : realLockTitle}
            stale={recoStale}
          />
        </div>

        <SimulatorMapDetails
          appData={appData}
          currentRegion={currentRegion}
          allowSeed={demo}
          note={realUnavailableText("시설물·격자 정보는")}
        />

        {/* ── 분석: 예측 + 전 지자체 분포 ── */}
        <div className="grid-cols-2" style={{ marginTop: 24 }}>
          <Card
            title={demo ? "10개년 인구 예측 시뮬레이션 결과" : `다음 달 ${targetName} 예측 결과`}
            icon="fa-wand-magic-sparkles"
            dataSource={cardSource}
            sourceLabel={demo ? undefined : realSourceLabel}
            data-slot="sim.sim.forecast"
          >
            {demo && resultsReady ? (
              <>
                <div style={{ position: "relative", height: 200, width: "100%" }}>
                  <Line data={chartData} options={chartOpts} />
                </div>
                <div style={{ display: "flex", justifyContent: "space-around", marginTop: 16, textAlign: "center" }}>
                  <div>
                    <div style={{ fontSize: 11, color: "var(--text-secondary)" }}>시뮬레이션 인구 (10년 후)</div>
                    <div className="mock-data-output" style={{ fontSize: 18, fontWeight: 700, color: "var(--accent-blue)" }}>
                      {formatUnitValue(finalPop, "명")}명
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: 11, color: "var(--text-secondary)" }}>인구 증가율 예측</div>
                    <div className={growthClass} style={{ fontSize: 18, fontWeight: 700 }}>
                      {parseFloat(growthPercent) > 0 ? "+" : ""}
                      {growthPercent}%
                    </div>
                  </div>
                </div>
                <p className="chart-summary">
                  자연감소 기준 10년 후 {formatUnitValue(baseFinalPop, "명")}명, 정책 적용 시 {formatUnitValue(finalPop, "명")}명으로
                  기준 대비 {policyDifference >= 0 ? "+" : ""}{formatUnitValue(policyDifference, "명")}명입니다.
                </p>
              </>
            ) : !demo && simShown ? (
              <RealForecastResult chartData={chartData} chartOpts={chartOpts} result={simShown} unit={unit} label={label} />
            ) : demo ? (
              <PendingData running={analysisRunning} text="[요인분석 실행] 후 인구 예측 결과가 표시됩니다." />
            ) : real.ready ? (
              // 모델은 준비됐고 예측을 계산 중이거나 실패한 상태.
              <PendingData
                running={!real.predictFailed}
                text="다음 달 예측을 계산하지 못했습니다. 관측 입력을 다시 조정하거나 [정책 추천 도출]로 다시 계산하세요."
              />
            ) : (
              realPending
            )}
            {/* 계산 근거 — 예측 입력 기여도·검증 오차·기준선·활성 버전(이전 STAGE ①·② 실데이터 내용) */}
            <ModelEvidence slot="sim.sim.evidence" demo={demo} state={evidenceState} result={simShown ?? real.context} regions={regions} datasetName={real.datasetName} label={label} />
          </Card>

          <Card title="전 지자체 위험지수 vs 인구 분포" icon="fa-braille" dataSource={cardSource} data-slot="sim.sim.distribution">
            {demo && resultsReady ? (
              <>
                <p style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 12 }}>
                  전국 인구감소 지역(회색 점) 분포 속 시범 분석 대상 지자체의 상대 위치. 파란 점이 현재 선택
                  지자체입니다.
                </p>
                <div style={{ position: "relative", height: 220, width: "100%" }}>
                  <Scatter data={scatterData} options={scatterOpts} />
                </div>
                <p className="chart-summary">
                  {currentRegion.name}은 위험지수 {currentRegion.riskIndex}, 인구 {currentRegion.population.toLocaleString()}명입니다.
                  비교 대상 {distributionRegions.length}곳 중 위험지수는 낮은 순 {riskRank}위(낮을수록 고위험), 인구는 큰 순 {populationRank}위입니다.
                  회색은 전국 비교군, 주황은 분석 대상, 파랑은 현재 선택 지역입니다.
                </p>
              </>
            ) : (
              <PendingData
                running={analysisRunning && demo}
                text={demo ? "[요인분석 실행] 후 전 지자체 분포 분석이 표시됩니다." : realUnavailableText("지자체 비교 분포는")}
              />
            )}
          </Card>
        </div>

        {/* ── 시나리오 저장·비교 ── */}
        <ScenarioCompare
          region={currentRegion}
          allowSeed={demo}
          snapshot={{ budgetTotal, welfareWeight, industryWeight, housingWeight, controlValues }}
          onApply={(sc) => {
            setBudgetTotal(sc.budgetTotal);
            setWelfareWeight(sc.welfareWeight);
            setIndustryWeight(sc.industryWeight);
            setHousingWeight(sc.housingWeight);
            setControlValues((prev) => ({ ...prev, ...(sc.controlValues ?? {}) }));
            addConsoleLog(`INFO: 시나리오 '${sc.name}' 변수를 시뮬레이션에 적용했습니다.`);
          }}
          addConsoleLog={addConsoleLog}
        />
      </CollapsibleStage>

      {/* ── STAGE ④ 추천: 정책 추천 + AI 보고서 연계 (리포팅은 자동화 리포팅 탭 담당) ── */}
      <CollapsibleStage
        id="stage-report"
        no="STAGE ④"
        // reportFocus 는 데모 사례 정의의 문장이라 실데이터 표시에서는 쓰지 않는다.
        title="추천"
        sub={demo ? currentRegion.case.reportFocus : real.ready ? "관측 지표의 예측 민감도를 근거로 한 추천" : null}
        open={openStages["stage-report"]}
        onToggle={() => toggleStage("stage-report")}
      >
        <button type="button" className="btn btn-primary"
          onClick={() => { if (demo ? analysisDone : real.ready || real.loadFailed) handleRecommend(); }}
          aria-disabled={(demo ? !analysisDone : !real.ready && !real.loadFailed) || undefined}
          title={demo ? !analysisDone ? "요인분석을 먼저 실행하세요." : undefined : realLockTitle ?? undefined}>
          <i className="fa-solid fa-lightbulb" aria-hidden="true"></i> 정책 추천 도출
        </button>
        {recoReady && (demo || recommendation.items.length > 0) ? (
          demo
            ? <PolicyRecommendation region={currentRegion} ranked={recommendation} />
            : <PolicyRecommendation region={currentRegion} view={recommendation} sourceLabel={realSourceLabel} />
        ) : (
          <Card title="맞춤 정책 추천" icon="fa-lightbulb" dataSource={cardSource} data-slot="sim.report.panel">
            <div style={{ textAlign: "center", padding: "32px 16px", color: "var(--text-secondary)", fontSize: 13 }}>
              <i
                className="fa-solid fa-wand-magic-sparkles"
                style={{ fontSize: 28, color: "var(--accent-blue)", marginBottom: 12, display: "block" }}
              ></i>
              {!demo && real.missing ? (
                guide(REAL_GUIDE_TEXT)
              ) : recoReady ? (
                "관측 범위 안에서 증가 예측을 만든 조정 항목이 없습니다. 행정동이나 학습 대상을 바꿔 확인하세요."
              ) : (
                <>
                  STAGE ③에서 {demo ? "정책 변수를" : "관측 입력을"} 조정한 뒤 <strong>[정책 추천 도출]</strong> 버튼을 누르면, 시뮬레이션
                  결과에 맞는 맞춤 정책 추천이 생성됩니다.
                </>
              )}
            </div>
          </Card>
        )}
        <ModelEvidence
          slot="sim.report.evidence"
          summary="추천 근거"
          demo={demo}
          state={evidenceState}
          result={(recoReady ? recoSim : null) ?? simShown ?? real.context}
          regions={regions}
          datasetName={real.datasetName}
          label={label}
        />

        <div className="pl-report-handoff" data-slot="sim.report.handoff">
          <div className="pl-report-handoff-text">
            <i className="fa-solid fa-file-invoice" aria-hidden="true"></i>
            <div>
              <strong>AI 정책 보고서 생성 및 평가</strong>
              <p className="mock-data-output">
                {demo
                  ? `${currentRegion.name} · ${currentRegion.case.reportFocus}`
                  : real.ready
                    ? `${currentRegion.name} · 다음 달 ${targetName} 예측 기반 추천`
                    : `${currentRegion.name} · 활성 모델 준비 필요`}
              </p>
            </div>
          </div>
          <button className="btn btn-primary" onClick={() => navigateToTab("tab-reporter")}>
            보고서 생성으로 이동 <i className="fa-solid fa-arrow-right"></i>
          </button>
        </div>
      </CollapsibleStage>
    </>
  );
}
