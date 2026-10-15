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
import { useAppState } from "../context/AppStateContext.jsx";
import { useChartTheme } from "../hooks/useChartTheme.js";
import { useRealSimulation } from "../hooks/useRealSimulation.js";
import { getScrollBehavior } from "../lib/motion.js";
import { rankStrategies } from "../constants/policyStrategies.js";
import { YEAR_LABELS, computeTrends, budgetToFactor, controlBoostOf, DEMO_ANALYSIS_STEPS as ANALYSIS_STEPS, DEMO_ANALYSIS_STEP_MS as ANALYSIS_STEP_MS } from "../lib/simulation.js";
import {
  REAL_TARGETS, REAL_ANALYSIS_STEPS, REAL_GUIDE_TEXT, REAL_CONTROLS_LOCK_TEXT, REAL_DIAGNOSIS_NOTE, realUnavailableText,
  observedFeatureLabel, rankSensitivity, targetUnit, formatValue, formatMonth,
  contributionFactors, contributionSplit, realSliderControls, forecastSeries, sensitivityRecommendation, regionRealStatus
} from "../lib/realSimulation.js";

// 실데이터 표시에서 관측 입력을 바꾼 뒤 시뮬레이션 추론(읽기 전용 계산)을 다시 부르기까지의 대기.
const SIM_DEBOUNCE_MS = 500;
const STAGE_IDS = ["stage-factor", "stage-result", "stage-sim", "stage-report"];

// 정책 시뮬레이터 — 기준선 4단계 구조 하나를 두 모드가 함께 쓴다(모드는 데이터 출처만 바꾼다).
// 데모 표시: 지역 사례(mock_data.json)와 화면 예시 계산. 실데이터 표시: 같은 지역·학습 대상의 활성 요인분석·
// 시뮬레이션 버전 추론(useRealSimulation). 실데이터 대응이 없는 요소는 같은 자리에 행동 안내를 둔다.
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
  const [dong, setDong] = useState("");
  const real = useRealSimulation({ enabled: !demo, region: currentRegion.id, target, dong });
  const unit = targetUnit(target);
  const targetName = REAL_TARGETS[target];
  const dongOptions = real.contexts?.simulation.dong_options ?? [];
  const label = (feature) => observedFeatureLabel(feature, target, dongOptions);
  const regions = appData.regions.map(({ id, name }) => ({ id, name }));
  const factorProof = real.contexts?.factor.provenance;
  const simProof = real.contexts?.simulation.provenance;
  const dongName = dongOptions.find((row) => row.code === simProof?.dong_code)?.name ?? simProof?.dong_code ?? "—";
  const realSourceLabel = real.ready ? `실데이터 · ${real.datasetName(factorProof.dataset_id)}` : undefined;

  // 조절변수(x) — 데모는 지역 사례의 시설 변수, 실데이터는 시뮬레이션 모델의 조정 가능한 관측 입력. 바뀌면 기본값으로.
  const controls = demo ? currentRegion.case.simulation.controls ?? [] : realSliderControls(real.contexts?.simulation, label, unit);
  const controlsKey = `${currentRegion.id}|${controls.map((c) => `${c.key}:${c.default}`).join(",")}`;
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
  const currentSig = `${currentRegion.id}|${target}|${dong}|${welfareWeight}|${industryWeight}|${housingWeight}|${budgetTotal}|${controlSig}`;
  const recoStale = recommendation !== null && recoSig !== currentSig;

  // 단계 접기/펼치기 상태 (기본 전부 펼침)
  const [openStages, setOpenStages] = useState(Object.fromEntries(STAGE_IDS.map((id) => [id, true])));
  const revealStage = (id, delay = 60) => {
    setOpenStages((s) => ({ ...s, [id]: true }));
    setTimeout(() => {
      document.getElementById(id)?.scrollIntoView({ behavior: getScrollBehavior(), block: "start" });
    }, delay);
  };

  // STAGE① 요인분석 실행 상태 — idle(미실행) → running(단계 진행) → done(결과 공개).
  const [analysisStatus, setAnalysisStatus] = useState("idle");
  const [analysisStep, setAnalysisStep] = useState(0); // 1-based 진행 단계
  const [analysis, setAnalysis] = useState(null); // 실데이터 요인분석 추론 결과
  const [simResult, setSimResult] = useState(null); // 실데이터 시뮬레이션 추론 결과
  const [simSig, setSimSig] = useState("");
  const analysisTimerRef = useRef(null);

  // 지역·학습 대상·행정동 변경 시 분석 상태 초기화 (지역별 요인분석은 별도 실행)
  useEffect(() => {
    if (analysisTimerRef.current) {
      clearTimeout(analysisTimerRef.current);
      analysisTimerRef.current = null;
    }
    setAnalysisStatus("idle");
    setAnalysisStep(0);
    setAnalysis(null);
    setSimResult(null);
    setRecommendation(null);
    setRecoSig("");
    setRecoSim(null);
    // real.revision: 활성 버전이 바뀌어(409) 입력을 다시 읽으면 이전 버전 기준 결과를 비운다.
  }, [currentRegion, target, dong, real.revision]);

  // 데모 분석 단계 진행 — 단계별 로그 출력 후 완료 시 STAGE② 자동 펼침
  useEffect(() => {
    if (!demo || analysisStatus !== "running" || analysisStep === 0) return undefined;

    if (analysisStep > ANALYSIS_STEPS.length) {
      setAnalysisStatus("done");
      addConsoleLog(
        `SUCCESS: ${currentRegion.name} 사회문제 요인분석 완료 — STAGE ② 요인분석 결과가 갱신되었습니다.`
      );
      addConsoleLog(
        "INFO: 도출 파라미터(조절변수·상관변수)가 STAGE ③ 시뮬레이션 입력으로 전달되었습니다."
      );
      setOpenStages((s) => ({ ...s, "stage-result": true, "stage-sim": true }));
      // 분석 결과가 갱신된 STAGE ②로 이동해 완료가 즉시 보이게 한다
      revealStage("stage-result", 120);
      return undefined;
    }

    const step = ANALYSIS_STEPS[analysisStep - 1];
    addConsoleLog(`INFO: [요인분석 ${analysisStep}/${ANALYSIS_STEPS.length}] ${step.log}`);
    analysisTimerRef.current = setTimeout(() => setAnalysisStep((s) => s + 1), ANALYSIS_STEP_MS);

    return () => {
      if (analysisTimerRef.current) clearTimeout(analysisTimerRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [demo, analysisStatus, analysisStep, currentRegion, addConsoleLog]);

  const handleRunAnalysis = async () => {
    if (analysisStatus === "running" || (!demo && !real.ready)) return;
    setOpenStages((s) => ({ ...s, "stage-factor": true }));
    setRecommendation(null);
    setRecoSig("");
    setRecoSim(null);
    setAnalysis(null);
    setSimResult(null);
    setAnalysisStatus("running");
    setAnalysisStep(1);
    if (demo) {
      addConsoleLog(`INFO: ${currentRegion.name} 사회문제 요인분석 실행 — XAI·딥러닝·통계 융합 파이프라인 시작.`);
      return;
    }
    // 실데이터: 1 목록·입력 재확인 → 2 활성 모델 추론 → 3 기여도 합계·근거 검증 → 4 완료
    addConsoleLog(`INFO: ${currentRegion.name} 요인분석 실행 — 활성 요인분석 모델로 ${targetName} 입력 기여도를 계산합니다.`);
    const result = await real.runAnalysis(setAnalysisStep);
    if (!result) {
      setAnalysisStatus("idle");
      setAnalysisStep(0);
      addConsoleLog(`WARN: ${currentRegion.name} 요인분석을 완료하지 못했습니다. 안내 문구를 확인한 뒤 다시 실행하세요.`, false, true);
      return;
    }
    setAnalysis(result);
    setAnalysisStatus("done");
    addConsoleLog(`SUCCESS: ${currentRegion.name} 요인분석 완료 — 입력 기여도 ${result.contributions.length}개를 STAGE ② 기여 방향에 반영했습니다.`);
    setOpenStages((s) => ({ ...s, "stage-result": true, "stage-sim": true }));
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
  // 관측 입력을 바꾸면 잠시 뒤 다음 달 예측을 다시 계산한다(데모의 실시간 추이 갱신과 같은 자리).
  const simTimerRef = useRef(null);
  useEffect(() => {
    if (demo || !analysisDone || !real.ready) return undefined;
    simTimerRef.current = setTimeout(runSimulation, SIM_DEBOUNCE_MS);
    return () => clearTimeout(simTimerRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [demo, analysisDone, real.ready, controlSig]);

  const handleRecommend = async () => {
    if (demo) {
      const ranked = rankStrategies(
        currentRegion,
        { welfare: welfareWeight, industry: industryWeight, housing: housingWeight },
        budgetFactor
      );
      setRecommendation(ranked);
      setRecoSig(currentSig);
      addConsoleLog(`INFO: ${currentRegion.name} 정책 추천 도출 완료 — 1순위 '${ranked[0].name}' (점수 ${ranked[0].score.toLocaleString()}).`);
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
        ? `INFO: ${currentRegion.name} 정책 추천 도출 완료 — 1순위 '${label(ranking[0].feature)} ${ranking[0].to > ranking[0].from ? "증가" : "감소"}' (모델 예측상 변화 +${formatValue(ranking[0].delta)}${unit}).`
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
  const series = demo ? null : forecastSeries(real.contexts?.simulation, simResult);
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
    [demo, baseTrend, simTrend, simResult, real.contexts]
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
  const realDeltaPercent = simResult && simResult.baseline_prediction
    ? ((simResult.delta / Math.abs(simResult.baseline_prediction)) * 100).toFixed(1)
    : "0.0";

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
  const resultsReady = analysisDone && (demo || Boolean(analysis));
  const recoReady = resultsReady && recommendation && !recoStale;

  // 스텝퍼 완료 표시: ①② 요인분석 완료, ③④ 정책 추천 도출 완료 기준
  const doneStages = [
    ...(analysisDone ? ["stage-factor", "stage-result"] : []),
    ...(analysisDone && recommendation && !recoStale ? ["stage-sim", "stage-report"] : [])
  ];

  // ── 실데이터 표시의 같은 자리 내용 ──
  const guide = (
    <div className="real-flow-guide" data-guide>
      <p>{REAL_GUIDE_TEXT}</p>
      <button type="button" className="btn btn-secondary" onClick={() => navigateToTab("tab-dataops")}>DataOps에서 학습데이터 등록</button>{" "}
      <button type="button" className="btn btn-primary" onClick={() => navigateToTab("tab-mlops-orch")}>오케스트레이터에서 학습·활성화</button>
    </div>
  );
  const realPending = (text) => (real.missing ? <PendingData text={REAL_GUIDE_TEXT} /> : <PendingData running={analysisRunning} text={text} />);
  const factorView = demo ? null : {
    sub: real.missing ? `${currentRegion.name} · 실데이터 모델 준비 필요` : `${currentRegion.name} · 실데이터 ${targetName}`,
    runInfo: real.ready
      ? `등록 학습데이터 ${real.datasetName(factorProof.dataset_id)} · 요인분석 활성 버전 ${String(factorProof.version_id).slice(0, 12)}`
      : "활성 모델 준비 후 실행할 수 있습니다.",
    sourceLabel: realSourceLabel,
    cards: [
      {
        title: "입력 지표 기여도 분석",
        body: real.missing ? guide : analysis ? (
          <>
            <ul className="pl-factor-list">
              {contributionFactors(analysis, label).map((f) => (
                <li key={f.feature} className={f.direction === "-" ? "neg" : "pos"}>
                  <span className="pl-factor-sign">{f.direction}</span>
                  {f.label} — 관측 입력 {formatValue(f.input)} · 기여도 {formatValue(f.value)} {unit}
                </li>
              ))}
            </ul>
            <div className="pl-source-note">
              <i className="fa-solid fa-database" aria-hidden="true"></i> 연계 데이터: 등록 학습데이터 {real.datasetName(factorProof.dataset_id)}
            </div>
          </>
        ) : realPending("[요인분석 실행] 후 입력 지표 기여도가 표시됩니다.")
      },
      {
        title: "예측 모델",
        body: analysis ? (
          <>
            <div className="pl-model-box pl-model-xai">
              <span className="pl-model-tag">요인분석 모델</span>
              <strong>활성 버전 {analysis.provenance.version_id}</strong>
              <p>입력별 선형 기여도로 분해합니다. 기여도 합 + 기준값 {formatValue(analysis.intercept)} {unit} = 예측값 {formatValue(analysis.prediction)} {unit}</p>
            </div>
            <div className="pl-model-box pl-model-dl">
              <span className="pl-model-tag">입력 항목</span>
              <div className="pl-model-chips">
                {analysis.provenance.features.map((feature) => <span className="pl-chip" key={feature}>{label(feature)}</span>)}
              </div>
            </div>
          </>
        ) : realPending("[요인분석 실행] 후 예측 모델 정보가 표시됩니다.")
      },
      {
        title: "검증 결과",
        body: analysis ? (
          <>
            <p className="pl-stat-baseline">
              검증 평균 절대 오차 {formatValue(analysis.provenance.metrics?.mae)} {unit} · 기준선 {formatValue(analysis.provenance.baseline?.mae)} {unit}
              {analysis.provenance.baseline?.name ? ` (${analysis.provenance.baseline.name})` : ""}
            </p>
            <div className="pl-stage-label">분석 대상</div>
            <ul className="pl-mini-list">
              <li>학습 대상: {targetName}</li>
              <li>행정동: {dongName}</li>
              <li>관측 기간: {formatMonth(analysis.provenance.observed_from)}~{formatMonth(analysis.provenance.observed_to)}</li>
              <li>예측월: {formatMonth(analysis.provenance.prediction_period)}</li>
            </ul>
          </>
        ) : realPending("[요인분석 실행] 후 검증 결과가 표시됩니다.")
      }
    ]
  };
  const split = contributionSplit(analysis, label);
  const resultView = demo ? null : {
    corrTitle: "기여 방향",
    posHead: "예측을 높인 입력",
    negHead: "예측을 낮춘 입력",
    positive: split.positive,
    negative: split.negative,
    lockedText: "[요인분석 실행] 후 기여 방향이 표시됩니다.",
    blockedText: real.missing ? REAL_GUIDE_TEXT : null,
    diagnosisNote: REAL_DIAGNOSIS_NOTE,
    paramText: real.contexts ? `조정 가능한 관측 입력 ${controls.length}개 → STAGE ③ 시뮬레이션 입력` : "→ STAGE ③ 시뮬레이션 입력"
  };
  const rangeProof = simProof?.range_provenance;
  const frame = demo
    ? { objective: sim.objective, demand: sim.factors.demand, supply: sim.factors.supply, adjust: sim.factors.adjust, constraint: `${sim.constraint} · ${budgetTotal.toLocaleString()}억` }
    : real.ready
      ? {
          objective: `다음 달 ${targetName} 예측 (예측월 ${formatMonth(simProof.prediction_period)})`,
          demand: `${targetName} 관측 · ${dongName}`,
          supply: simResult ? `기준 예측 ${formatValue(simResult.baseline_prediction)} ${unit}` : "–",
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
        행정동
        <select
          aria-label="행정동"
          value={demo ? "all" : dong || simProof?.dong_code || ""}
          disabled={analysisRunning || (!demo && !real.contexts)}
          onChange={(e) => { if (!demo) setDong(e.target.value); }}
        >
          {demo && <option value="all">전체</option>}
          {!demo && !real.contexts && <option value="">모델 준비 후 선택</option>}
          {!demo && dongOptions.map((row) => <option key={row.code} value={row.code}>{row.name}</option>)}
        </select>
      </label>
      {!demo && real.loading && <p role="status">모델과 관측 입력 확인 중…</p>}
      {!demo && real.feedback && <p className="async-feedback is-error" role="alert">{real.feedback}</p>}
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
        region={currentRegion}
        open={openStages["stage-factor"]}
        onToggle={() => toggleStage("stage-factor")}
        status={analysisStatus}
        stepIndex={analysisStep}
        steps={demo ? ANALYSIS_STEPS : REAL_ANALYSIS_STEPS}
        onRun={handleRunAnalysis}
        view={factorView}
        conditions={conditions}
        runLockTitle={!demo && !real.ready ? "활성 모델을 먼저 준비하세요." : null}
        evidence={
          <ModelEvidence slot="sim.factor.evidence" demo={demo} result={analysis ?? real.contexts?.factor} regions={regions} datasetName={real.datasetName} />
        }
      />

      {/* ── STAGE ② 요인분석 결과 ── */}
      <FactorResultStage
        region={currentRegion}
        open={openStages["stage-result"]}
        onToggle={() => toggleStage("stage-result")}
        locked={analysisStatus !== "done"}
        running={analysisRunning}
        view={resultView}
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
            controlsEmptyText={demo ? null : real.missing ? REAL_GUIDE_TEXT : real.loading ? "관측 입력 확인 중…" : null}
            onResetControls={() => { if (controls.length) setControlValues(defaultValues()); }}
            onRecommend={handleRecommend}
            recommendDisabled={!analysisDone}
            notReadyText={analysisDone ? null : "STAGE ① 요인분석 실행 후 추천을 도출할 수 있습니다."}
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
                      {finalPop.toLocaleString()}명
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
                  자연감소 기준 10년 후 {baseFinalPop.toLocaleString()}명, 정책 적용 시 {finalPop.toLocaleString()}명으로
                  기준 대비 {policyDifference >= 0 ? "+" : ""}{policyDifference.toLocaleString()}명입니다.
                </p>
              </>
            ) : !demo && resultsReady && simResult ? (
              <>
                <div style={{ position: "relative", height: 200, width: "100%" }}>
                  <Line data={chartData} options={chartOpts} />
                </div>
                <div style={{ display: "flex", justifyContent: "space-around", marginTop: 16, textAlign: "center" }}>
                  <div>
                    <div style={{ fontSize: 11, color: "var(--text-secondary)" }}>조정 예측 ({formatMonth(simResult.provenance.prediction_period)})</div>
                    <div style={{ fontSize: 18, fontWeight: 700, color: "var(--accent-blue)" }}>
                      {formatValue(simResult.prediction)}{unit}
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: 11, color: "var(--text-secondary)" }}>기준 대비 변화</div>
                    <div className={simResult.delta >= 0 ? "trend-up" : "trend-down"} style={{ fontSize: 18, fontWeight: 700 }}>
                      {simResult.delta > 0 ? "+" : ""}{formatValue(simResult.delta)}{unit} ({simResult.delta > 0 ? "+" : ""}{realDeltaPercent}%)
                    </div>
                  </div>
                </div>
                <p className="chart-summary">
                  기준 예측 {formatValue(simResult.baseline_prediction)}{unit}, 조정 예측 {formatValue(simResult.prediction)}{unit}로
                  모델 예측상 {formatValue(simResult.delta)}{unit} 변화입니다. 검증 평균 절대 오차는 {formatValue(simResult.provenance.metrics?.mae)}{unit}입니다.
                </p>
                {simResult.outside_observed_range.length > 0 && (
                  <p className="async-feedback is-error" role="status">
                    관측 범위 밖 입력: {simResult.outside_observed_range.map(label).join(", ")}. 해당 구간의 예측 정확도는 확인되지 않았습니다.
                  </p>
                )}
              </>
            ) : demo ? (
              <PendingData running={analysisRunning} text="[요인분석 실행] 후 인구 예측 결과가 표시됩니다." />
            ) : analysisDone ? (
              // 요인분석은 끝났고 예측을 계산 중이거나 실패한 상태 — 다시 실행하라고 하지 않는다.
              <PendingData
                running={!real.feedback}
                text="다음 달 예측을 계산하지 못했습니다. 위 안내를 확인한 뒤 관측 입력을 다시 조정하세요."
              />
            ) : (
              realPending("[요인분석 실행] 후 다음 달 예측 결과가 표시됩니다.")
            )}
            <ModelEvidence slot="sim.sim.evidence" demo={demo} result={simResult ?? real.contexts?.simulation} regions={regions} datasetName={real.datasetName} />
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
                guide
              ) : recoReady ? (
                "관측 범위 안에서 증가 예측을 만든 조정 항목이 없습니다. 행정동이나 학습 대상을 바꿔 확인하세요."
              ) : (
                <>
                  STAGE ③에서 정책 변수를 조정한 뒤 <strong>[정책 추천 도출]</strong> 버튼을 누르면, 시뮬레이션
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
          result={recoSim ?? simResult ?? real.contexts?.simulation}
          regions={regions}
          datasetName={real.datasetName}
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
