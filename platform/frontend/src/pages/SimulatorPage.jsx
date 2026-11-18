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
import ForecastCompare from "../components/ForecastCompare.jsx";
import { useAppState } from "../context/AppStateContext.jsx";
import { useChartTheme } from "../hooks/useChartTheme.js";
import { useRealSimulation } from "../hooks/useRealSimulation.js";
import { useFactorAnalysis } from "../hooks/useFactorAnalysis.js";
import { useForecastCompare } from "../hooks/useForecastCompare.js";
import { usePublicTrends, useRegionStay } from "../hooks/usePublicTrends.js";
import { getScrollBehavior } from "../lib/motion.js";
import { facilityRows, recoControlSig, unlinkedControlNote, withRegionStay } from "../lib/facilityRecommendation.js";
import { CHART_ITEMS, GUIDE, reportHandoffOf } from "../lib/recoText.js";
import { YEAR_LABELS, computeTrends, budgetToFactor, controlBoostOf, ANALYSIS_STEPS } from "../lib/simulation.js";
import { instanceLabel } from "../lib/modelops.js";
import {
  REAL_TARGETS, REAL_GUIDE_TEXT, REAL_CONTROLS_LOCK_TEXT, FACTOR_GUIDE_TEXT, FACTOR_PENDING_TEXT, FACTOR_RUN_LOCK_TEXT,
  diagnosisView, simFrameView, quarterYoyChange, AUTO_TARGET, AUTO_TARGET_LABEL, DEMO_FACTOR_TARGET, realUnavailableText, selectedFactorModel, inferenceMatches, inferenceContextKey, factorAnalysisView,
  observedFeatureLabel, targetUnit, formatUnitValue,
  realSliderControls, forecastSeries, regionRealStatus, FACTOR_SCOPES, activeModelPair, forecastCompareView,
  periodBounds, resolvePeriod, monthInputValue, parseMonthInput, PERIOD_BASELINES
} from "../lib/realSimulation.js";

// 실데이터 표시에서 관측 입력을 바꾼 뒤 시뮬레이션 추론(읽기 전용 계산)을 다시 부르기까지의 대기.
const SIM_DEBOUNCE_MS = 500;
const STAGE_IDS = ["stage-factor", "stage-result", "stage-sim", "stage-report"];

// 정책 시뮬레이터 — 기준선 4단계 구조 하나를 두 모드가 함께 쓴다(모드는 데이터 출처만 바꾼다).
// ①·②: 두 모드 모두 선택한 기존 활성 요인분석 모델의 결과(useFactorAnalysis).
// ③·④: 데모는 지역 사례(mock_data.json)와 화면 예시 계산, 실데이터는 같은 지역·학습 대상의 활성 시뮬레이션 버전
// 추론(useRealSimulation) — 실데이터 ③·④는 ① 실행과 무관하다. 실데이터 대응이 없는 요소는 같은 자리에 행동 안내를 둔다.
/** ③ 설정 띠의 한 칸 — 이름은 작은 표지, 값은 크게(조절변수는 항목 칩), 보조 정보는 한 줄. */
function SimFrameCell({ kind = "", tag, cell, status = null }) {
  return (
    <div className={`pl-sim-cell${kind ? ` pl-sim-${kind}` : ""}`}>
      <div className="pl-sim-cell-head">
        <span className="pl-sim-tag">{tag}</span>
        {status && (
          <span className="pl-param-chip">
            <i className="fa-solid fa-circle-check" aria-hidden="true"></i> {status}
          </span>
        )}
      </div>
      {cell.chips ? (
        <div className="pl-sim-chips mock-data-output">
          {cell.chips.length ? cell.chips.map((chip) => <span key={chip} className="pl-sim-chip">{chip}</span>) : <strong className="pl-sim-value">–</strong>}
        </div>
      ) : (
        <strong className="pl-sim-value mock-data-output">{cell.value}</strong>
      )}
      {cell.meta && <span className="pl-sim-meta">{cell.meta}</span>}
    </div>
  );
}

export default function SimulatorPage() {
  const {
    appData, currentRegion, setCurrentRegion,
    welfareWeight, setWelfareWeight, industryWeight, setIndustryWeight, housingWeight, setHousingWeight,
    budgetTotal, setBudgetTotal, addConsoleLog, navigateToTab, mockDataVisible, simulatorRuns, updateSimulatorRun, sendToReport
  } = useAppState();
  const ct = useChartTheme();
  const demo = mockDataVisible;
  const [searchText, setSearchText] = useState("");

  // 실데이터 분석 조건 — 두 모드에 같은 select 를 두고 데모는 선택지 1개.
  const [target, setTarget] = useState("observed_sales_krw");
  // 실데이터 학습 대상 "자동(가장 나빠진 지표)" — 기본값. [요인분석 실행]이 지표별로 추론해 target 을 고른 지표로 옮긴다.
  const [auto, setAuto] = useState(true);
  // ① 요인 범위 — ② 기여도 카드의 표시 범위(전체 / 정책 관련 요인만). 두 모드 같음, 기본 전체.
  const [factorScope, setFactorScope] = useState("all");
  // ① 분석 기간·비교 기준(결정 29) — 기간은 고른 달({ from, to }), null이면 최근 12개월. 기준 기본은 학습 기간 평균(기존 기여도와 같은 뜻).
  const [periodPick, setPeriodPick] = useState(null);
  const [baseline, setBaseline] = useState("train_mean");
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
    modelId: picked?.id, versionId: picked?.version, period: periodPick, baseline,
    log: (message) => addConsoleLog(message, false, message.startsWith("WARN")) });
  const factorModels = factor.catalog?.mode === mode && factor.catalog?.source === mode
    ? factor.catalog.instances.filter((row) => appData.regions.some((r) => r.id === row.region_id)
      && (demo ? row.target === DEMO_FACTOR_TARGET : Object.hasOwn(REAL_TARGETS, row.target))
      && selectedFactorModel(factor.catalog, row.id, row.region_id, row.target, mode)) : [];
  const factorModel = selectedFactorModel(factor.catalog, picked?.id, currentRegion.id, factorTarget, mode);
  // 분석 기간 입력의 범위 — 선택 모델 활성 버전의 학습데이터 관측 기간(실행 때 훅도 같은 규칙으로 맞춘다).
  const factorDatasetId = factor.catalog?.versions?.find((row) => row.id === factorModel?.active_version)?.dataset_id;
  const periodLimits = periodBounds(factor.catalog?.datasets?.find((row) => row.id === factorDatasetId), demo);
  const periodRange = resolvePeriod(periodLimits, periodPick);
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
  const regionNameOf = (id) => appData.regions.find((r) => r.id === id)?.name ?? id;
  const simProof = real.context?.provenance;
  const dongName = dongOptions.find((row) => row.code === simProof?.dong_code)?.name ?? simProof?.dong_code ?? "—";
  // ③ 릿지·Chronos-2 비교 줄 — 실데이터는 시뮬레이션 활성 모델(선택 행정동), 데모는 합성 패널의 요인분석 활성 모델(같은 경로).
  const compareModel = demo ? factorModel : real.ready ? activeModelPair(real.catalog, currentRegion.id, target).simulation : undefined;
  const compareDong = demo ? "" : simProof?.dong_code;
  const compare = useForecastCompare({ mode, model: compareModel, dong: compareDong });
  const compareView = forecastCompareView(compare.result, { mode, modelId: compareModel?.id, versionId: compareModel?.active_version,
    dong: compareDong }, demo ? targetUnit(DEMO_FACTOR_TARGET) : unit);
  // 조절변수(x) — 데모는 지역 사례의 시설 변수, 실데이터는 시뮬레이션 모델의 조정 가능한 관측 입력. 바뀌면 기본값으로.
  const controls = demo ? currentRegion.case.simulation.controls ?? [] : realSliderControls(real.context, label, unit);
  const controlsKey = [mode, currentRegion.id, demo ? factorModel?.active_version : simProof?.version_id,
    controls.map((c) => `${c.key}:${c.default}`).join(",")].join("|");
  const defaultValues = () => Object.fromEntries(controls.map((c) => [c.key, c.default]));
  const [controlValues, setControlValues] = useState(defaultValues);
  useEffect(() => {
    // 모델·버전이 늦게 도착해 조절변수 구성이 바뀌어도, 이 지역에서 사용자가 이미 바꾼 값(같은 키)은 지키고
    // 새 키만 기본값으로 둔다. 지역이 바뀌면(userEdited 초기화) 기본값으로 시작한다.
    const defaults = defaultValues();
    setControlValues((prev) => (userEdited.current === currentRegion.id
      ? Object.fromEntries(Object.entries(defaults).map(([key, value]) => [key, Object.hasOwn(prev, key) ? prev[key] : value]))
      : defaults));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [controlsKey]);

  // 총 예산(제약요소) → 체감수익 곡선. 시설 조절변수 → 정규화×weight 합산 부스트(데모 계산).
  const budgetFactor = budgetToFactor(budgetTotal);
  const controlBoost = controlBoostOf(controls, controlValues);

  // 정책 추천(실행 버튼으로 도출) 상태 — 시설 추천 행({ items }). 실데이터는 시설 대장이 없어 도출하지 않는다(같은 자리에 행동 안내).
  const [recommendation, setRecommendation] = useState(null);
  // ④ 고른 사업 — 지도·추천 지수 차트·칩·표가 같은 값을 바꾼다(없거나 1~5순위 밖이면 1순위).
  const [recoPick, setRecoPick] = useState(null);
  const [recoSig, setRecoSig] = useState("");
  const controlSig = controls.map((c) => `${c.key}:${controlValues[c.key] ?? c.default}`).join(",");
  const contextKey = inferenceContextKey({ mode, region: currentRegion.id, target: factorTarget, dong: demo ? "" : dong,
    factorModel, simulationProof: demo ? null : simProof });
  const currentSig = `${contextKey}|${welfareWeight}|${industryWeight}|${housingWeight}|${budgetTotal}|${controlSig}`;
  // 추천을 다시 도출해야 하는 입력 — 데모 시설 추천은 시설 값에 연결된 조절변수만(예산·정책 강조는 10년 예측에만 쓰인다).
  // 시설 값이 없는 조절변수를 바꾸면 재도출 안내 대신 같은 자리에 자료 등록 안내를 둔다.
  // 지역 월평균 체류인구는 공표 통계 API 값(시드에 없음) — 늦게 도착하면 추천 입력이 바뀐 것으로 본다.
  // 그 기간은 두 모드 모두 ④ 선정 사유의 분석 기간으로 쓴다.
  const stayStat = useRegionStay(currentRegion.name);
  const regionStay = demo ? stayStat?.value ?? null : null;
  // 데모 ② 진단 머리 칸의 근거 — 생활인구 공표 통계(정책 추천 전망 그래프와 같은 값)
  const demoTrends = usePublicTrends(demo ? currentRegion.name : undefined);
  const recoInputSig = demo ? `${contextKey}|${recoControlSig(controls, controlValues)}|stay:${regionStay ?? ""}` : currentSig;
  const controlNote = demo ? unlinkedControlNote(controls, controlValues) : null;
  const recoStale = recommendation !== null && recoSig !== recoInputSig;
  const deriveFacilities = (region) => ({
    items: facilityRows(region.id, appData.facilities, controls, controlValues, withRegionStay(appData.conversion, region.id, region.id === currentRegion.id ? regionStay : null))
  });
  // 자동 계산은 ① 완료 시점의 입력(공표 통계 도착 포함)으로 추천을 도출한다.
  const deriveNow = useRef(null);
  deriveNow.current = () => [deriveFacilities(currentRegion), recoInputSig];

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
  const autoReady = !demo && auto && factorModels.some((row) => row.region_id === currentRegion.id);
  const factorLockTitle = factorModel || factorRetry || autoReady ? null
    : factor.loading ? "모델 목록을 확인하는 중입니다." : FACTOR_RUN_LOCK_TEXT;
  const factorMissing = !factor.loading && !factor.loadFailed && (!(factorModel || autoReady) || factor.missing);

  // 지역·학습 대상·행정동 변경 시 결과 초기화 (지역별 요인분석은 별도 실행 — 요인분석 상태는 훅이 비운다)
  useEffect(() => {
    // 다른 지역에 갔다가 돌아와도 이전 행정동이 되살아나지 않게 기억을 지운다(화면 값은 이미 "전체").
    setDongPick((pick) => (pick.regionId === currentRegion.id ? pick : { regionId: currentRegion.id, code: "" }));
    if (userEdited.current !== currentRegion.id) userEdited.current = null;
    setSimResult(null);
    setRecommendation(null);
    setRecoSig("");
    setRecoPick(null);
    // real.revision: 활성 버전이 바뀌거나(409) 다시 확인해 입력을 다시 읽으면 이전 결과를 비운다.
  }, [demo, currentRegion.id, target, dong, real.revision]);

  // 데모 첫 화면: 활성 요인분석 모델이 준비된 현재 세대에서 지역마다 한 번 ①→④를 같은 입력으로 계산한다.
  // 이미 계산한 지역(같은 활성 버전)은 앱 상태의 결과를 되살린다. 실패해도 다시 시도하지 않고, 도중에 입력을
  // 바꿨으면 결과를 버린다(사용자 입력을 덮지 않음). 화면 이동·초점 이동은 하지 않는다.
  const sigRef = useRef(currentSig);
  sigRef.current = currentSig;
  // 사용자가 직접 입력을 바꾼 지역 id — 현재 지역과 같으면 보존 결과를 되살리지 않고 늦은 기본값 초기화도 하지 않는다.
  const userEdited = useRef(null);
  const edited = (setter) => (...args) => { userEdited.current = currentRegion.id; setter(...args); };
  const demoModelReady = demo && Boolean(factorModel) && !factor.loading && picked?.version === factorModel?.active_version;
  useEffect(() => {
    if (!demoModelReady) return;
    const regionId = currentRegion.id;
    const version = factorModel.active_version;
    const saved = simulatorRuns[regionId]?.version === version ? simulatorRuns[regionId] : null;
    if (saved?.result) {
      // 결과와 그때 계산에 쓴 입력(예산·정책 가중치·시설)을 함께 되살린다. 이 지역에 들어온 뒤 사용자가 이미 입력을
      // 바꿨으면 과거 결과로 덮지 않는다(복원하지 않음).
      if (analysisStatus === "idle" && !factorResult && userEdited.current !== regionId) {
        const inputs = saved.inputs ?? {};
        if (inputs.controlValues) setControlValues(inputs.controlValues);
        if (Number.isFinite(inputs.budgetTotal)) setBudgetTotal(inputs.budgetTotal);
        if (Number.isFinite(inputs.welfareWeight)) setWelfareWeight(inputs.welfareWeight);
        if (Number.isFinite(inputs.industryWeight)) setIndustryWeight(inputs.industryWeight);
        if (Number.isFinite(inputs.housingWeight)) setHousingWeight(inputs.housingWeight);
        factor.restore(saved.result);
        setRecommendation(saved.recommendation ?? null);
        setRecoSig(saved.recoSig ?? "");
      }
      return;
    }
    if (saved?.attempted || analysisStatus !== "idle") return;
    updateSimulatorRun(regionId, { version, attempted: true, result: null, recommendation: null, recoSig: "" });
    const startSig = sigRef.current;
    const region = currentRegion;
    (async () => {
      // 입력이 바뀌었으면 ①의 결과도 게시하지 않고 버린다(훅 안에서 결과 게시 직전에 확인).
      const analysis = await factor.run({ accept: () => sigRef.current === startSig });
      if (!analysis || sigRef.current !== startSig) return;
      const [derived, recoSigNow] = deriveNow.current();
      setRecommendation(derived);
      setRecoSig(recoSigNow);
      addConsoleLog(`INFO: ${region.name} 요인분석·시설 활용 추천 계산 완료 — 시설 ${derived.items.length}곳`);
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps -- 지역·활성 버전·준비 상태가 바뀔 때만 판정한다
  }, [demoModelReady, currentRegion.id, factorModel?.active_version]);

  // ①·④가 현재 입력 기준으로 끝난 데모 결과는 지역별로 보존한다(수동 실행 포함).
  useEffect(() => {
    if (!demo || !factorModel || analysisStatus !== "done" || !factor.result || !recommendation || recoSig !== recoInputSig) return;
    updateSimulatorRun(currentRegion.id, { version: factorModel.active_version, attempted: true,
      result: factor.result, recommendation, recoSig,
      inputs: { controlValues, budgetTotal, welfareWeight, industryWeight, housingWeight } });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [demo, factor.result, recommendation, recoSig]);

  const handleRunAnalysis = async () => {
    if (factorRetry) { factor.reload(); return; }
    // 실행할 수 없으면(버튼 aria-disabled + 사유 title) 클릭을 무시한다.
    if (analysisStatus === "running" || factorLockTitle) return;
    setOpenStages((s) => ({ ...s, "stage-factor": true }));
    if (demo) {
      // 데모 ③·④는 ① 결과 뒤에 열린다(기준선 흐름) — 다시 실행하면 이전 추천을 비운다.
      setRecommendation(null);
      setRecoSig("");
    }
    addConsoleLog(`INFO: ${currentRegion.name} 요인분석 실행`);
    const analysis = !demo && auto
      ? await factor.runAuto({ onPick: (model) => {
        setModelPick({ context: [mode, model.region_id, model.target].join("|"), id: model.id, version: model.active_version, explicit: true });
        setTarget(model.target);
      } })
      : await factor.run();
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

  const handleRecommend = () => {
    if (demo) {
      const derived = deriveFacilities(currentRegion);
      setRecommendation(derived);
      setRecoSig(recoInputSig);
      setRecoPick(null);
      addConsoleLog(`INFO: ${currentRegion.name} 시설 활용 추천 도출 완료 — 시설 ${derived.items.length}곳`);
    } else {
      addConsoleLog(`INFO: ${currentRegion.name} 시설 대장이 없어 추천 사업을 만들지 않았습니다. 시설 대장을 등록하세요.`);
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
          label: demo ? "자연 감소 예측 추이" : "기준 입력 예측",
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
          label: "인구감소 지역",
          data: (appData.benchmark_regions ?? []).map((r) => ({ x: r.riskIndex, y: r.population, name: r.name })),
          backgroundColor: "rgba(148, 163, 184, 0.55)",
          pointRadius: 5,
          pointHoverRadius: 8
        },
        {
          label: "시범 지자체",
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
        title: { display: true, text: "인구소멸 위험지수", color: ct.tick },
        grid: { color: ct.grid },
        ticks: { color: ct.tick }
      },
      y: {
        title: { display: true, text: "인구수", color: ct.tick },
        grid: { color: ct.grid },
        ticks: { color: ct.tick }
      }
    },
    plugins: {
      // 점 색의 뜻: 회색 = 인구감소 지역, 파랑 = 선택 지자체(두 번째 계열은 점마다 색이 달라 범례 색을 직접 정한다).
      legend: {
        display: true,
        position: "bottom",
        labels: {
          color: ct.tick, boxWidth: 8, font: { size: 11 }, usePointStyle: true,
          // 두 번째 계열은 시범 지자체(주황)와 선택 지자체(파랑)가 섞여 있어 범례는 점 색 세 개를 직접 그린다.
          generateLabels: (chart) => [
            ["인구감소 지역", "rgba(148, 163, 184, 0.8)", 0],
            ["시범 지자체", "rgba(245, 158, 11, 0.8)", 1],
            ["선택 지자체", "rgba(59, 130, 246, 1)", 1]
          ].map(([text, color, datasetIndex]) => ({ text, fillStyle: color, strokeStyle: color, pointStyle: "circle", datasetIndex, hidden: false, fontColor: ct.tick })),
        },
        onClick: () => {}
      },
      tooltip: {
        callbacks: {
          label: (cx) => `${cx.raw.name}: 위험지수 ${cx.raw.x}, 인구 ${cx.raw.y.toLocaleString()}명`
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

  // ④ 추천 행·고른 사업 — 데모만 행이 있다. 실데이터는 시설 대장이 없어 같은 자리에 행동 안내.
  const recoItems = demo && recoReady ? recommendation.items : [];
  const recoSelected = recoItems.slice(0, CHART_ITEMS).find((it) => it.id === recoPick) ?? recoItems[0] ?? null;
  const recoGuide = !demo ? GUIDE.registry : GUIDE.derive;

  // 요인분석 모델이 있는 지역·대상으로 옮긴다(모델 선택·빈 상태 [이동] 공용). 진입 때 자동으로 옮기지는 않는다.
  const moveToModel = (model) => {
    const region = appData.regions.find((row) => row.id === model?.region_id);
    if (!model || !region) return;
    setModelPick({ context: [mode, model.region_id, model.target].join("|"), id: model.id, version: model.active_version, explicit: true });
    setDongPick({ regionId: region.id, code: "" });
    if (!demo) { setTarget(model.target); setAuto(false); }
    setCurrentRegion(region);
    addConsoleLog(`INFO: 요인분석 모델이 있는 ${region.name}(으)로 이동했습니다.`);
  };
  const movableModel = factorModels.find((row) => row.target === factorTarget) ?? factorModels[0];

  // ── 실데이터 표시의 같은 자리 내용 ──
  const guide = (text, action = null) => (
    <div className="real-flow-guide" data-guide>
      <p>{text}</p>
      {action}
    </div>
  );
  // 선택 지역에 요인분석 활성 모델이 없을 때 — 안내 아래에 모델이 있는 지역으로 옮기는 버튼(있을 때만).
  const moveAction = movableModel ? (
    <button type="button" className="btn btn-secondary" style={{ padding: "5px 12px", fontSize: 12 }} onClick={() => moveToModel(movableModel)}>
      <i className="fa-solid fa-location-arrow" aria-hidden="true"></i> 모델이 있는 지역으로 이동
    </button>
  ) : null;
  // ③ 자리의 시뮬레이션 모델 대기 문구 — 모델 없음(전문)·조회 중·조회 실패를 구분한다.
  const realPending = <PendingData running={real.loading} text={real.missing ? REAL_GUIDE_TEXT : realLockTitle} />;
  // ①·② 요인분석 — 두 모드 같은 매핑(factorAnalysisView). 실데이터 모델 없음이면 안내(전문은 ① 첫 카드 한 곳, 나머지는 한 줄).
  const factorLabel = (feature) => observedFeatureLabel(feature, factorTarget, factorResult?.analysis.dong_options ?? dongOptions);
  const factorView = factorResult ? factorAnalysisView(factorResult.analysis, { demo, dataset: factorResult.dataset, label: factorLabel }) : null;
  // 지표별 상태 — 자동 실행일 때만. 데모는 대상이 하나(생활인구)라 자동 = 그 지표 하나의 변화율.
  // 데모 ② 머리 칸·지표별: 생활인구 공표 통계(정책 추천 전망과 같은 API)의 전년 같은 분기 대비. 못 받으면 null → "전년 비교 불가".
  const demoYoy = demo ? quarterYoyChange(demoTrends.population?.points) : null;
  const diagHead = demo ? { change: demoYoy, basis: "전년 같은 분기 대비" } : {};
  const autoRanking = !auto || !factorView ? null : demo ? [{ target: DEMO_FACTOR_TARGET, change: demoYoy }] : factor.ranking;
  // ① 실행 바 아래 안내 — 활성 요인분석 모델이 없을 때만(모델이 있는 지역으로 이동 버튼 포함). 결과 자리 안내는 ②가 맡는다.
  const factorNotice = factorMissing ? guide(FACTOR_GUIDE_TEXT, moveAction) : null;
  const factorRunInfo = factorView ? factorView.runInfo
    : demo ? "인구 구조·시설 지표 9개 요인 · 학습 대상 생활인구" : `등록 학습데이터 기준 · 학습 대상 ${targetName}`;
  // ③ 설정 띠 다섯 칸 — 두 모드 같은 형식(값·표지·칩), 값만 다르다.
  const frame = simFrameView({ demo, sim, budgetTotal, controls, target, context: real.ready ? real.context : null, dongName });
  const cardSource = demo ? "mock" : "api";

  const conditions = (
    <div className="real-flow-controls" data-slot="sim.factor.conditions">
      <label>
        학습 대상
        <select aria-label="학습 대상" value={auto ? AUTO_TARGET : demo ? DEMO_FACTOR_TARGET : target} disabled={analysisRunning} onChange={(e) => {
          // 두 모드 같은 선택지 체계: 첫 항목 "자동(가장 나빠진 지표)"(기본) + 학습 대상. 데모는 대상이 생활인구 하나다.
          const isAuto = e.target.value === AUTO_TARGET;
          setAuto(isAuto);
          if (demo) return;
          setDong("");
          if (!isAuto) setTarget(e.target.value);
        }}>
          <option value={AUTO_TARGET}>{AUTO_TARGET_LABEL}</option>
          {demo ? <option value={DEMO_FACTOR_TARGET}>생활인구</option>
            : Object.entries(REAL_TARGETS).map(([id, name]) => <option key={id} value={id}>{name}</option>)}
        </select>
      </label>
      <label>
        모델 선택
        <select
          aria-label="요인분석 모델"
          value={factorModel?.id ?? ""}
          disabled={analysisRunning || factor.loading || !factorModels.length}
          title={factor.loadFailed ? "모델 목록 조회 실패: 요인분석 실행을 눌러 다시 확인하세요." : factorLockTitle ?? undefined}
          onChange={(event) => moveToModel(factorModels.find((row) => row.id === event.target.value))}
        >
          {!factorModel && <option value="">{factor.loading ? "모델 확인 중…" : factor.loadFailed ? "모델 목록 조회 실패" : "이 지역·대상의 활성 모델 없음"}</option>}
          {/* 내부 등록 이름 대신 사람이 읽는 이름(종류 · 지역 · 학습 대상, 오케스트레이터와 같은 규칙) */}
          {factorModels.map((row) => <option key={row.id} value={row.id}>{instanceLabel(row, regionNameOf)}</option>)}
        </select>
      </label>
      <label>
        요인 범위
        <select aria-label="요인 범위" value={factorScope} onChange={(e) => setFactorScope(e.target.value)}>
          {FACTOR_SCOPES.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
        </select>
      </label>
      <label>
        분석 기간
        <span className="real-flow-period">
          <input type="month" aria-label="분석 시작 달" value={monthInputValue(periodRange?.from)} min={monthInputValue(periodLimits?.from)}
            max={monthInputValue(periodRange?.to)} disabled={analysisRunning || !periodRange}
            onChange={(e) => { const from = parseMonthInput(e.target.value); if (from) setPeriodPick({ from, to: periodRange.to }); }} />
          <span aria-hidden="true">~</span>
          <input type="month" aria-label="분석 끝 달" value={monthInputValue(periodRange?.to)} min={monthInputValue(periodRange?.from)}
            max={monthInputValue(periodLimits?.to)} disabled={analysisRunning || !periodRange}
            onChange={(e) => { const to = parseMonthInput(e.target.value); if (to) setPeriodPick({ from: periodRange.from, to }); }} />
        </span>
      </label>
      <label>
        비교 기준
        <select aria-label="비교 기준" value={baseline} disabled={analysisRunning} onChange={(e) => setBaseline(e.target.value)}>
          {PERIOD_BASELINES.map((b) => <option key={b.id} value={b.id}>{b.label}</option>)}
        </select>
      </label>
      {!demo && real.loading && <p role="status">모델과 관측 입력 확인 중…</p>}
      {!demo && real.feedback && <p className="async-feedback is-error" role="alert">{real.feedback}</p>}
      {factor.feedback && <p className="async-feedback is-error" role="alert">{factor.feedback}</p>}
    </div>
  );

  const regionInfo = (r) => {
    if (!demo) return { chip: null, meta: regionRealStatus(real.catalog, r.id, target) };
    const g = riskGrade(r.riskIndex);
    return {
      chip: { label: g.label, style: { color: g.color, backgroundColor: `rgba(${g.rgb}, 0.12)`, borderColor: `rgba(${g.rgb}, 0.3)` } },
      meta: `${r.theme} · ${r.population.toLocaleString()}명 · 위험지수 ${r.riskIndex}`
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
        versionTitle={factorView?.model.versionId}
        notice={factorNotice}
        conditions={conditions}
        runLockTitle={factorLockTitle}
        runTitle={factorRetry ? "다시 확인" : null}
      />

      {/* ── STAGE ② 요인분석 결과 ── */}
      <FactorResultStage
        open={openStages["stage-result"]}
        onToggle={() => toggleStage("stage-result")}
        locked={analysisStatus !== "done"}
        running={analysisRunning}
        view={factorView}
        scope={factorScope}
        period={factorResult?.period ?? null}
        label={factorLabel}
        evidence={<ModelEvidence slot="sim.factor.evidence" result={factorResult?.analysis} regions={regions} label={factorLabel}
          datasetName={() => factorResult?.dataset?.name ?? factorResult?.analysis.provenance.dataset_id}
          emptyText={factorMissing ? FACTOR_PENDING_TEXT : "[요인분석 실행] 후 모델·학습데이터 근거가 표시됩니다."} />}
        diagnosis={diagnosisView(factorView, autoRanking, demo ? currentRegion.case.problemDiagnosis : [], diagHead)}
        diagnosisSource={cardSource}
        pendingText={factorMissing ? FACTOR_PENDING_TEXT : null}
      />

      {/* ── STAGE ③ 시뮬레이션: 공간정보 탐색 + 강화학습 변수 + 예측 ── */}
      <CollapsibleStage
        id="stage-sim"
        no="STAGE ③"
        title="시뮬레이션"
        open={openStages["stage-sim"]}
        onToggle={() => toggleStage("stage-sim")}
      >
        {/* 강화학습 변수 프레임 */}
        <div className="pl-sim-frame" data-slot="sim.sim.frame">
          <SimFrameCell kind="obj" tag="목적함수" cell={frame.objective}
            status={analysisDone ? "② 진단 반영" : null} />
          <SimFrameCell tag="수요" cell={frame.demand} />
          <SimFrameCell tag="공급" cell={frame.supply} />
          <SimFrameCell kind="adjust" tag="조절변수" cell={frame.adjust} />
          <SimFrameCell kind="constraint" tag="제약요소" cell={frame.constraint} />
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
            overlayNote={null}
            resizeKey={`${openStages["stage-sim"]}|${analysisStatus}`}
            reco={{ items: recoItems.slice(0, CHART_ITEMS), selected: recoSelected, onPick: setRecoPick }}
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
              budgetTotal, setBudgetTotal: edited(setBudgetTotal), welfare: welfareWeight, setWelfare: edited(setWelfareWeight),
              industry: industryWeight, setIndustry: edited(setIndustryWeight), housing: housingWeight, setHousing: edited(setHousingWeight),
              lockText: demo ? null : REAL_CONTROLS_LOCK_TEXT
            }}
            controls={controls}
            controlValues={controlValues}
            onControlChange={edited((key, value) => setControlValues((prev) => ({ ...prev, [key]: value })))}
            controlsLabel={demo ? "시설 조절변수" : "관측 입력 조정"}
            controlsEmptyText={demo ? null : real.missing ? "시뮬레이션 모델 준비 후 조정할 수 있습니다." : real.loading ? "관측 입력 확인 중…" : null}
            controlsLockTitle={realLockTitle ?? "조정할 조절변수가 없습니다."}
            onResetControls={edited(() => { if (controls.length) setControlValues(defaultValues()); })}
            onRecommend={handleRecommend}
            // 실데이터는 ① 실행 없이 시뮬레이션 모델만으로 추천한다. 조회 실패면 눌러서 다시 확인(재시도).
            recommendDisabled={demo && !analysisDone}
            notReadyText={demo && !analysisDone ? "STAGE ① 요인분석 실행 후 추천을 도출할 수 있습니다." : null}
            stale={recoStale}
            controlNote={controlNote}
          />
        </div>

        <SimulatorMapDetails
          appData={appData}
          currentRegion={currentRegion}
          allowSeed={demo}
          note={realUnavailableText("시설물·격자 정보는")}
        />

        {/* ── 분석: 예측 + 전 지자체 분포 — 카드마다 한 줄(결정 36). 나란히 두면 분포 카드 아래가 비어 세로로 쌓는다. ── */}
        <div className="sim-result-stack">
          <Card
            title={demo ? "10개년 인구 예측 시뮬레이션 결과" : `다음 달 ${targetName} 예측 결과`}
            icon="fa-wand-magic-sparkles"
            dataSource={cardSource}
            data-slot="sim.sim.forecast"
          >
            {demo && resultsReady ? (
              <>
                <div style={{ position: "relative", height: 200, width: "100%" }}>
                  <Line data={chartData} options={chartOpts} />
                </div>
                <div style={{ display: "flex", justifyContent: "space-around", marginTop: 16, textAlign: "center" }}>
                  <div>
                    <div style={{ fontSize: 11, color: "var(--text-secondary)" }}>10년 후 인구</div>
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
                <p className="chart-summary sr-only">
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
            <ForecastCompare view={compareView} loading={compare.loading} />
            {/* 계산 근거 — 예측 입력 기여도·검증 오차·기준선·활성 버전(이전 STAGE ①·② 실데이터 내용) */}
            <ModelEvidence slot="sim.sim.evidence" demo={demo} state={evidenceState} result={simShown ?? real.context} regions={regions} datasetName={real.datasetName} label={label} />
          </Card>

          <Card title="전 지자체 위험지수 vs 인구 분포" icon="fa-braille" dataSource={cardSource} data-slot="sim.sim.distribution">
            {demo && resultsReady ? (
              <>
                <p className="sr-only">
                  전국 인구감소 지역(회색 점) 분포 속 시범 분석 대상 지자체의 상대 위치. 파란 점이 현재 선택
                  지자체입니다.
                </p>
                <div style={{ position: "relative", height: 220, width: "100%" }}>
                  <Scatter data={scatterData} options={scatterOpts} />
                </div>
                <p className="chart-summary sr-only">
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
          // 실데이터: 같은 열 자리에 시뮬레이션 모델의 다음 달 예측·기준 대비 변화율(③ 예측 카드와 같은 계산)
          forecast={demo ? null : {
            ready: real.ready, key: contextKey, unit, current: simShown,
            predict: (values) => real.predictOnly(Object.fromEntries(controls.map((c) => [c.key, Number(values?.[c.key] ?? c.default)])))
          }}
          onApply={(sc) => {
            userEdited.current = currentRegion.id;
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

      {/* ── STAGE ④ 추천: 구역 4개(추천 결과·정책 효과 전망·세부 비교·보고서로 보내기), 두 모드 같은 구조 ── */}
      <PolicyRecommendation
        demo={demo}
        region={currentRegion}
        items={recoItems}
        selected={recoSelected}
        onPick={setRecoPick}
        inputs={{ budgetTotal, welfare: welfareWeight, industry: industryWeight, housing: housingWeight,
          period: stayStat ? `${stayStat.period_from}~${stayStat.period_to}` : null }}
        guide={recoGuide}
        onSend={(item) => {
          sendToReport(reportHandoffOf(item, currentRegion, { budgetTotal }));
          addConsoleLog(`INFO: ${currentRegion.name} 추천 사업을 보고서로 보냈습니다.`);
          navigateToTab("tab-reporter");
        }}
        open={openStages["stage-report"]}
        onToggle={() => toggleStage("stage-report")}
        subTip={demo ? currentRegion.case.reportFocus : null}
        action={
          <button type="button" className="btn btn-primary"
            onClick={() => { if (!demo || analysisDone) handleRecommend(); }}
            aria-disabled={(demo && !analysisDone) || undefined}
            title={demo && !analysisDone ? "요인분석을 먼저 실행하세요." : undefined}>
            <i className="fa-solid fa-lightbulb" aria-hidden="true"></i> 정책 추천 도출
          </button>
        }
      />
    </>
  );
}
