import { TARGET_LABELS, displayId } from "./modelops.js";

export const REAL_TARGETS = { observed_sales_krw: "관측 소비", nonlocal_visitors: "타지역 방문객" };
// 데모 요인분석의 학습 대상(합성 요인 패널) — 실데이터 학습 대상 목록과 별개다.
export const DEMO_FACTOR_TARGET = "living_population";
export const DEMO_FACTOR_SOURCE_PREFIX = "mock_factor_";
// 요인 패널 변수 이름·단위(백엔드 factor_panel.FEATURES 키와 같다).
const FACTOR_PANEL = {
  youth_ratio: ["유소년 비율", "%"], aging_index: ["노령화지수", "지수"], working_age_pop: ["생산연령인구", "명"],
  elderly_pop: ["고령인구", "명"], death_cancellations: ["사망말소", "명/월"], edu_facility_density: ["교육시설 밀도", "개소/만 명"],
  medical_facility_density: ["의료시설 밀도", "개소/만 명"], child_welfare_density: ["아동복지시설 밀도", "개소/만 명"],
  facility_capacity: ["시설 수용력(인구 대비)", "정원/천 명"]
};
export const FACTOR_PANEL_FEATURES = Object.keys(FACTOR_PANEL);
export const targetName = (target) => TARGET_LABELS[target] ?? target;
export function createRequestGate(epochReader) {
  let sequence=0;
  return { cancel:() => { sequence+=1; }, begin:() => {
    const request=++sequence, epoch=epochReader();
    return () => request===sequence && epoch===epochReader();
  } };
}
export function observedFeatureLabel(name, target, dongs = []) {
  const noun = targetName(target);
  const labels = { y_lag1: `전월 ${noun}`, y_lag2: `2개월 전 ${noun}`, y_lag3: `3개월 전 ${noun}`, y_yoy: `전년 같은 달 ${noun}`,
    has_yoy: "전년 관측값 유무", month_sin: "월별 계절성 1", month_cos: "월별 계절성 2" };
  if (name.startsWith("dong_")) return `행정동: ${dongs.find((row) => row.code === name.slice(5))?.name ?? name.slice(5)}`;
  return FACTOR_PANEL[name]?.[0] ?? labels[name] ?? name;
}
/** 입력값 단위 — 요인 패널 변수는 고유 단위, 과거 관측값(y_*)은 학습 대상 단위, 그 밖(행정동·계절)은 없음. */
export const featureUnit = (name, target) => FACTOR_PANEL[name]?.[1] ?? (name.startsWith("y_") ? targetUnit(target) : "");
/** 같은 지역·학습 대상의 활성 요인분석·시뮬레이션 인스턴스. mode: 카탈로그가 속해야 하는 모드(기본 실데이터). */
export function activeModelPair(catalog, region, target, mode = "real") {
  if (catalog.mode !== mode || catalog.source !== mode) throw new Error("모델 목록의 데이터 모드를 확인하세요.");
  const candidates = catalog.instances.filter((row) => row.mode === mode && row.region_id === region && row.target === target);
  return { factor: candidates.find((row) => row.model_type === "factor_analysis" && row.active_version),
    simulation: candidates.find((row) => row.model_type === "simulation" && row.active_version) };
}
/** 선택한 활성 요인 모델은 모드·지역·대상·버전이 모두 일치해야 한다. */
export function selectedFactorModel(catalog, id, region, target, mode) {
  if (catalog?.mode !== mode || catalog?.source !== mode) return undefined;
  return catalog.instances?.find((row) => row.id === id && row.mode === mode && row.region_id === region
    && row.target === target && row.model_type === "factor_analysis" && row.active_version
    && catalog.versions?.some((version) => version.mode === mode && version.id === row.active_version && version.instance_id === row.id));
}
/** 선택이 바뀐 직후에도 이전 조건의 추론 결과를 표시·재사용하지 않는다. */
export function inferenceMatches(result, { mode, region, target, modelId, versionId, dong }) {
  const proof = result?.provenance;
  return Boolean(modelId && versionId && proof?.mode === mode && proof.source === mode
    && proof.region_id === region && proof.target === target
    && proof.instance_id === modelId && proof.version_id === versionId
    && (!dong || proof.dong_code === dong));
}
/** 추천의 계산 조건: 모드·선택 모델과 실제 추론 버전/데이터가 바뀌면 이전 추천은 낡았다. */
export function inferenceContextKey({ mode, region, target, dong, factorModel, simulationProof }) {
  return JSON.stringify([mode, region, target, dong || "", factorModel?.id ?? null, factorModel?.active_version ?? null,
    simulationProof?.instance_id ?? null, simulationProof?.version_id ?? null,
    simulationProof?.dataset_id ?? null, simulationProof?.dataset_hash ?? null]);
}
export function inferenceRequest(context, changes = {}) {
  const p = context.provenance;
  return { region_id: p.region_id, target: p.target, dataset_id: p.dataset_id, dataset_hash: p.dataset_hash,
    dong_code: p.dong_code, adjustments: changes };
}
export function validateInference(result, context) {
  const expected = context.provenance, actual = result?.provenance;
  if (!actual || ["instance_id", "model_type", "version_id", "job_id", "dataset_id", "dataset_hash", "region_id", "target", "dong_code", "prediction_period", "mode", "source"].some((key) => actual[key] !== expected[key])) {
    throw new Error("분석 결과와 선택한 모델의 근거가 다릅니다. 목록을 다시 확인하세요.");
  }
  const values = [result.intercept, result.prediction, result.baseline_prediction, result.delta, ...(result.contributions ?? []).map((row) => row.value)];
  const reconstructed = result.intercept+(result.contributions ?? []).reduce((sum,row) => sum+row.value,0);
  if (!result.inference_id || !result.contributions?.length || !values.every(Number.isFinite) || Math.abs(reconstructed-result.prediction)>1e-9*Math.max(1,Math.abs(result.prediction))) {
    throw new Error("예측 계산 결과를 확인하지 못했습니다.");
  }
  return result;
}
export function rankSensitivity(result) {
  if (result.provenance.model_type !== "simulation") throw new Error("시뮬레이션 결과를 먼저 확인하세요.");
  const byFeature = new Map();
  for (const row of result.recommendations ?? []) {
    if (![row.from,row.to,row.prediction,row.delta].every(Number.isFinite) || row.delta <= 0) continue;
    if (!byFeature.has(row.feature) || row.delta > byFeature.get(row.feature).delta) byFeature.set(row.feature,row);
  }
  return [...byFeature.values()].sort((a,b) => b.delta-a.delta || a.feature.localeCompare(b.feature)).map((row,index) => ({
    ...row, rank:index+1, reference_prediction:result.prediction, inference_id:result.inference_id, provenance:result.provenance
  }));
}

// ── 시뮬레이터 4단계 화면(실데이터 표시)용 문구·매핑 ──
// 실데이터가 없을 때 같은 자리에 두는 행동 안내. G1 = 준비 가능(③·④는 시뮬레이션 모델 기준), G2 = 실데이터 항목 없음.
export const REAL_GUIDE_TEXT = "이 지역·학습 대상의 시뮬레이션 활성 모델이 없습니다. DataOps에서 학습데이터를 등록하고 오케스트레이터에서 시뮬레이션 모델을 학습·활성화하세요.";
// subject 는 조사까지 포함한다(예: "문제 유형 진단은", "지자체 비교 분포는").
// 행동 안내만 둔다 — 제공 여부·사유는 쓰지 않는다.
export const realUnavailableText = (subject) => `${subject} 설정에서 '데모 데이터 표시'를 켜면 예시로 볼 수 있습니다.`;
export const REAL_CONTROLS_LOCK_TEXT = "예산·정책 강조 가중치는 데모 데이터 시뮬레이션에서 조정합니다. 실데이터에서는 아래 관측 입력을 조정하세요.";
// STAGE ①·② 요인분석 — 두 모드가 활성 요인분석 모델 결과를 같은 자리에 보인다.
// 실데이터에 요인분석 활성 모델이 없을 때만 안내(전문은 ① 첫 카드 한 곳, 나머지는 한 줄).
export const FACTOR_GUIDE_TEXT = "이 지역·학습 대상의 요인분석 활성 모델이 없습니다. DataOps에서 학습데이터를 등록하고 오케스트레이터에서 요인분석 모델을 학습·활성화하세요.";
export const FACTOR_PENDING_TEXT = "오케스트레이터에서 요인분석 모델을 학습·활성화하세요.";
export const FACTOR_RUN_LOCK_TEXT = "이 지역·학습 대상의 요인분석 활성 모델을 먼저 준비하세요.";
export const FACTOR_IDLE_TEXT = "[요인분석 실행] 후 표시됩니다.";
export const FACTOR_FAILED_TEXT = "요인분석을 실행하지 못했습니다. 잠시 뒤 다시 실행하세요.";
export const FACTOR_VERSION_CHANGED_TEXT = "활성 버전이 바뀌었습니다. [요인분석 실행]을 다시 누르세요.";
export const FACTOR_EXPLAIN_TEXT = "기여도는 학습 기간 평균 대비 최근 관측값이 모델 추정을 높인(+)·낮춘(−) 크기입니다.";
// 알림(role=alert) 고정 문구 — 서버·브라우저 원문은 활동 로그에만 남긴다.
export const REAL_LOAD_FAILED_TEXT = "모델 정보를 불러오지 못했습니다. 잠시 뒤 다시 시도하세요.";
export const REAL_PREDICT_FAILED_TEXT = "예측을 계산하지 못했습니다. 입력을 다시 조정하거나 잠시 뒤 다시 시도하세요.";
export const REAL_VERSION_CHANGED_TEXT = "활성 버전이 바뀌었습니다. 새 활성 버전의 관측 입력으로 예측을 다시 계산합니다.";

export const targetUnit = (target) => ({ observed_sales_krw: "원", population_change_pct: "%" })[target] ?? "명";
// 단위별 반올림 — 인원·금액·개수·점수(명·원·억·개소·호·명/월·점)는 정수(천 단위 구분), 비율·지수·밀도는 소수 1~2자리, 단위 없음은 최대 2자리.
// 시뮬레이터 ①~④·오케스트레이터가 모두 이 함수로 수치를 보인다.
const COUNT_UNITS = new Set(["명", "원", "억", "개소", "호", "명/월", "점"]);
const integerText = (value) => (Math.round(value) || 0).toLocaleString("ko-KR");
export function formatUnitValue(value, unit) {
  if (!Number.isFinite(value)) return "—";
  if (COUNT_UNITS.has(unit)) return integerText(value);
  return value.toLocaleString("ko-KR", unit ? { minimumFractionDigits: 1, maximumFractionDigits: 2 } : { maximumFractionDigits: 2 });
}
/** 큰 수 축약 — 1억 이상 "1.03억", 1만 이상 "71.2만", 그보다 작으면 천 단위 구분(최대 소수 2자리). */
export function compactNumber(value) {
  if (!Number.isFinite(value)) return "—";
  const abs = Math.abs(value);
  const text = (number, digits) => number.toLocaleString("ko-KR", { maximumFractionDigits: digits });
  if (abs >= 1e8) return `${text(value / 1e8, 2)}억`;
  if (abs >= 1e4) return `${text(value / 1e4, 1)}만`;
  return text(value, 2);
}
/** 구간 라벨("712,425–613,218,650") 안의 큰 수만 축약한다 — 부호·구분 기호는 그대로 둔다. */
export const compactRangeLabel = (label) =>
  String(label).replace(/\d[\d,]*(?:\.\d+)?/g, (token) => {
    const number = Number(token.replace(/,/g, ""));
    return number >= 1e4 ? compactNumber(number) : token;
  });
// 슬라이더 값 — 한 칸이 1 이상이면 정수 천 단위(데모 '50%'·'18㏊' 그대로), 1 미만이면 단위 규칙.
export const sliderValueText = (value, unit, step = 1) =>
  step >= 1 && Number.isFinite(value) ? integerText(value) : formatUnitValue(value, unit);
export const formatMonth = (value) => value ? `${String(value).slice(0, 4)}-${String(value).slice(4)}` : "—";

/** 예측 입력 기여도 → |값| 내림차순 목록(부호는 direction). */
export function contributionFactors(result, label) {
  return [...(result?.contributions ?? [])]
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .map((row) => ({ feature: row.feature, label: label(row.feature), direction: row.value < 0 ? "-" : "+", input: row.input, value: row.value }));
}

/** 예측을 높인 입력 / 낮춘 입력(|값| 내림차순 행). 기여 0은 어느 쪽에도 넣지 않는다. */
export function contributionSplit(result, label) {
  const rows = contributionFactors(result, label);
  return { positive: rows.filter((row) => row.value > 0), negative: rows.filter((row) => row.value < 0) };
}

// 학습 시 비교한 기준선 방법 — 알려진 것만 한글로 보인다(모르는 이름은 표시하지 않는다).
const BASELINE_NAMES = { yoy_or_prev_month: "전년 같은 달·전월 관측값", same_month_last_year: "전년 같은 달 관측값", leave_one_out_mean: "검증 시점을 뺀 관측 평균" };
export const baselineName = (name) => BASELINE_NAMES[name];
export const FACTOR_LIST_LIMIT = 10;

/**
 * 요인분석 추론 결과 → STAGE ①(카드 3개)·②(상관관계) 표시 데이터. 두 모드가 같은 함수를 쓰고,
 * 모드는 출처 문구(basis)만 바꾼다. dataset: 결과의 등록 학습데이터(카탈로그 행, 없으면 ID로 표기).
 */
export function factorAnalysisView(result, { demo, dataset, label }) {
  const p = result.provenance;
  const unit = targetUnit(p.target);
  const all = contributionFactors(result, label).map((row) => ({ ...row, inputUnit: featureUnit(row.feature, p.target) }));
  const factors = all.slice(0, FACTOR_LIST_LIMIT);
  const name = dataset?.name ?? p.dataset_id;
  const observed = `${formatMonth(p.observed_from)}~${formatMonth(p.observed_to)}`;
  const evaluation = p.evaluation_period;
  return {
    // rest: 10행 밖 요인 — ①-1 카드 안 "모두 보기" 펼침에서 같은 형식으로 보인다.
    unit, factors, rest: all.slice(FACTOR_LIST_LIMIT), more: all.length - factors.length,
    split: { positive: factors.filter((row) => row.value > 0), negative: factors.filter((row) => row.value < 0) },
    basis: demo ? `연계 데이터: 데모 요인 패널 · ${name}` : `현재 등록 데이터 기준 · ${name} (관측 ${observed})`,
    runInfo: `학습데이터 ${name} · 요인분석 활성 버전 ${displayId(p.version_id)}`,
    // listKey: 요인 목록 쪽 번호를 1쪽으로 되돌리는 기준 — 같은 버전이라도 행정동이 바뀌면 다른 목록이다.
    listKey: `${p.version_id}|${p.dong_code ?? ""}`,
    model: { version: displayId(p.version_id), versionId: p.version_id, intercept: result.intercept, prediction: result.prediction,
      features: (p.features ?? []).map(label) },
    validation: {
      mae: p.metrics?.mae, baselineMae: p.baseline?.mae, baselineName: baselineName(p.baseline?.name),
      targets: [`${targetName(p.target)} · 요인 ${(p.features ?? []).length}개`, `관측 기간 ${observed}`,
        `검증 기간 ${evaluation ? `${formatMonth(evaluation.from)}~${formatMonth(evaluation.to)}` : "—"}`,
        `기준월 ${formatMonth(p.prediction_period)}`]
    }
  };
}

// 문제 유형 판정 폭 — 다음 달 추정이 전년 같은 달보다 이 비율(%) 넘게 낮으면 감소형, 높으면 증가형, 그 사이는 정체형.
const DIAGNOSIS_TREND_PCT = 5;
// 계절 영향 표시 기준 — 월별 계절성 두 항의 |기여도| 합이 전체 |기여도| 합에서 차지하는 비율.
const DIAGNOSIS_SEASONAL_SHARE = 0.15;

const trendKind = (change) => (change < -DIAGNOSIS_TREND_PCT ? "감소형" : change > DIAGNOSIS_TREND_PCT ? "증가형" : "정체형");

/** 다음 달 추정이 전년 같은 달 관측보다 몇 % 변하는지(요인분석 view 기준). 전년 같은 달 값이 없으면 null. */
export function yoyChange(view) {
  const all = [...view.factors, ...view.rest];
  const input = (feature) => all.find((row) => row.feature === feature)?.input;
  const yoy = input("y_yoy");
  const prediction = view.model.prediction;
  if (Number.isFinite(yoy) && yoy > 0 && input("has_yoy") !== 0 && Number.isFinite(prediction)) return ((prediction - yoy) / yoy) * 100;
  return null;
}

export const AUTO_TARGET = "auto";
export const AUTO_TARGET_LABEL = "자동(가장 나빠진 지표)";

/** 학습 대상별 변화율 → 가장 많이 줄어든 순(전년 비교 불가는 맨 뒤, 같으면 입력 순서). */
export function rankTargets(entries) {
  const rank = (e) => (Number.isFinite(e.change) ? e.change : Infinity);
  return entries.map((e, i) => [e, i]).sort(([a, i], [b, j]) => rank(a) - rank(b) || i - j).map(([e]) => e);
}


// 범위/100을 유효숫자 2자리로 — 슬라이더 한 칸.
function sliderStep(span) {
  if (!(span > 0)) return 1;
  const raw = span / 100;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  return Number((Math.round(raw / magnitude * 10) / 10 * magnitude).toPrecision(2));
}

/** 시뮬레이션 활성 모델의 조정 가능한 관측 입력 → 슬라이더 정의(시설 조절변수와 같은 모양). */
export function realSliderControls(context, label, unit) {
  return (context?.controls ?? []).map((row) => {
    const span = row.max - row.min;
    return { key: row.feature, label: label(row.feature), min: row.min, max: row.max, default: row.default ?? context.inputs?.[row.feature],
      step: sliderStep(span), unit, exactValue: true, locked: !(span > 0) };
  });
}

/** 다음 달 예측 선 차트: 관측 3점(3개월 전·2개월 전·전월) + 예측월. */
export function forecastSeries(context, simulation) {
  const base = simulation?.baseline_inputs ?? context?.inputs ?? {};
  const observed = [base.y_lag3, base.y_lag2, base.y_lag1];
  const period = formatMonth(simulation?.provenance?.prediction_period ?? context?.provenance?.prediction_period);
  const baseline = simulation?.baseline_prediction ?? null;
  const adjusted = simulation?.prediction ?? null;
  return { labels: ["3개월 전", "2개월 전", "전월", `예측월 ${period}`],
    baseline: [...observed, baseline], adjusted: [...observed, adjusted] };
}

/** 민감도 순위 → 정책 추천 패널(RICE 패널과 같은 형식, 점수 대신 모델 예측상 변화). */
export function sensitivityRecommendation(ranking, label, unit, targetName) {
  const items = ranking.map((row) => ({
    id: row.feature, name: `${label(row.feature)} ${row.to > row.from ? "증가" : "감소"}`, category: "관측 입력 가정",
    score: row.delta, scoreText: `+${formatUnitValue(row.delta, unit)}`,
    cells: [`${formatUnitValue(row.from, unit)} → ${formatUnitValue(row.to, unit)}${unit}`, `${formatUnitValue(row.delta, unit)}${unit}`]
  }));
  return {
    items,
    scoreLabel: "모델 예측상 변화",
    scoreUnit: unit,
    columns: [{ label: "관측 지표 입력 가정" }, { label: "입력 변화", numeric: true }, { label: "모델 예측상 변화", numeric: true }],
    caption: "시뮬레이션 예측 변화량에 근거한 관측 지표 정책 추천 순위",
    infoTip: `각 관측 입력을 선택 행정동 학습 범위의 1%만큼 늘리거나 줄인 단일 변경 예측 중, ${targetName} 증가가 큰 방향을 지표별로 고릅니다.`
  };
}

/** 지역 목록 메타 — 같은 지역·학습 대상의 활성 시뮬레이션 모델이 있으면 준비됨(③·④는 요인분석 모델과 무관). */
export function regionRealStatus(catalog, regionId, target) {
  const ready = (catalog?.instances ?? []).some((row) => row.mode === "real" && row.region_id === regionId && row.target === target &&
    row.model_type === "simulation" && row.active_version);
  return ready ? "실데이터 시뮬레이션 모델 준비됨" : "실데이터 모델 없음";
}

// ── ② 문제 유형 진단 카드 표시 형식 ──

/**
 * 생활인구 공표 통계(분기, 생활인구 = 등록 + 체류)의 가장 최근 분기 대 전년 같은 분기 변화율(%) — 데모 ② 머리 칸.
 * 정책 추천 전망 그래프와 같은 API 값. 전년 같은 분기 값이 없으면 null.
 * @param {{ year: number, quarter: number, registered: number, stay: number }[] | null | undefined} points
 */
export function quarterYoyChange(points) {
  if (!points?.length) return null;
  const latest = points.reduce((a, p) => (p.year * 4 + p.quarter > a.year * 4 + a.quarter ? p : a));
  const prior = points.find((p) => p.year === latest.year - 1 && p.quarter === latest.quarter);
  const living = (p) => p.registered + p.stay;
  if (!prior || !(living(prior) > 0)) return null;
  return ((living(latest) - living(prior)) / living(prior)) * 100;
}
const signedText = (value, unit) => `${value < 0 ? "−" : "+"}${formatUnitValue(Math.abs(value), unit)}${unit}`;
const percentText = (change) => `${change < 0 ? "−" : "+"}${Math.abs(change).toFixed(1)}%`;

/**
 * ② 문제 유형 진단 카드(실데이터) — 유형·변화율(머리), 하방·상방 요인 행, 계절 영향 행, 지표별 상태(자동 실행일 때).
 * 행정동 항목은 요인으로 내세우지 않는다. 읽을 신호가 하나도 없으면 빈 형식(rows 0·kind null).
 * @param {ReturnType<typeof factorAnalysisView> | null} view
 * @param {{ target: string, change: number|null }[] | null} ranking 자동 실행의 지표별 순위
 * @param {string[]} items 지역 점검 항목(데모 사례 정의에만 있음, 실데이터는 빈 배열 → 같은 자리에 "해당 항목 없음")
 * @param {{ change?: number|null, basis?: string }} head 머리 칸 변화율을 모델 대신 다른 근거로 줄 때(데모: 공표 통계 전년 같은 분기 대비)
 * 두 모드가 같은 함수·같은 형식을 쓴다. 변화율을 못 구하면 머리의 변화율 자리에 "전년 비교 불가".
 */
export function diagnosisView(view, ranking = null, items = [], head = {}) {
  if (!view) return null;
  const all = [...view.factors, ...view.rest];
  const change = "change" in head ? head.change : yoyChange(view);
  const drivers = all.filter((row) => !row.feature.startsWith("dong_")); // |기여도| 내림차순 유지
  const down = drivers.find((row) => row.value < 0);
  const up = drivers.find((row) => row.value > 0);
  const rows = [
    down && { dir: "down", name: "하방 요인", label: down.label, value: signedText(down.value, view.unit) },
    up && { dir: "up", name: "상방 요인", label: up.label, value: signedText(up.value, view.unit) }
  ].filter(Boolean);
  const total = all.reduce((sum, row) => sum + Math.abs(row.value), 0);
  const seasonal = all.filter((row) => row.feature === "month_sin" || row.feature === "month_cos")
    .reduce((sum, row) => sum + Math.abs(row.value), 0);
  if (total > 0 && seasonal / total >= DIAGNOSIS_SEASONAL_SHARE) {
    rows.push({ dir: "season", name: "계절 영향", label: "월별 계절성", value: `${Math.round((seasonal / total) * 100)}%` });
  }
  return {
    kind: change === null ? null : trendKind(change),
    change: change === null ? "전년 비교 불가" : percentText(change),
    basis: head.basis ?? "전년 같은 달 대비",
    rows,
    items: [...items],
    indicators: (ranking ?? []).map((e) => ({
      target: e.target,
      name: REAL_TARGETS[e.target] ?? targetName(e.target),
      kind: Number.isFinite(e.change) ? trendKind(e.change) : null,
      change: Number.isFinite(e.change) ? percentText(e.change) : "전년 비교 불가"
    }))
  };
}

// ── ③ 시뮬레이션 설정 띠 표시 형식 ──
/** YYYYMM → "2019년 4월" */
export const monthText = (value) => (value ? `${String(value).slice(0, 4)}년 ${Number(String(value).slice(4))}월` : "—");

// 데모 조절변수 중 정책 강조 가중치 슬라이더(SimulatorDock) — 시설 조절변수 앞에 둔다.
const DEMO_WEIGHT_CONTROLS = ["복지 예산", "산업 일자리", "주거·주택"];

/**
 * ③ 시뮬레이션 설정 띠 — 다섯 칸(목적함수·수요·공급·조절변수·제약요소)의 값. 두 모드가 같은 형식({ value, meta, chips })이다.
 * 데모: 지역 사례의 시뮬레이션 정의와 현재 총 예산. 실데이터: 활성 시뮬레이션 모델의 근거(예측월·전월 관측·조정 입력·학습 구간).
 * 실데이터 모델에는 공급 입력이 없어 공급 칸은 "해당 없음"이다.
 */
export function simFrameView({ demo, sim = null, budgetTotal = null, controls = [], target = null, context = null, dongName = "" }) {
  if (demo) {
    return {
      // 사례 정의의 "방식 — 목표" 가운데 목표만 보인다(방식은 화면 문구 규칙상 싣지 않음).
      objective: { value: String(sim.objective).split(" — ").at(-1), meta: null },
      demand: { value: sim.factors.demand, meta: null },
      supply: { value: sim.factors.supply, meta: null },
      adjust: { chips: [...DEMO_WEIGHT_CONTROLS, ...controls.map((c) => c.label)], meta: null },
      constraint: { value: `${Number(budgetTotal).toLocaleString("ko-KR")}억 원`, meta: `총 ${sim.constraint}` }
    };
  }
  const proof = context?.provenance;
  if (!proof) {
    const empty = { value: "–", meta: null };
    return { objective: empty, demand: empty, supply: empty, adjust: { chips: [], meta: null }, constraint: empty };
  }
  const noun = targetName(target);
  const unit = targetUnit(target);
  const lastObserved = context.inputs?.y_lag1;
  const range = proof.range_provenance;
  return {
    objective: { value: `${noun} 증가`, meta: `예측월 ${monthText(proof.prediction_period)}` },
    demand: { value: Number.isFinite(lastObserved) ? `${formatUnitValue(lastObserved, unit)}${unit}` : "–", meta: `전월 ${noun} · ${dongName}` },
    supply: { value: "해당 없음", meta: null },
    // 조정 입력 이름에서 반복되는 학습 대상 이름은 칸 표지(meta)로 한 번만 둔다.
    adjust: { chips: controls.map((c) => c.label.replace(noun, "").trim() || c.label), meta: noun },
    constraint: { value: `${monthText(range?.from)}~${monthText(range?.to)}`, meta: "입력 조정 범위" }
  };
}
