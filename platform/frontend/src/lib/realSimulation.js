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
export const realUnavailableText = (subject) => `${subject} 실데이터로 아직 제공되지 않습니다. 설정에서 데모 데이터 표시를 켜면 예시를 볼 수 있습니다.`;
export const REAL_CONTROLS_LOCK_TEXT = "예산·정책 강조 가중치는 데모 데이터 시뮬레이션에서 조정합니다. 실데이터에서는 아래 관측 입력을 조정하세요.";
// STAGE ①·② 요인분석 — 두 모드가 활성 요인분석 모델 결과를 같은 자리에 보인다.
// 실데이터에 요인분석 활성 모델이 없을 때만 안내(전문은 ① 첫 카드 한 곳, 나머지는 한 줄).
export const FACTOR_GUIDE_TEXT = "이 지역·학습 대상의 요인분석 활성 모델이 없습니다. DataOps에서 학습데이터를 등록하고 오케스트레이터에서 요인분석 모델을 학습·활성화하세요.";
export const FACTOR_PENDING_TEXT = "요인분석 활성 모델 준비 후 표시";
export const FACTOR_RUN_LOCK_TEXT = "이 지역·학습 대상의 요인분석 활성 모델을 먼저 준비하세요.";
export const FACTOR_IDLE_TEXT = "[요인분석 실행] 후 표시됩니다.";
export const FACTOR_FAILED_TEXT = "요인분석을 실행하지 못했습니다. 잠시 뒤 다시 실행하세요.";
export const FACTOR_VERSION_CHANGED_TEXT = "활성 버전이 바뀌었습니다. [요인분석 실행]을 다시 누르세요.";
export const FACTOR_DIAGNOSIS_TEXT = "문제 유형 진단은 시설 지표 학습데이터 등록 후 제공됩니다.";
export const FACTOR_EXPLAIN_TEXT = "기여도는 학습 기간 평균 대비 최근 관측값이 모델 추정을 높인(+)·낮춘(−) 크기입니다.";
export const FACTOR_FOOTNOTE = "※ 모델 추정의 기여도이며 정책의 인과 효과가 아닙니다.";
export const CORRELATION_BASIS_TEXT = "STAGE ① 요인 기여도의 방향(+ 높임 / − 낮춤) 기준";
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
 * 모드는 출처 문구(basis·sourceLabel)만 바꾼다. dataset: 결과의 등록 학습데이터(카탈로그 행, 없으면 ID로 표기).
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
    sourceLabel: demo ? "데모" : `실데이터 · ${name}`,
    runInfo: `학습데이터 ${name} · 요인분석 활성 버전 ${displayId(p.version_id)}`,
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
      step: sliderStep(span), unit, locked: !(span > 0) };
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
    cells: [`${formatUnitValue(row.from, unit)} → ${formatUnitValue(row.to, unit)}`, formatUnitValue(row.delta, unit)]
  }));
  return {
    items,
    scoreLabel: `모델 예측상 변화 (${unit})`,
    scoreUnit: unit,
    columns: [{ label: "관측 지표 입력 가정" }, { label: `입력 변화 (${unit})`, numeric: true }, { label: `모델 예측상 변화 (${unit})`, numeric: true }],
    caption: "시뮬레이션 예측 변화량에 근거한 관측 지표 정책 추천 순위",
    infoTip: `각 관측 입력을 선택 행정동 학습 범위의 1%만큼 늘리거나 줄인 단일 변경 예측 중, ${targetName} 증가가 큰 방향을 지표별로 고릅니다.`,
    footnote: "※ 추천 순위는 모델 예측의 민감도이며, 정책의 실제 인과 효과를 검증한 결과가 아닙니다."
  };
}

/** 지역 목록 메타 — 같은 지역·학습 대상의 활성 시뮬레이션 모델이 있으면 준비됨(③·④는 요인분석 모델과 무관). */
export function regionRealStatus(catalog, regionId, target) {
  const ready = (catalog?.instances ?? []).some((row) => row.mode === "real" && row.region_id === regionId && row.target === target &&
    row.model_type === "simulation" && row.active_version);
  return ready ? "실데이터 시뮬레이션 모델 준비됨" : "실데이터 모델 없음";
}
