export const REAL_TARGETS = { observed_sales_krw: "관측 소비", nonlocal_visitors: "타지역 방문객" };
export function createRequestGate(epochReader) {
  let sequence=0;
  return { cancel:() => { sequence+=1; }, begin:() => {
    const request=++sequence, epoch=epochReader();
    return () => request===sequence && epoch===epochReader();
  } };
}
export function observedFeatureLabel(name, target, dongs = []) {
  const noun = REAL_TARGETS[target] ?? target;
  const labels = { y_lag1: `전월 ${noun}`, y_lag2: `2개월 전 ${noun}`, y_lag3: `3개월 전 ${noun}`, y_yoy: `전년 같은 달 ${noun}`,
    has_yoy: "전년 관측값 유무", month_sin: "월별 계절성 1", month_cos: "월별 계절성 2" };
  if (name.startsWith("dong_")) return `행정동: ${dongs.find((row) => row.code === name.slice(5))?.name ?? name.slice(5)}`;
  return labels[name] ?? name;
}
export function activeModelPair(catalog, region, target) {
  if (catalog.mode !== "real" || catalog.source !== "real") throw new Error("실데이터 모델 목록을 확인하세요.");
  const candidates = catalog.instances.filter((row) => row.mode === "real" && row.region_id === region && row.target === target);
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
// STAGE ①·② 요인분석 자리 — 요인분석 학습데이터가 등록되기 전까지 실데이터 표시는 안내만 둔다(전문은 한 곳, 나머지는 한 줄).
export const FACTOR_GUIDE_TEXT = "요인분석은 인구 구조·시설 지표 학습데이터를 등록한 뒤 제공됩니다. DataOps에서 요인분석용 학습데이터를 등록하고 오케스트레이터에서 요인분석 모델을 학습·활성화하세요.";
export const FACTOR_PENDING_TEXT = "요인분석 학습데이터 등록 후 표시";
export const FACTOR_RUN_LOCK_TEXT = "요인분석 학습데이터를 등록한 뒤 실행할 수 있습니다.";
// 알림(role=alert) 고정 문구 — 서버·브라우저 원문은 활동 로그에만 남긴다.
export const REAL_LOAD_FAILED_TEXT = "모델 정보를 불러오지 못했습니다. 잠시 뒤 다시 시도하세요.";
export const REAL_PREDICT_FAILED_TEXT = "예측을 계산하지 못했습니다. 입력을 다시 조정하거나 잠시 뒤 다시 시도하세요.";
export const REAL_VERSION_CHANGED_TEXT = "활성 버전이 바뀌었습니다. 새 활성 버전의 관측 입력으로 예측을 다시 계산합니다.";

export const targetUnit = (target) => target === "observed_sales_krw" ? "원" : "명";
export const formatValue = (value) => Number.isFinite(value) ? value.toLocaleString("ko-KR", { maximumFractionDigits: 3 }) : "—";
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
    score: row.delta, scoreText: `+${formatValue(row.delta)}`,
    cells: [`${formatValue(row.from)} → ${formatValue(row.to)}`, formatValue(row.delta)]
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
