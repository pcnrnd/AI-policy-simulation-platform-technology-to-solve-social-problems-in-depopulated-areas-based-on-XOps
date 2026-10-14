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
export function matchingFlowBasis(factor, simulation) {
  const a = factor?.provenance, b = simulation?.provenance;
  return Boolean(a && b && a.model_type === "factor_analysis" && b.model_type === "simulation" &&
    ["mode", "source", "region_id", "target", "dataset_hash", "prediction_period", "dong_code"].every((key) => a[key] === b[key]) &&
    a.mode === "real" && JSON.stringify(a.features) === JSON.stringify(b.features));
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
