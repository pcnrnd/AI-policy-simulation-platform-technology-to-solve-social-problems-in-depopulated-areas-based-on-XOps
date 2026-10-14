export const MODEL_TYPES = [{ id: "factor_analysis", name: "요인분석" }, { id: "simulation", name: "시뮬레이션" }];
export const TARGET_LABELS = { nonlocal_visitors: "타지역 방문객", observed_sales_krw: "관측 소비", population_change_pct: "인구 변화율" };
export function pipelineDefinition(form, datasets) {
  if (!MODEL_TYPES.some((type) => type.id === form.model_type)) throw new Error("모델 종류를 선택하세요.");
  const dataset = datasets.find((row) => row.id === form.dataset_id);
  if (!dataset) throw new Error("등록된 학습데이터를 선택하세요.");
  if (!/^[A-Za-z0-9_-]{3,128}$/.test(form.id.trim())) throw new Error("파이프라인 ID는 영문·숫자·밑줄·하이픈 3~128자로 입력하세요.");
  if (!form.name.trim()) throw new Error("파이프라인 이름을 입력하세요.");
  return { id: form.id.trim(), name: form.name.trim(), model_type: form.model_type,
    dataset_id: dataset.id, dataset_hash: dataset.content_hash, trigger_policy: form.trigger_policy || "수동" };
}
export function versionStatus(status) { return ({ active: "활성", previous: "이전", candidate: "등록" })[status] ?? "등록"; }
