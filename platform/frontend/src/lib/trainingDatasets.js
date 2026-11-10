/** 모델과 target이 모두 일치하는 학습 입력만 사용한다. */
export function datasetMatchesModel(dataset, model) {
  return Boolean(model?.target && dataset?.spec?.model_id === model.model_id && dataset.spec.target === model.target);
}

/** 새 항목을 먼저 두고 동일 ID의 스냅샷을 갱신한다. */
export function mergeDatasets(incoming = [], existing = []) {
  const seen = new Set();
  return [...incoming, ...existing].filter((dataset) => {
    if (!dataset?.dataset_id || seen.has(dataset.dataset_id)) return false;
    seen.add(dataset.dataset_id);
    return true;
  });
}

/** 준비 결과는 같은 모드·원본·스키마·조건에서만 등록할 수 있다. */
export function catalogSelectionKey(source, filter = "", sort = "", mode = "real") {
  return JSON.stringify([mode, source?.id ?? null, source?.source, source?.object, source?.lineage,
    source?.columns ?? [], source?.range ?? null, filter.trim(), sort]);
}

export function catalogRegistrationBody(source, filter, sort, prepared, name, purpose, mode) {
  const key = catalogSelectionKey(source, filter, sort, mode);
  if (!source || !prepared?.data?.complete || prepared.key !== key) return null;
  return { source_id: source.id, filter: filter.trim(), sort, name: name.trim(), purpose,
    columns: prepared.data.query.columns, source_hash: prepared.data.content_hash };
}
