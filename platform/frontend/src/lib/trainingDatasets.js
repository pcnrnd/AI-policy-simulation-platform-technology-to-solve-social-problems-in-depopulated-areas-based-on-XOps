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
