/** 모델과 target이 모두 일치하는 학습 입력만 사용한다. */
export function datasetMatchesModel(dataset, model) {
  return Boolean(model?.target && dataset?.spec?.model_id === model.model_id && dataset.spec.target === model.target);
}
