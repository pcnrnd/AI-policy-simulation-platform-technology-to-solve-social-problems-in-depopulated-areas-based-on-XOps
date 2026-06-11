// 서빙 중인 운영 모델 레지스트리 — 모니터링 대상 선택과 재학습 파이프라인이 공유한다.
// accDelta/errRatio: 기준 시계열(mock metrics_history) 대비 모델별 결정적 오프셋.
export const MODEL_REGISTRY = [
  {
    id: "population-forecast",
    name: "인구이동 예측",
    version: "v3.0-R3",
    domain: "전입·전출 이동 패턴",
    accDelta: 0,
    errRatio: 1
  },
  {
    id: "vital-population",
    name: "생활인구 추정",
    version: "v2.4",
    domain: "방문·숙박 생활인구",
    accDelta: -0.014,
    errRatio: 1.18
  },
  {
    id: "settlement-demand",
    name: "정주여건 수요예측",
    version: "v1.7",
    domain: "임대주택·시설 수요",
    accDelta: -0.031,
    errRatio: 1.42
  }
];

// 등록된 재학습 파이프라인 카탈로그 — 모델 레지스트리와 1:1.
export const RETRAIN_PIPELINES = [
  {
    id: "PL-POP-RETRAIN-01",
    name: "인구이동 예측 재학습",
    modelId: "population-forecast",
    model: "population-forecast",
    baseVersion: "v3.0-R3",
    candidateVersion: "v3.1",
    experiment: "EXP-POP-DECLINE-031",
    triggerPolicy: "드리프트 자동 (PSI > 0.2) · 수동"
  },
  {
    id: "PL-VITAL-RETRAIN-02",
    name: "생활인구 추정 재학습",
    modelId: "vital-population",
    model: "vital-population",
    baseVersion: "v2.4",
    candidateVersion: "v2.5",
    experiment: "EXP-VITAL-POP-012",
    triggerPolicy: "주간 배치 (매주 월 02:00)"
  },
  {
    id: "PL-SETTLE-RETRAIN-03",
    name: "정주여건 수요예측 재학습",
    modelId: "settlement-demand",
    model: "settlement-demand",
    baseVersion: "v1.7",
    candidateVersion: "v1.8",
    experiment: "EXP-SETTLE-DMD-007",
    triggerPolicy: "수동"
  }
];
