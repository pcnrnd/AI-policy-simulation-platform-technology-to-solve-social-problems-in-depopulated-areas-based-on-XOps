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

// Model Store 초기 이력 — 2차년도 "Feature/Model Store 기반 버전 관리" 산출물 표면화.
// 파이프라인 승급 완료 시 신규 버전이 운영으로 추가되고 직전 운영 버전은 '이전'으로 강등된다.
export const MODEL_STORE = [
  { modelId: "population-forecast", version: "v3.0-R3", dataVersion: "ds-v12", params: "lr 0.003 · depth 8", accuracy: 0.892, status: "운영", registeredAt: "2026-05-21" },
  { modelId: "population-forecast", version: "v2.9", dataVersion: "ds-v9", params: "lr 0.005 · depth 6", accuracy: 0.871, status: "이전", registeredAt: "2026-02-14" },
  { modelId: "vital-population", version: "v2.4", dataVersion: "ds-v8", params: "window 14d · units 128", accuracy: 0.864, status: "운영", registeredAt: "2026-04-30" },
  { modelId: "vital-population", version: "v2.3", dataVersion: "ds-v7", params: "window 7d · units 96", accuracy: 0.852, status: "이전", registeredAt: "2026-01-22" },
  { modelId: "settlement-demand", version: "v1.7", dataVersion: "ds-v5", params: "trees 400 · depth 10", accuracy: 0.847, status: "운영", registeredAt: "2026-03-18" },
  { modelId: "settlement-demand", version: "v1.6", dataVersion: "ds-v4", params: "trees 300 · depth 8", accuracy: 0.83, status: "롤백", registeredAt: "2025-12-02" }
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
    triggerPolicy: "드리프트(PSI > 0.2)·성능 저하(Acc < 0.85) 자동 · 수동"
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
