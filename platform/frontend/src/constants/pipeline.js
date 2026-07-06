export const PIPELINE_STEPS = [
  {
    name: "node-event",
    desc: "1. Event Trigger",
    log: "ALERT: 데이터 드리프트 감지 이벤트 수신. (PSI: 0.384 > 임계치 0.20) 재학습 자동 스케줄 파이프라인 트리거 완료.",
    warn: true
  },
  {
    name: "node-prep",
    desc: "2. Data Prep",
    log: "INFO: 분산 데이터 소스 수집 및 정제 처리 가동. 6개 연계 데이터 소스(주민·복지·산업·공간·스마트팜·시설) 메타데이터 검증 완료. 데이터셋 로딩 중..."
  },
  {
    name: "node-train",
    desc: "3. Retraining",
    log: "INFO: Flower 연합학습 엔진 기반 분산 학습 개시. 연합 클라이언트 노드에서 분할 가중치 병렬 트레이닝 시작 (Differential Privacy 적용)."
  },
  {
    name: "node-eval",
    desc: "4. Evaluation",
    log: "INFO: 학습 완료. 모델 메트릭 자동 비교 연산 수행. [SOTA 검증성공: Accuracy 기존 0.892 -> 신규 0.925 (+3.3% 향상, 승급 기준 1.5% 돌파)]. 자동 테스트 통과 — 유닛 12/12 · 통합 5/5 · 성능 P95 138ms."
  },
  {
    name: "node-deploy",
    desc: "5. Canary Deploy",
    log: "INFO: 신규 모델 카나리 점진적 배포 단계 진입. 트래픽 10% 자동 유입 및 롤백 임계 상태 모니터링 개시. 지연속도 120ms 양호."
  },
  {
    name: "node-rollback",
    desc: "6. SOTA Promoted",
    log: "SUCCESS: 최적 성능 모델(SOTA)로 최종 승급 승격 완료! Helm Chart 및 모델 레지스트리 정보 자동 갱신 완료."
  }
];

export const PIPELINE_NODES = [
  { id: "node-event", label: "1. Event Trigger", icon: "fa-bell" },
  { id: "node-prep", label: "2. Data Prep", icon: "fa-gears" },
  { id: "node-train", label: "3. Retraining", icon: "fa-brain" },
  { id: "node-eval", label: "4. Evaluation", icon: "fa-square-poll-vertical" },
  { id: "node-deploy", label: "5. Canary Deploy", icon: "fa-rocket" },
  { id: "node-rollback", label: "6. SOTA Promoted", icon: "fa-shield-halved" }
];
