// 인구감소 대응 정책 전략 카탈로그 + RICE 우선순위 산정.
// Notion "정책별 우선 순위 도출 / 인구 감소 유형별 맞춤형 정책"을 RICE 프레임워크로 정량화.
// RICE = (Reach × Impact × Confidence) / Effort.  fit = 지자체 policyImpacts 축과 연동(없으면 평균).

export const POLICY_STRATEGIES = [
  { id: "youth_job", name: "청년 일자리 창출", category: "청년 인구 유입", reach: 8200, impact: 3, confidence: 0.8, effort: 9, fit: "industry" },
  { id: "housing", name: "청년 주거·전세 안정", category: "청년 인구 유입", reach: 6400, impact: 2, confidence: 0.85, effort: 6, fit: "housing" },
  { id: "childcare", name: "돌봄·양육비 경감", category: "사회적 약자 돌봄", reach: 5100, impact: 2, confidence: 0.9, effort: 5, fit: "welfare" },
  { id: "amenity", name: "편의시설(의료·교육·복지) 확충", category: "도시 쾌적성", reach: 7300, impact: 2, confidence: 0.75, effort: 8, fit: "welfare" },
  { id: "transit", name: "교통 접근성 강화", category: "도시 쾌적성", reach: 6900, impact: 1, confidence: 0.7, effort: 7, fit: null },
  { id: "living_pop", name: "생활인구(관광·통근·귀농) 관리", category: "생활인구 관리", reach: 9100, impact: 1, confidence: 0.65, effort: 5, fit: null },
  { id: "culture", name: "문화·주택정비 도시 쾌적성", category: "도시 쾌적성", reach: 5600, impact: 1, confidence: 0.7, effort: 6, fit: "housing" },
  { id: "safety", name: "외국인·보호종료아동 안전망", category: "안전한 도시", reach: 3200, impact: 2, confidence: 0.8, effort: 4, fit: "welfare" }
];

/**
 * 지자체 특성(policyImpacts) + 시뮬레이터 정책 변수 가중치(슬라이더)를 함께 반영한
 * RICE 추천 점수를 계산하고 내림차순 정렬해 반환.
 * @param {object} region
 * @param {{ welfare:number, industry:number, housing:number }} [weights] 0~100 슬라이더 값
 * @param {number} [budgetFactor] 총 예산 제약요소(√(예산/기준)) — 추천 점수를 일관 스케일링
 */
export function rankStrategies(region, weights = { welfare: 50, industry: 50, housing: 50 }, budgetFactor = 1) {
  const impacts = region.policyImpacts;
  const avgFit = (impacts.welfare + impacts.industry + impacts.housing) / 3;
  const avgWeight = (weights.welfare + weights.industry + weights.housing) / 3 / 100;

  return POLICY_STRATEGIES.map((s) => {
    const fitValue = s.fit ? impacts[s.fit] : avgFit;
    // 시뮬레이터 슬라이더 강조도(0~1) → 0.5~1.5배 가중. 강조한 정책 축의 전략이 부스트됨.
    const emphasis = s.fit ? weights[s.fit] / 100 : avgWeight;
    const adjImpact = s.impact * (1 + fitValue) * (0.5 + emphasis);
    // 총 예산(제약요소)이 추천 실행 규모를 좌우 — 예산이 작으면 동일 전략도 점수↓.
    const score = ((s.reach * adjImpact * s.confidence) / s.effort) * budgetFactor;
    return {
      ...s,
      fitValue,
      emphasis: Number(emphasis.toFixed(2)),
      adjImpact: Number(adjImpact.toFixed(2)),
      score: Math.round(score)
    };
  }).sort((a, b) => b.score - a.score);
}
