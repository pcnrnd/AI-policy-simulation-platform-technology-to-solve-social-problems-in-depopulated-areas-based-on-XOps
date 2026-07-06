// 인구 예측 시뮬레이션 모델 — SimulatorPage와 시나리오 비교가 공유하는 순수 계산 유틸.

export const YEAR_LABELS = ["2026", "2027", "2028", "2029", "2030", "2031", "2032", "2033", "2034", "2035"];
export const NATURAL_DECLINE = 0.98;
export const BASELINE_BUDGET = 600; // 억 — budgetFactor=1 기준 예산

// 총 예산(제약요소) → 체감수익 곡선 스케일 팩터.
export function budgetToFactor(budgetTotal) {
  return Math.sqrt(budgetTotal / BASELINE_BUDGET);
}

// 시설 조절변수(x) 정규화×weight 합산 부스트.
export function controlBoostOf(controls, controlValues) {
  return (controls ?? []).reduce((sum, c) => {
    const v = controlValues?.[c.key] ?? c.default;
    const norm = (v - c.min) / (c.max - c.min);
    return sum + norm * c.weight;
  }, 0);
}

export function computeTrends(region, welfare, industry, housing, budgetFactor = 1, controlBoost = 0) {
  const basePop = region.population;
  const combined =
    region.policyImpacts.welfare * (welfare / 100) +
    region.policyImpacts.industry * (industry / 100) +
    region.policyImpacts.housing * (housing / 100);

  // 정책 배분 + 시설 조절변수(controlBoost)를 합산한 효과를 총 예산(제약요소)이 스케일.
  const growthModifier = (combined + controlBoost) * 0.06 * budgetFactor;
  const simRate = NATURAL_DECLINE + growthModifier;

  const baseTrend = [];
  const simTrend = [];
  let basePopTracker = basePop;
  let simPopTracker = basePop;

  for (let i = 0; i < 10; i++) {
    basePopTracker = Math.round(basePopTracker * NATURAL_DECLINE);
    simPopTracker = Math.round(simPopTracker * simRate);
    baseTrend.push(basePopTracker);
    simTrend.push(simPopTracker);
  }
  return { baseTrend, simTrend };
}

// 시나리오(저장된 변수 조합)의 10개년 추이 계산.
export function computeScenarioTrend(region, scenario) {
  const controls = region.case?.simulation?.controls ?? [];
  const factor = budgetToFactor(scenario.budgetTotal);
  const boost = controlBoostOf(controls, scenario.controlValues);
  return computeTrends(
    region,
    scenario.welfareWeight,
    scenario.industryWeight,
    scenario.housingWeight,
    factor,
    boost
  );
}
