// 데모 표시의 합성 값 — 원본이 없는 시설 수치(현재 사용률·초기 입주율·규모 보정·정주 전환·소요 예산)와 방안 선택 규칙.
// 규칙 범위는 시드 conversion.synthetic.rules, 생성 방법은 conversion.synthetic.method에 있다.
// 난수는 고정 시드(random_seed)와 시설 id(·방안)로 만든 결정적 난수라 같은 시드면 항상 같은 값이다.
// 실데이터 표시는 이 파일을 쓰지 않는다(합성 금지).

function mulberry32(a) {
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const hash = (text) => [...text].reduce((h, ch) => (Math.imul(h, 31) + ch.codePointAt(0)) | 0, 0);
/** 시드·키별 결정적 난수 함수(0~1). */
export const rngFor = (seed, key) => mulberry32(seed ^ hash(key));
const between = (r, [lo, hi]) => lo + r * (hi - lo);
const pickInt = (rng, range) => Math.round(between(rng(), range));
const mid = (r) => (r ? (r.low + r.high) / 2 : 0);

const isIdle = (f) => ["계획", "조성"].includes(f.stage);
const utilKey = (f, rules) => {
  if (isIdle(f)) return "기타|유휴";
  const key = `${f.type}|운영`;
  return rules.base_util_range[key] ? key : "기타|운영";
};

/** 공급 호수만 있고 입주율이 없는 주거 시설의 초기 입주율(%) — 그 밖은 null. */
export function leaseUpOccupancy(f, seed, rules) {
  if (!Number.isFinite(f.supply_units) || Number.isFinite(f.occupancy_rate)) return null;
  return pickInt(rngFor(seed, `${f.id}|leaseup`), rules.lease_up_range);
}

/** 현재 사용률(%) — 입주율이 있으면 그 값, 없으면 유형·단계 범위에서 시설 id 난수. */
export function baseUtil(f, seed, rules) {
  if (Number.isFinite(f.occupancy_rate)) return f.occupancy_rate;
  return pickInt(rngFor(seed, f.id), rules.base_util_range[utilKey(f, rules)]);
}

/** 규모 보정 배수 — (연면적 ÷ 등급 대표 연면적)의 제곱근을 자른 값 × 시설 id 흔들림. */
export function scaleFactor(f, grade, seed, rules) {
  const rep = rules.capacity_grade_area_m2[grade];
  const area = Number.isFinite(f.scale_area_m2) ? f.scale_area_m2 : rep;
  const [lo, hi] = rules.scale_adjust.clamp;
  const base = Math.min(hi, Math.max(lo, Math.sqrt(area / rep)));
  return base * between(rngFor(seed, `${f.id}|scale`)(), rules.scale_adjust.jitter);
}

/** 방문형 방안의 정주인구 변화(연) — 운영 인력 × 세대당 인구 + 생활인구 효과 × 정주 전환 비율. */
export function settleSynth(f, planType, livingMid, { area, pph, seed, rules }) {
  const ratio = rules.settle_ratio_pct[planType] ?? [0.2, 0.5];
  const staff = rules.staff_per_1000m2[planType] ?? [0.5, 1];
  const r = rngFor(seed, `${f.id}|settle|${planType}`)();
  const jobs = (area / 1000) * between(rngFor(seed, `${f.id}|staff|${planType}`)(), staff);
  return Math.round(jobs * pph + (livingMid * between(r, ratio)) / 100);
}

/** 소요 예산(억 원) 범위 — 대상 면적 × 방안 유형별 단가. 주거 시설은 공실 비율만큼의 면적. */
export function budgetOf(f, planType, grade, rules) {
  const sourced = Number.isFinite(f.scale_area_m2) && String(f.basis?.scale_area_m2 ?? "").startsWith("source");
  let area = sourced ? f.scale_area_m2 : rules.capacity_grade_area_m2[grade];
  if (Number.isFinite(f.occupancy_rate)) area *= (100 - f.occupancy_rate) / 100;
  const [lo, hi] = rules.unit_cost_manwon_per_m2[planType];
  const toEok = (manwon) => Math.round(((area * manwon) / 10000) * 10) / 10;
  return { low: toEok(lo), high: toEok(hi) };
}

/** 예산 ±20%일 때 생활인구 효과 범위 — 방안 유형별 탄력도. */
export function budgetSensitivity(option, rules) {
  const e = rules.budget_elasticity[option.planType] ?? 0.5;
  const d = rules.budget_change * e;
  return { low: Math.round(option.living.low * (1 - d)), high: Math.round(option.living.high * (1 + d)) };
}

/**
 * 방안 선택 규칙 — 고를 방안이 후보에 없으면 다음 규칙으로 넘어간다. option이 null이면 근거 등급·점수로 고른 방안을 쓴다.
 *  1) 주거 시설: 정착형  2) 열린 공간이 있으면 개방·공유  3) 현재 사용률이 기준 미만이면 운영 개선
 *  4) 계획·조성 단계면 용도 전환  5) 그 밖은 근거 등급·점수
 */
export function choosePlan(f, options, util, rules) {
  const has = (t) => options.find((o) => o.planType === t && o.living.high > 0);
  if (f.type === "임대주택") return { option: options.find((o) => o.kind === "house") ?? null, rule: "주거시설 정착" };
  if (f.open_space && has("개방·공유")) return { option: has("개방·공유"), rule: `열린 공간 ${f.open_space}` };
  if (util < rules.low_util_pct && has("운영 개선")) return { option: has("운영 개선"), rule: `사용률 ${rules.low_util_pct}% 미만` };
  if (isIdle(f) && has("용도 전환")) return { option: has("용도 전환"), rule: `${f.stage === "계획" ? "계획" : "조성 중"} 단계` };
  return { option: null, rule: "생활인구 효과가 큰 방안" };
}

/** 도입 곡선 — 전환 뒤 k분기의 효과 비율(0~1), 방안 유형별 분기 수에 걸쳐 1 − (1 − t)². */
export function adoption(planType, k, rules) {
  const q = rules.adoption_quarters[planType] ?? 4;
  const t = Math.min(1, k / q);
  return 1 - (1 - t) ** 2;
}

/** 반영 효과 추이 — 현재(0)부터 n분기까지 정상 상태 값 × 도입 곡선(명). */
export const effectPath = (steady, planType, n, rules) => Array.from({ length: n + 1 }, (_, k) => Math.round(steady * adoption(planType, k, rules)));

export { mid as rangeMid };
