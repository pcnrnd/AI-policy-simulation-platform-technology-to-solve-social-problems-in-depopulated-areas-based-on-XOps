// 시설 × 전환 방안 추천 — 시설마다 전환 방안 후보 2~3개의 효과 범위(사용률·생활인구·정주인구)를 계산하고,
// 생활인구 효과 중심의 추천 지수(0~80)로 시설 순위와 시설별 방안을 정한다.
// 화면에는 결과만 보이고 계산식·비율·가중치는 보이지 않는다(계산 방법은 이 파일·recoSynthetic.js·시드 conversion.synthetic.method에만).
import {
  baseUtil, budgetOf, budgetSensitivity, choosePlan, effectPath, leaseUpOccupancy, rangeMid, scaleFactor, settleSynth
} from "./recoSynthetic.js";

export const PLAN_TYPES = ["용도 전환", "복합화", "운영 개선", "개방·공유"];

// 전환 방안 후보 — [방안 유형, 방안 문구, 효과 종류]. 효과 종류: visit 방문형(기본) · stay 공실 단기 체류 · house 주거 · job 일자리.
// 폐교 활용은 운영 단계로 후보가 갈린다(운영 중이면 운영형, 계획·조성 단계면 전환형). 시설 자료에 use_plan이 있으면 같은 유형 후보 문구를 바꾼다.
const OPTION_RULES = {
  문화관광: [["복합화", "전시·정원·광장 복합 운영으로 체류 시간 확대 검토"], ["운영 개선", "야간·계절 행사 운영 개선 검토"], ["개방·공유", "광장·정원 지역 행사 개방 검토"]],
  "섬 관광": [["운영 개선", "테마 행사·체류 프로그램 운영 개선 검토"], ["복합화", "숙박·체험 기능 복합화 검토"]],
  "폐교 활용(운영)": [["복합화", "전시·체험 프로그램 복합 운영 검토"], ["운영 개선", "전시 해설·연계 관광 운영 개선 검토"], ["개방·공유", "주민 문화 프로그램 공간 개방 검토"]],
  "폐교 활용": [["복합화", "체험과 숙박 기능 복합 운영 검토"], ["용도 전환", "캠핑·체험 시설로 전환 검토"], ["용도 전환", "귀농·귀촌 체류형 주거로 전환 검토", "house"]],
  교육: [["용도 전환", "교육·지원 거점으로 전환 검토", "job"], ["복합화", "평생학습·창업 공간 복합화 검토"]],
  스마트팜: [["운영 개선", "청년 창업·교육 프로그램 운영 개선 검토", "job"], ["개방·공유", "재배 체험·견학 프로그램 개방 검토"]],
  임대주택: [["용도 전환", "공실 호실 단기 체류형 전환 검토", "stay"], ["운영 개선", "입주 지원·정착 프로그램 운영 개선 검토", "house"]],
  신재생: [["개방·공유", "견학·체험 프로그램 개방 검토"], ["운영 개선", "이익공유 연계 일자리 프로그램 운영 개선 검토", "job"]],
  정원수: [["복합화", "정원 체험·방문 기능 복합화 검토"], ["운영 개선", "조합 일자리·교육 프로그램 운영 개선 검토", "job"]]
};
const DEFAULT_OPTIONS = [["운영 개선", "이용 실적 확인 후 운영 개선 검토"], ["개방·공유", "유휴 공간 개방 검토"]];
const SETTLE_KINDS = new Set(["house", "job"]);

// 추천 지수 = 생활인구 효과(중심) 0.6 + 정주 효과 0.2(각 0~100 점수). 근거 없는 추진 여건 예시값은 지수에서 뺐다(결정 52) — 만점 80.
// 생활인구 효과 점수는 지역 상한(월평균 체류인구 × 상한 비율) 대비 로그 척도, 정주 효과 점수는 지역 연간 다른 시군구 전입의 1%를 만점으로 둔다.
// 고정 가중치 — 시설 대장 실적(2단계)이 들어오면 실적 기반으로 바꾼다.
const INDEX_WEIGHTS = { living: 0.6, settle: 0.2 };
export const INDEX_MAX = 100 * (INDEX_WEIGHTS.living + INDEX_WEIGHTS.settle);
// 결정 72 — ③ 목적함수가 정주를 겨누면 정주 효과를 무겁게(합 0.8 그대로, 만점 80). 그 밖은 생활인구 중심.
const SETTLE_FOCUS_WEIGHTS = { living: 0.3, settle: 0.5 };
/** 목적함수 문구 → 추천 지수 가중(생활인구·정주). */
export const indexWeights = (objective = "") => (/정주/.test(objective) ? SETTLE_FOCUS_WEIGHTS : INDEX_WEIGHTS);
// ③ 정책 강조 가중치(0~100, 기본 50) → 방안 유형 가중 배수 0.5~1.5. 정착·주거 = 주거·체류형, 창업·일자리 = 일자리형, 돌봄·커뮤니티 = 주민 개방형.
/** 방안이 받는 정책 강조 축 — housing | industry | welfare | null. */
export const policyAxis = (o) => (o.kind === "house" || o.kind === "stay" ? "housing" : o.kind === "job" ? "industry" : o.planType === "개방·공유" ? "welfare" : null);
const policyFactor = (o, weights) => { const axis = policyAxis(o); const w = axis ? weights?.[axis] : null; return Number.isFinite(w) ? 0.5 + w / 100 : 1; };
export const FORECAST_QUARTERS = 12; // 정책 효과 전망 3년
const SETTLE_FULL_SHARE = 0.01;
// 판단: 시설 안에서 고른 방안이 추천, 고른 방안 점수 대비 이 비율 이상이면 검토, 그 아래는 보류.
const REVIEW_RATIO = 0.8;

const isLinked = (c) => Boolean(c.facility_id && c.field);
const valueOf = (c, controlValues) => controlValues?.[c.key] ?? c.default;

/** 시설 조절변수 중 특정 시설 값에 연결된 것만 — 이 값이 바뀔 때만 추천을 다시 도출한다. */
export function linkedControls(controls) {
  return (controls ?? []).filter(isLinked);
}

/** 추천 재도출 판정 서명 — 시설 값에 연결된 조절변수만 넣는다. */
export function recoControlSig(controls, controlValues) {
  return linkedControls(controls).map((c) => `${c.key}:${valueOf(c, controlValues)}`).join(",");
}

/** 시설 값이 없는 조절변수를 바꿨을 때 재도출 안내 자리에 둘 행동 안내(바꾸지 않았으면 null). */
export function unlinkedControlNote(controls, controlValues) {
  const changed = (controls ?? []).filter((c) => !isLinked(c) && valueOf(c, controlValues) !== c.default);
  return changed.length ? `${changed.map((c) => c.label).join("·")}: 시설 자료를 등록하면 추천에 반영됩니다.` : null;
}

// 주거 시설 입주 상태 — 공급 호수가 조절되면 현재 입주 호수 기준으로 다시 계산한다(공급보다 많이 입주할 수 없음).
function housingOf(facility, supply) {
  if (!Number.isFinite(facility.occupancy_rate)) return null;
  if (!Number.isFinite(facility.supply_units)) return { occupancy: facility.occupancy_rate, units: null, vacant: null };
  const units = Number.isFinite(supply) && supply > 0 ? supply : facility.supply_units;
  const occupied = (facility.occupancy_rate / 100) * facility.supply_units;
  return { occupancy: Math.min(100, (occupied / units) * 100), units, vacant: Math.max(0, units - occupied) };
}

/** 시설 유형·운영 단계로 정한 전환 방안 후보(2~3개). */
export function optionCandidates(facility) {
  const key = facility.type === "폐교 활용" && facility.stage === "운영" ? "폐교 활용(운영)" : facility.type;
  const plan = facility.use_plan;
  let replaced = false;
  return (OPTION_RULES[key] ?? DEFAULT_OPTIONS).map(([planType, text, kind = "visit"]) => {
    const own = !replaced && plan?.type === planType;
    if (own) replaced = true;
    return { planType, text: own ? plan.text : text, kind };
  });
}

const span = (low, high) => ({ low: Math.round(Math.min(low, high)), high: Math.round(Math.max(low, high)) });
const mid = (r) => (r.low + r.high) / 2;
const logScore = (value, full) => (full > 1 ? Math.min(100, (100 * Math.log10(1 + value)) / Math.log10(1 + full)) : 0);

// 규모 등급 — 출처 연면적이 있으면 그 값으로, 없으면 시설 유형별 기본 등급(합성 규칙)으로 정한다.
export function scaleGrade(item, rules) {
  const [small, large] = rules.scale_area_thresholds_m2;
  if (Number.isFinite(item.scale_area_m2)) return item.scale_area_m2 < small ? "소형" : item.scale_area_m2 < large ? "중형" : "대형";
  return rules.scale_grade_by_type[item.facility_type ?? item.type] ?? "중형";
}
// 규모 보정 배수 — 시설 등급 계수 ÷ 사례 등급 계수. 2차 보정부터 큰 시설에도 그대로 걸고(1로 막지 않음), 결과는 시설 수용 상한·지역 상한으로 자른다.
const scaleRatio = (facility, kase, rules) => rules.scale_factor[scaleGrade(facility, rules)] / rules.scale_factor[scaleGrade(kase, rules)];
// 운영 중 시설 — 계획·조성 단계가 아니면(단계 자료가 없는 기존 운영 시설 포함) 운영 중으로 본다.
const isOperating = (facility) => !["계획", "조성"].includes(facility.stage);

/**
 * 시설 수용 상한(월 방문 인원). 출처 수용 인원이 등록형(계획 목표 포함)이면 그 인원 × 등록 회전,
 * 일 최대 방문(관측)이면 그 인원 × 월 운영일, 그 밖의 수용 인원이면 × 일 회전 × 월 운영일.
 * 수용 인원이 없으면 유형별 일 방문 상한(섬 관광처럼 건물 바닥면적 기준이 맞지 않는 유형) × 월 운영일,
 * 그것도 없으면 (출처 연면적 또는 규모 등급 대표 연면적) ÷ 1인당 면적(법정 수용인원 산정 기준) × 일 회전 × 월 운영일.
 */
export function capacityCap(facility, rules) {
  const turnover = rules.capacity_turnover;
  if (Number.isFinite(facility.capacity_people)) {
    const kind = facility.capacity_kind ?? "";
    const per = kind.includes("등록") ? turnover.enrolled : kind.startsWith("일 최대 방문") ? turnover.open_days : turnover.daily * turnover.open_days;
    return facility.capacity_people * per;
  }
  const daily = rules.capacity_daily_by_type?.[facility.type];
  if (Number.isFinite(daily)) return daily * turnover.open_days;
  const area = Number.isFinite(facility.scale_area_m2) ? facility.scale_area_m2 : rules.capacity_grade_area_m2[scaleGrade(facility, rules)];
  return Math.round(area / rules.capacity_area_per_person_m2) * turnover.daily * turnover.open_days;
}

// 비교 가능 사례 — 같은 방안 유형·비슷한 지역 유형(섬·농촌·소도시)·인구감소지역, 그 시설 자신의 사례 제외. 시설·사례 속성만 본다.
// 추이 참고 사례(trend_only, 월평균 값 없음)는 효과 추정에 넣지 않고 사례 성과 그래프에만 쓴다(conversionOf·facilityRows).
// 같은 시설 유형 사례가 있으면 그것만 쓰고(2곳 이상이면 사례값 범위, 1곳이면 규모 보정 추정),
// 없으면 그 방안을 적용할 수 있는 다른 유형 사례를 규모로 보정한 추정(참고 사례)으로 쓴다.
export function comparableCases(option, facility, cases, rules) {
  const base = cases.filter((c) => c.plan_type === option.planType && c.facility_id !== facility.id
    && c.depopulation_area === true && rules.comparable_region_types.includes(c.region_type));
  const same = base.filter((c) => c.facility_type === facility.type);
  if (same.length) return { tier: same.length >= rules.min_cases_for_range ? "range" : "single", cases: same };
  const reference = base.filter((c) => c.applies_to.includes(facility.type));
  return { tier: reference.length ? "reference" : "none", cases: reference };
}

// 방안 하나의 효과 범위 — 값마다 basis(source | synthetic)를 붙인다.
function optionEffect(option, facility, housing, ctx) {
  const { region, cases, rules } = ctx;
  const cap = region.stay_monthly * rules.visit_cap_share;
  const inflowRatio = region.inflow_external / region.inflow_total;
  const fill = rules.housing_fill;
  const useful = housing?.vacant !== null && housing?.vacant !== undefined;
  const utilization = useful
    ? { ...span(((100 - housing.occupancy) * fill[0]), ((100 - housing.occupancy) * fill[1])), basis: "synthetic" }
    : { ...span(...rules.utilization_pp[option.planType]), basis: "synthetic" };
  if (SETTLE_KINDS.has(option.kind)) {
    const settle = option.kind === "house" && useful
      ? span(housing.vacant * fill[0] * region.persons_per_household * inflowRatio, housing.vacant * fill[1] * region.persons_per_household * inflowRatio)
      : span(region.inflow_external * rules.settle_share[0], region.inflow_external * rules.settle_share[1]);
    return { utilization, living: { ...settle, basis: "synthetic" }, settle: { ...settle, basis: "synthetic" }, cases: [], caseTier: "none" };
  }
  if (option.kind === "stay") {
    const vacant = useful ? housing.vacant : 0;
    const raw = span(vacant * rules.short_stay_per_unit[0], vacant * rules.short_stay_per_unit[1]);
    return { utilization, living: { ...span(Math.min(raw.low, cap), Math.min(raw.high, cap)), basis: "synthetic" }, settle: null, cases: [], caseTier: "none" };
  }
  // 방문형: 비교 가능 사례마다 규모 보정 배수와 운영 중 시설의 증가분(신규 개관 총량 사례일 때)을 곱하고,
  // 시설 수용 상한·지역 상한 가운데 작은 값으로 자른다.
  const facilityCap = capacityCap(facility, rules);
  const limit = Math.min(cap, facilityCap);
  const { tier, cases: matched } = comparableCases(option, facility, cases, rules);
  if (matched.length) {
    const ranges = matched.map((c) => {
      const scale = scaleRatio(facility, c, rules);
      const [incLow, incHigh] = c.opening === "new" && isOperating(facility) ? rules.operating_increment : [1, 1];
      return { low: c.monthly * scale * incLow, high: c.monthly * scale * incHigh, adjusted: scale !== 1 || incLow < 1 };
    });
    const low = Math.min(...ranges.map((r) => r.low)), high = Math.max(...ranges.map((r) => r.high));
    const adjusted = ranges.some((r) => r.adjusted) || high > limit;
    return {
      utilization,
      living: { ...span(Math.min(low, limit), Math.min(high, limit)), basis: tier === "range" && !adjusted ? "source" : "synthetic" },
      settle: null,
      cases: matched.map((c) => c.id),
      caseTier: tier
    };
  }
  const [a, b] = rules.visit_share[option.planType];
  return {
    utilization,
    living: { ...span(Math.min(region.stay_monthly * a, limit), Math.min(region.stay_monthly * b, limit)), basis: "synthetic" },
    settle: null,
    cases: [],
    caseTier: "none"
  };
}

// 근거 등급·점수로 고르는 방안 선택용 점수(생활인구 중심 + 정주 보조, 사례·통계 계산값 그대로) — 방안 선택 규칙 5번에만 쓴다.
function optionScore(effect, region, rules) {
  const living = logScore(mid(effect.living), region.stay_monthly * rules.visit_cap_share);
  const settle = effect.settle ? Math.min(100, (100 * mid(effect.settle)) / (region.inflow_external * SETTLE_FULL_SHARE)) : 0;
  return INDEX_WEIGHTS.living * living + INDEX_WEIGHTS.settle * settle;
}

// 근거 등급(3차 보정) — 같은 유형 사례 2곳 이상 범위 > 같은 유형 1곳 규모 보정 > 다른 유형 참고 사례 보정 > 통계 기준 추정
// (지역 생활인구·전입·공실 규모). 시설 안에서는 등급이 높은 방안을 먼저 고르고 같은 등급끼리만 점수로 비교한다.
// 그래서 등급이 낮은 추정은 값이 커도 추천이 되지 않고, 고른 방안 점수의 일정 비율 이상이면 검토에 머문다.
const EVIDENCE_GRADE = { range: 4, single: 3, reference: 2, none: 1 };
const gradeOf = (o) => EVIDENCE_GRADE[o.caseTier];

// 시설 하나의 방안 비교(효과·판단) — 지역 통계가 없으면 null(화면에서 행동 안내).
// 월평균 체류인구(stay_monthly)는 시드에 없고 플랫폼 공표 통계 API에서 받아 withRegionStay로 붙인다 — 받지 못하면 null.
function conversionOf(facility, housing, conversion) {
  const region = conversion?.regions?.[facility.region_id];
  const rules = conversion?.synthetic?.rules;
  if (!region || !rules || !(region.stay_monthly > 0)) return null;
  const ctx = { region, cases: (conversion.cases ?? []).filter((c) => !c.trend_only), rules };
  const options = optionCandidates(facility).map((option) => {
    const effect = optionEffect(option, facility, housing, ctx);
    return { ...option, ...effect, score: optionScore(effect, region, rules) };
  });
  // 같은 등급·같은 점수면 후보 순서가 앞선 방안.
  const chosen = options.reduce((a, o) => (gradeOf(o) > gradeOf(a) || (gradeOf(o) === gradeOf(a) && o.score > a.score) ? o : a));
  return options.map((o) => ({
    ...o,
    chosen: o === chosen,
    verdict: o === chosen && o.living.high > 0 ? "추천" : o.score >= chosen.score * REVIEW_RATIO && o.living.high > 0 ? "검토" : "보류"
  }));
}

/**
 * 시드 conversion에 지역 월평균 체류인구(플랫폼 API 값)를 붙인 새 객체 — 값이 없으면 그 지역은 미연결 그대로.
 * @param {object|null} conversion 시드 conversion
 * @param {string} regionId
 * @param {number|null|undefined} stayMonthly 공표 통계 API /public-stats/stay-average 값
 */
export function withRegionStay(conversion, regionId, stayMonthly) {
  const region = conversion?.regions?.[regionId];
  if (!region || !(stayMonthly > 0)) return conversion;
  return { ...conversion, regions: { ...conversion.regions, [regionId]: { ...region, stay_monthly: stayMonthly } } };
}

// 방안 점수(추천 지수 = 반올림 전 값) — 목적함수 가중(생활인구·정주) 합 × 정책 강조 배수.
function indexParts(option, region, rules, ctx) {
  const w = indexWeights(ctx.objective);
  const living = w.living * logScore(rangeMid(option.living), region.stay_monthly * rules.visit_cap_share);
  const settle = w.settle * Math.min(100, (100 * option.settleValue) / (region.inflow_external * SETTLE_FULL_SHARE));
  const policy = policyFactor(option, ctx.policyWeights);
  return { living, settle, policy, total: (living + settle) * policy };
}

// 고른 방안 기준 판단 — 고른 방안 추천, 고른 방안 점수의 일정 비율 이상이면 검토, 그 아래 보류.
// 소요 예산 하한이 ③ 총 예산을 넘는 방안은 "예산 초과"(고르지 않고 표 아래로).
const verdictsOf = (options, chosen) => options.map((o) => ({
  ...o,
  chosen: o === chosen,
  verdict: o.overBudget ? "예산 초과" : o === chosen && o.living.high > 0 ? "추천" : o.score >= chosen.score * REVIEW_RATIO && o.living.high > 0 ? "검토" : "보류"
}));

/**
 * 시설 하나의 방안을 시설 여건으로 보정하고 고른다(데모 합성 규칙, recoSynthetic.js).
 * 효과 = 사례·통계 계산값 × 규모 보정, 주거 정착형은 공급 호수 × 입주율 × 세대당 인구 × 외부 전입 비율,
 * 방문형의 정주 = 운영 인력 + 방문객 정주 전환. 전환 후 사용률 = 현재 + 방안별 변화, 소요 예산 = 면적 × 단가.
 */
function settleOptions(f, units, options, region, conversion, ctx) {
  const rules = conversion.synthetic.rules;
  const seed = conversion.synthetic.random_seed ?? 0;
  const grade = scaleGrade(f, rules);
  const util = baseUtil(f, seed, rules);
  const factor = scaleFactor(f, grade, seed, rules);
  const area = Number.isFinite(f.scale_area_m2) ? f.scale_area_m2 : rules.capacity_grade_area_m2[grade];
  const pph = region.persons_per_household;
  const houseSettle = Number.isFinite(units) && Number.isFinite(f.occupancy_rate)
    ? Math.round(units * (f.occupancy_rate / 100) * pph * (region.inflow_external / region.inflow_total)) : null;
  const band = (v) => ({ low: Math.round(v * rules.house_settle_band[0]), high: Math.round(v * rules.house_settle_band[1]), basis: "synthetic" });
  const scale = (r) => ({ ...r, low: Math.round(r.low * factor), high: Math.round(r.high * factor), basis: "synthetic" });
  const adjusted = options.map((o0) => {
    const o = o0.kind === "house" && houseSettle !== null ? { ...o0, living: band(houseSettle), settle: band(houseSettle) } : o0;
    const next = { ...o, living: scale(o.living), settle: o.settle ? scale(o.settle) : null };
    const livingMid = Math.round(rangeMid(next.living));
    const settleValue = next.settle ? Math.round(rangeMid(next.settle))
      : Math.max(livingMid > 0 ? 1 : 0, settleSynth(f, o.planType, livingMid, { area, pph, seed, rules }));
    const scored = { ...next, settleValue, settleSynthetic: !o.settle };
    const parts = indexParts(scored, region, rules, ctx);
    const budget = budgetOf(f, o.planType, grade, rules);
    return {
      ...scored,
      score: parts.total,
      parts,
      utilAfter: { low: Math.min(100, util + o.utilization.low), high: Math.min(100, util + o.utilization.high) },
      budget,
      overBudget: Number.isFinite(ctx.budgetTotal) && budget.low > ctx.budgetTotal
    };
  });
  // 고를 수 있는 방안 = 예산 안(모두 넘으면 전부). 규칙에 걸리지 않으면 근거 등급 → 가중 점수 순.
  const within = adjusted.filter((o) => !o.overBudget);
  const pool = within.length ? within : adjusted;
  const pick = choosePlan(f, pool, util, rules);
  const best = pool.reduce((a, o) => (gradeOf(o) > gradeOf(a) || (gradeOf(o) === gradeOf(a) && o.score > a.score) ? o : a));
  const chosen = pick.option ?? best;
  const final = verdictsOf(adjusted, chosen).map((o) => ({ ...o, budgetSens: budgetSensitivity(o, rules) }));
  const ch = final.find((o) => o.chosen);
  const livingMid = Math.round(rangeMid(ch.living));
  return {
    options: final,
    choice: ch,
    util,
    grade,
    scaleFactor: factor,
    choiceRule: pick.rule,
    forecast: {
      living: livingMid,
      settle: ch.settleValue,
      livingPath: effectPath(livingMid, ch.planType, FORECAST_QUARTERS, rules),
      settlePath: effectPath(ch.settleValue, ch.planType, FORECAST_QUARTERS, rules)
    }
  };
}

/**
 * 지역 시설 → 추천 행(추천 지수 높은 순, 순위 포함). 연결된 조절변수 값은 해당 시설의 필드를 대신한다.
 * 데모 표시에서만 부른다 — 시설 자료가 없는 실데이터 표시는 facilities가 비어 빈 배열이다.
 * 행: 시설(facility, 초기 입주율 합성 반영)·방안(options: 효과·전환 후 사용률·소요 예산·예산 ±20% 효과·판단)·고른 방안·
 *     추천 지수(index, 0~80)·현재 사용률(util)·방안 기준(choiceRule)·전망 효과 추이(forecast). 지역 통계가 없으면 options·index가 null.
 * @param {string} regionId
 * @param {object[]} facilities 시설 자료(id·name·type·region_id·stage·address·scale_area_m2·supply_units·open_space·use_plan·basis)
 * @param {object[]} controls 시설 조절변수(facility_id·field가 있으면 그 시설 값에 연결)
 * @param {Record<string, number>} controlValues
 * @param {object|null} conversion 시드 conversion — 지역 통계(regions)·출처 사례(cases)·합성 규칙(synthetic)
 * @param {{ objective?: string, budgetTotal?: number, policyWeights?: { welfare: number, industry: number, housing: number } }} [context]
 *   ③ 설정(결정 72): 목적함수 → 지수 가중, 총 예산(억 원) → 넘는 방안 제외, 정책 강조 → 방안 유형 가중
 */
export function facilityRows(regionId, facilities, controls = [], controlValues = {}, conversion = null, context = {}) {
  const linked = linkedControls(controls);
  const rules = conversion?.synthetic?.rules;
  const seed = conversion?.synthetic?.random_seed ?? 0;
  const here = (facilities ?? []).filter((f) => f.region_id === regionId);
  const rows = here.map((f0, order) => {
    const supply = linked.filter((c) => c.facility_id === f0.id && c.field === "supply_units").map((c) => valueOf(c, controlValues))[0];
    const lease = rules ? leaseUpOccupancy(f0, seed, rules) : null;
    const f = lease === null ? f0 : { ...f0, occupancy_rate: lease, basis: { ...f0.basis, occupancy_rate: "synthetic" } };
    const units = Number.isFinite(supply) && supply > 0 ? supply : f.supply_units;
    const options = conversionOf(f, housingOf(f, units), conversion);
    const settled = options ? settleOptions(f, units, options, conversion.regions[f.region_id], conversion, context) : null;
    const fallback = optionCandidates(f)[0];
    return {
      id: f.id,
      name: f.name,
      type: f.type,
      facility: f,
      order,
      planType: (settled?.choice ?? fallback).planType,
      plan: (settled?.choice ?? fallback).text,
      options: settled?.options ?? null,
      score: settled?.choice.score ?? null,
      index: settled ? Math.min(INDEX_MAX, Math.round(settled.choice.score)) : null,
      overBudget: Boolean(settled?.choice.overBudget),
      util: settled ? settled.util : null,
      choiceRule: settled?.choiceRule ?? null,
      scaleFactor: settled?.scaleFactor ?? null,
      grade: settled?.grade ?? null,
      budget: settled?.choice.budget ?? null,
      forecast: settled?.forecast ?? null
    };
  });
  // 추천 순위: 점수 높은 순(값 없는 시설·모든 방안이 예산 초과인 시설은 뒤, 같은 값은 시설 자료 순서).
  const ranked = [...rows].sort((a, b) => ((a.score === null) - (b.score === null)) || (a.overBudget - b.overBudget)
    || (b.score ?? 0) - (a.score ?? 0) || a.order - b.order);
  return ranked.map((row, i) => ({ ...row, rank: i + 1 }));
}

const num = (v) => v.toLocaleString("ko-KR");
const rangeText = (r, unit) => (r.low === r.high ? `+${num(r.low)}${unit}` : `+${num(r.low)}~${num(r.high)}${unit}`);
/** 생활인구 변화 문구(월) — 전망 머리 숫자와 세부 비교 표가 같은 문구를 쓴다. */
export const livingText = (r) => `월 ${rangeText(r, "명")}`;
/** 정주인구 변화 문구(연) — 계산값이면 범위, 방문형 합성 값이면 한 값. */
export const settleText = (o) => (o.settleSynthetic ? (o.settleValue ? `연 +${num(o.settleValue)}명` : "해당 없음") : `연 ${rangeText(o.settle, "명")}`);

const round2 = (v) => Math.round(v * 100) / 100;
const quarterLabel = (idx) => `${String(Math.floor(idx / 4)).slice(2)}년 ${(idx % 4) + 1}분기`;
const OBSERVED_QUARTERS = 4; // 가로축에 그리는 최근 관측 분기 수
const RAMP_QUARTERS = 4; // 효과가 100%에 닿는 데 걸리는 분기(25·50·75·100%)
const PERSONS_PER_MAN = 10000; // 효과는 명, 그래프 단위는 만 명
const AXIS_PAD = 0.15; // 세로축 위아래 여유(값 범위 대비)

/**
 * 인구 전망 — 관측 마지막 4분기 + 향후 12분기. 방법과 가정(화면에는 쓰지 않는다):
 *  - 정책 미반영: 같은 분기 전년 값 + 그 해의 증감. 증감은 평균 연간 증감 d(관측 구간에서 4분기 차이 나는 쌍들의 평균)를 1년차 d, 2년차 d/2, 3년차 d/4로 줄여 쓴다(추세 감쇠).
 *    예측값끼리 이어 12분기까지, 쌍이 없으면 마지막 관측값 유지.
 *  - 정책 반영: 예측 구간에서만 미반영 + 효과 중간값(low·high 평균)을 첫 4분기에 25·50·75·100%로 늘리고 그 뒤 유지. 효과가 null이면 범례 항목만 두고 값은 null.
 * @param {{ unit: string, points: {year: number, quarter: number, registered: number, stay: number}[] }} data 공표 분기 값(만 명)
 * @param {(p: object) => number} valueOf 그릴 값(생활인구 = 등록+체류, 정주인구 = 등록)
 * @param {{low: number, high: number}|null} effect 고른 방안의 효과(명), 없으면 null
 * @param {string} reflectLabel 정책 반영 선 범례 이름
 */
export function populationForecast(data, valueOf, effect, reflectLabel = REFLECT_LABEL) {
  const byIdx = new Map(data.points.map((p) => [p.year * 4 + p.quarter - 1, valueOf(p)]));
  const idxs = [...byIdx.keys()].sort((a, b) => a - b);
  const last = idxs[idxs.length - 1];
  const diffs = idxs.filter((i) => byIdx.has(i - 4)).map((i) => byIdx.get(i) - byIdx.get(i - 4));
  const annual = diffs.length ? diffs.reduce((a, b) => a + b, 0) / diffs.length : null;
  const shown = idxs.slice(-OBSERVED_QUARTERS);
  const base = new Map(byIdx);
  const future = Array.from({ length: FORECAST_QUARTERS }, (_, k) => last + k + 1);
  future.forEach((i, k) => base.set(i, annual === null ? byIdx.get(last) : (base.get(i - 4) ?? byIdx.get(last)) + annual / 2 ** Math.floor(k / 4)));
  const axis = [...shown, ...future];
  const mid = effect ? (effect.low + effect.high) / 2 / PERSONS_PER_MAN : 0;
  const bounds = effect ? [effect.low / PERSONS_PER_MAN, effect.high / PERSONS_PER_MAN] : [0, 0];
  const ramp = (i) => Math.min(Math.max(i - last, 0), RAMP_QUARTERS) / RAMP_QUARTERS;
  const series = [{ label: "정책 미반영", values: axis.map((i) => round2(base.get(i))) }];
  // 두 모드 같은 범례(두 항목) — 효과가 없으면(실데이터 표시) 반영 선 값은 만들지 않고 빈 값(null)으로 둔다.
  series.push(effect ? {
    label: reflectLabel,
    values: axis.map((i) => round2(base.get(i) + mid * ramp(i))),
    // 예측 구간의 하한~상한(툴팁용) — 관측 구간은 null
    ranges: axis.map((i) => (i > last ? bounds.map((b) => round2(base.get(i) + b * ramp(i))) : null))
  } : { label: reflectLabel, values: axis.map(() => null) });
  const gap = effect ? round2(series[1].values[axis.length - 1] - series[0].values[axis.length - 1]) : 0;
  // 마지막 예측 분기의 반영-미반영 차이(0이면 null: 칠하기·값 표기 없음)
  return { unit: data.unit, labels: axis.map(quarterLabel), observed: shown.length, endDiff: gap || null, series };
}

export const REFLECT_LABEL = "정책 반영";

/**
 * 시설 짧은 이름 — 괄호 부분과 앞의 지역 이름을 뺀다. 예: "신안 이세돌바둑기념관(1도1뮤지엄)" → "이세돌바둑기념관".
 * @param {{ name: string }} item 시설
 * @param {string} [regionName] 지역 이름(예: 전남광주통합특별시 신안군)
 */
export function facilityShortName(item, regionName = "") {
  const region = String(regionName).split(" ").at(-1)?.replace(/[시군구]$/, "") ?? "";
  const words = item.name.replace(/\([^)]*\)/g, "").trim().split(/\s+/);
  return (region && words[0] === region && words.length > 1 ? words.slice(1) : words).join(" ");
}

/** 정책 짧은 이름 — 시설 짧은 이름 + 방안 유형. 예: "퍼플섬 운영 개선". */
export const policyShortName = (item, choice, regionName = "") => `${facilityShortName(item, regionName)} ${choice.planType}`;
