// 시설 × 전환 방안 추천 — 시설마다 전환 방안 후보 2~3개의 효과 범위(사용률·생활인구·정주인구)를 계산하고,
// 생활인구 효과 중심의 추천 지수(0~100)로 시설 순위와 시설별 방안을 정한다.
// 화면에는 결과와 근거 문장만 보이고 계산식·비율·가중치는 보이지 않는다(계산 방법은 이 파일·시드 conversion.synthetic.method·보고서에만).

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

// 추천 지수 = 생활인구 효과(중심)·정주 효과·추진 여건의 가중 평균(각 0~100). 방안 선택도 같은 비중의 앞 두 항으로 한다.
// 생활인구 효과 점수는 지역 상한(월평균 체류인구 × 상한 비율) 대비 로그 척도, 정주 효과 점수는 지역 연간 다른 시군구 전입의 1%를 만점으로 둔다.
// 고정 가중치 — 시설 대장 실적(2단계)이 들어오면 실적 기반으로 바꾼다.
const INDEX_WEIGHTS = { living: 0.6, settle: 0.2, readiness: 0.2 };
const SETTLE_FULL_SHARE = 0.01;
// 판단: 시설 안 최고 방안이 추천, 최고 대비 이 비율 이상이면 검토, 그 아래는 보류.
const REVIEW_RATIO = 0.8;

// 표 정렬 기준 — rank(추천 순위)가 기본. 값이 없는 시설은 방향과 관계없이 뒤에 둔다.
export const SORT_CRITERIA = {
  rank: { label: "순위", unit: "", ascending: true, text: "추천 순위" },
  utilization: { label: "활용도", unit: "%", ascending: true, text: "활용도 낮은 순" },
  budget: { label: "운영 예산", unit: "억 원", ascending: false, text: "운영 예산 큰 순" }
};
export const DEFAULT_CRITERION = "rank";

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
 * 시설 수용 상한(월 방문 인원). 출처 수용 인원이 등록형이면 그 인원 × 등록 회전, 아니면
 * (출처 연면적 또는 규모 등급 대표 연면적) ÷ 1인당 면적(법정 수용인원 산정 기준) × 일 회전 × 월 운영일.
 */
export function capacityCap(facility, rules) {
  const turnover = rules.capacity_turnover;
  if (Number.isFinite(facility.capacity_people)) {
    return facility.capacity_people * (facility.capacity_kind === "등록" ? turnover.enrolled : turnover.daily * turnover.open_days);
  }
  const area = Number.isFinite(facility.scale_area_m2) ? facility.scale_area_m2 : rules.capacity_grade_area_m2[scaleGrade(facility, rules)];
  return Math.round(area / rules.capacity_area_per_person_m2) * turnover.daily * turnover.open_days;
}

// 비교 가능 사례 — 같은 방안 유형·비슷한 지역 유형(섬·농촌·소도시)·인구감소지역, 그 시설 자신의 사례 제외. 시설·사례 속성만 본다.
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
const ESTIMATE_NOTE = { single: "같은 유형 시설 규모 기준 추정", reference: "유사 방안 사례 규모 기준 추정" };
// 기간이 짧은 행사 사례를 연간 효과로 쓸 때 근거 문장에 그 사실을 드러낸다.
const EVENT_NOTE = "·연 1회 행사 방문을 12개월로 나눈 값";
const isEvent = (c) => Number.isFinite(c.event_days);

// 방안 하나의 효과 범위 — 값마다 basis(source | synthetic)와 화면 근거 문장을 붙인다.
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
    return { utilization, living: { ...settle, basis: "synthetic" }, settle: { ...settle, basis: "synthetic" }, note: "전입 통계 기준 추정", cases: [], caseTier: "none" };
  }
  if (option.kind === "stay") {
    const vacant = useful ? housing.vacant : 0;
    const raw = span(vacant * rules.short_stay_per_unit[0], vacant * rules.short_stay_per_unit[1]);
    return { utilization, living: { ...span(Math.min(raw.low, cap), Math.min(raw.high, cap)), basis: "synthetic" }, settle: null, note: "공실 규모 기준 추정", cases: [], caseTier: "none" };
  }
  // 방문형: 비교 가능 사례마다 규모 보정 배수와 운영 중 시설의 증가분(신규 개관 총량 사례일 때)을 곱하고,
  // 시설 수용 상한·지역 상한 가운데 작은 값으로 자른다.
  const limit = Math.min(cap, capacityCap(facility, rules));
  const { tier, cases: matched } = comparableCases(option, facility, cases, rules);
  if (matched.length) {
    const ranges = matched.map((c) => {
      const scale = scaleRatio(facility, c, rules);
      const [incLow, incHigh] = c.opening === "new" && isOperating(facility) ? rules.operating_increment : [1, 1];
      return { low: c.monthly * scale * incLow, high: c.monthly * scale * incHigh, adjusted: scale !== 1 || incLow < 1 };
    });
    const low = Math.min(...ranges.map((r) => r.low)), high = Math.max(...ranges.map((r) => r.high));
    const adjusted = ranges.some((r) => r.adjusted) || high > limit;
    const event = matched.some(isEvent) ? EVENT_NOTE : "";
    const isRange = tier === "range";
    return {
      utilization,
      living: { ...span(Math.min(low, limit), Math.min(high, limit)), basis: isRange && !adjusted ? "source" : "synthetic" },
      settle: null,
      note: isRange ? `유사 사례 ${matched.length}곳 기준${adjusted ? "·규모 반영" : ""}${event}` : `${ESTIMATE_NOTE[tier]}${event}`,
      cases: matched.map((c) => c.id),
      caseTier: tier
    };
  }
  const [a, b] = rules.visit_share[option.planType];
  return {
    utilization,
    living: { ...span(Math.min(region.stay_monthly * a, limit), Math.min(region.stay_monthly * b, limit)), basis: "synthetic" },
    settle: null,
    note: "지역 생활인구 규모 기준 추정",
    cases: [],
    caseTier: "none"
  };
}

// 방안 점수(생활인구 중심 + 정주 보조, 0~100 척도의 가중 합) — 추천 지수에서 추진 여건만 뺀 부분.
function optionScore(effect, region, rules) {
  const living = logScore(mid(effect.living), region.stay_monthly * rules.visit_cap_share);
  const settle = effect.settle ? Math.min(100, (100 * mid(effect.settle)) / (region.inflow_external * SETTLE_FULL_SHARE)) : 0;
  return INDEX_WEIGHTS.living * living + INDEX_WEIGHTS.settle * settle;
}

// 시설 하나의 방안 비교(효과·판단) — 지역 통계가 없으면 null(화면에서 행동 안내).
function conversionOf(facility, housing, conversion) {
  const region = conversion?.regions?.[facility.region_id];
  const rules = conversion?.synthetic?.rules;
  if (!region || !rules) return null;
  const ctx = { region, cases: conversion.cases ?? [], rules };
  const options = optionCandidates(facility).map((option) => {
    const effect = optionEffect(option, facility, housing, ctx);
    return { ...option, ...effect, score: optionScore(effect, region, rules) };
  });
  const best = Math.max(...options.map((o) => o.score));
  const chosen = options.find((o) => o.score === best);
  return options.map((o) => ({
    ...o,
    chosen: o === chosen,
    verdict: o === chosen && o.living.high > 0 ? "추천" : o.score >= best * REVIEW_RATIO && o.living.high > 0 ? "검토" : "보류"
  }));
}

/**
 * 지역 시설 → 추천 행(순위 포함, 추천 지수 높은 순). 연결된 조절변수 값은 해당 시설의 필드를 대신한다.
 * @param {string} regionId
 * @param {object[]} facilities 시설 자료(id·name·type·region_id·stage·operating_budget·occupancy_rate·supply_units·use_plan·reco_inputs)
 * @param {object[]} controls 시설 조절변수(facility_id·field가 있으면 그 시설 값에 연결)
 * @param {Record<string, number>} controlValues
 * @param {object|null} conversion 시드 conversion — 지역 통계(regions)·출처 사례(cases)·합성 규칙(synthetic)
 */
export function facilityRows(regionId, facilities, controls = [], controlValues = {}, conversion = null) {
  const linked = linkedControls(controls);
  const rows = (facilities ?? []).filter((f) => f.region_id === regionId).map((f) => {
    const adjusted = { ...f };
    linked.filter((c) => c.facility_id === f.id).forEach((c) => { adjusted[c.field] = valueOf(c, controlValues); });
    const housing = housingOf(f, adjusted.supply_units);
    const options = conversionOf(f, housing, conversion);
    const choice = options?.find((o) => o.chosen) ?? null;
    const fallback = optionCandidates(f)[0];
    const readiness = f.reco_inputs?.readiness;
    const index = choice && Number.isFinite(readiness) ? Math.round(choice.score + INDEX_WEIGHTS.readiness * readiness) : null;
    return {
      id: f.id,
      name: f.name,
      type: f.type,
      planType: (choice ?? fallback).planType,
      plan: (choice ?? fallback).text,
      options,
      // 사례 성과 그래프는 고른 방안의 근거 문장과 같은 사례 묶음(자기 사례 제외·비교 가능 사례만)을 그린다.
      planCases: choice ? (conversion.cases ?? []).filter((c) => choice.cases.includes(c.id)) : [],
      index,
      utilization: housing === null ? null : Math.round(housing.occupancy),
      utilizationLabel: housing === null ? null : "입주율",
      budget: Number.isFinite(f.operating_budget) ? f.operating_budget : null
    };
  });
  // 추천 순위: 지수 높은 순(값 없는 시설은 뒤, 같은 값은 시설 자료 순서).
  const byIndex = [...rows].sort((a, b) => ((a.index === null) - (b.index === null)) || (b.index ?? 0) - (a.index ?? 0));
  return byIndex.map((row, i) => ({ ...row, rank: i + 1 }));
}

/** 선택한 기준 하나로 정렬(행 필드 이름 = 기준 키) — 같은 값은 입력 순서를 유지한다. */
export function sortFacilityRows(rows, criterionKey = DEFAULT_CRITERION) {
  const { ascending } = SORT_CRITERIA[criterionKey];
  const key = criterionKey;
  return [...rows].sort((a, b) => {
    if (a[key] === null || b[key] === null) return (a[key] === null) - (b[key] === null);
    return ascending ? a[key] - b[key] : b[key] - a[key];
  });
}

const utilizationText = (row) => (row.utilization === null ? "이용실적 등록" : `${row.utilizationLabel} ${row.utilization}%`);
const budgetText = (row) => (row.budget === null ? "예산 자료 등록" : row.budget.toLocaleString("ko-KR"));
const num = (v) => v.toLocaleString("ko-KR");
const rangeText = (r, unit) => (r.low === r.high ? `+${num(r.low)}${unit}` : `+${num(r.low)}~${num(r.high)}${unit}`);
/** 생활인구 변화 문구(월) — 카드 효과 줄과 비교 표가 같은 문구를 쓴다. */
export const livingText = (r) => `월 ${rangeText(r, "명")}`;

/**
 * 추천 행 → 추천 패널 표시 형식(실데이터 민감도 패널과 같은 자리·같은 구조).
 * items(Top3·주 그래프)는 항상 추천 순위, tableItems(표)는 사용자가 고른 기준으로 정렬한다.
 */
export function facilityRecommendationView(rows, criterionKey = DEFAULT_CRITERION) {
  const criterion = SORT_CRITERIA[criterionKey];
  const items = sortFacilityRows(rows, "rank").map((row) => {
    const choice = row.options?.find((o) => o.chosen) ?? null;
    return {
      id: row.id,
      name: row.name,
      chartLabel: `${row.name} · ${row.planType}`,
      category: row.plan,
      planType: row.planType,
      type: row.type,
      rank: row.rank,
      utilization: row.utilization,
      budget: row.budget,
      score: row.index,
      options: row.options,
      planCases: row.planCases,
      caseTier: choice?.caseTier ?? "none",
      // Top3 카드 효과 줄: 고른 방안 유형과 기대 생활인구 변화.
      effectLine: choice ? `${choice.planType} · 생활인구 ${livingText(choice.living)}` : "",
      // Top3 카드 값 줄: 큰 글씨는 숫자만, 지표 이름은 작은 글씨 칸.
      scoreText: row.index === null ? "" : String(row.index),
      scoreUnit: row.index === null ? "자료 등록 후 산정" : "추천 지수",
      cells: [row.plan, utilizationText(row), "사용 승인 후 확인", "투자 대장 확인", "기준일 확인"],
      scoreCell: budgetText(row),
      missing: row.index === null
    };
  });
  return {
    items,
    tableItems: sortFacilityRows(items, criterionKey),
    criterion: criterionKey,
    title: "시설 활용 추천",
    scoreLabel: "추천 지수(0~100)",
    scoreMax: 100,
    scoreUnit: "",
    // 표 9열 = 순위 + columns 6 + 숫자 열(scoreColumn) 1 + 검토 구분 — 기존 추천 표와 같은 열 수·같은 숫자 열 자리.
    rankColumn: { label: "순위", sortKey: "rank" },
    columns: [
      { label: "시설" },
      { label: "활용 방안" },
      { label: "활용도(주거 입주율·개방 시설 이용률)", sortKey: "utilization" },
      { label: "생활인구 관련 관측(외지인 방문·소비)" },
      { label: "기금 투자 여부" },
      { label: "자료 기준일" }
    ],
    scoreColumn: { label: "운영 예산(억 원)", sortKey: "budget" },
    rankText: (i) => `${i + 1}순위`,
    statusLabel: "검토 구분",
    // 검토 구분은 표 정렬과 관계없이 추천 순위로 정한다.
    status: (item) => (item.missing ? { text: "자료 등록 후 검토", strong: false } : { text: item.rank <= 3 ? "우선 검토" : "추가 검토", strong: item.rank <= 3 }),
    caption: `시설별 활용 방안과 관측값 — 정렬 기준 ${criterion.text}`,
    basisTip: "전환 뒤 기대 생활인구 효과가 큰 방안을 중심으로 정주 효과와 추진 여건을 함께 보아 순위를 정합니다.",
    footnote: `※ 순위: 추천 지수(0~100) 높은 순 · 표 정렬 기준: ${criterion.text}(순위·활용도·운영 예산 열 제목으로 변경) · 운영 예산은 시설별 관측값이며 총 예산과 합산하지 않습니다.`
  };
}

// 전환 방안 비교(순위 그래프 아래) — 두 모드가 같은 표 구조를 갖는다. 값이 없으면 같은 자리에 행동 안내만.
export const COMPARE_COLUMNS = ["전환 방안", "사용률 변화", "생활인구 변화", "정주인구 변화", "근거", "판단"];
const COMPARE_GUIDE = "시설 자료를 등록하면 전환 방안을 비교합니다.";

/** 고른 시설의 전환 방안 비교 표시 형식 — item이 없거나 방안이 없으면 행동 안내. */
export function conversionCompareView(item) {
  const options = item?.options ?? [];
  if (!options.length) return { title: "전환 방안 비교", scope: "", rows: [], empty: COMPARE_GUIDE };
  return {
    title: "전환 방안 비교",
    scope: `${item.name} · 전환 뒤 변화 범위`,
    rows: options.map((o, i) => ({
      id: `${item.id}-${i}`,
      verdict: o.verdict,
      basis: { utilization: o.utilization.basis, living: o.living.basis, settle: o.settle?.basis ?? null },
      cells: [
        `${o.planType} · ${o.text}`,
        rangeText(o.utilization, "%p"),
        livingText(o.living),
        o.settle ? `연 ${rangeText(o.settle, "명")}` : "해당 없음",
        o.note,
        o.verdict
      ]
    })),
    empty: null
  };
}

// 근거 그래프 묶음(전환 방안 비교 아래) — 두 모드가 같은 세 자리를 갖는다. 값이 없으면 같은 자리에 행동 안내만.
export const EVIDENCE_CHART_SLOTS = [
  { id: "population", title: "생활인구 추이(공식 통계)", guide: "생활인구 통계를 연결하면 표시됩니다." },
  { id: "case", title: "유사 사례 성과", guide: "시설 이용실적을 등록하면 표시됩니다." },
  { id: "plans", title: "추천 활용 방안 구성", guide: "시설 자료를 등록하면 표시됩니다." }
];

// 시드 그래프 값 → 공통 형식(labels + series). 한 계열 그래프는 points로 적는다.
const seriesOf = (data) => (data.series
  ? { labels: data.labels, series: data.series }
  : { labels: data.points.map((p) => p.label), series: [{ label: data.label ?? "", values: data.points.map((p) => p.value) }] });

/**
 * 그래프 묶음 표시 형식. regionCharts(지역 출처 수치)와 items(추천 행)가 없으면 모든 자리가 행동 안내.
 * 사례 성과는 고른 시설(selected)의 추천 방안 근거와 같은 사례 묶음(출처 월평균 방문)으로 그린다. 그 묶음이 비면 같은 자리에 안내.
 * @param {object|null} regionCharts mock_data.recommendation_charts[지역] — { population }
 * @param {object[]|null} items facilityRecommendationView(...).items
 * @param {object|null} selected items 가운데 고른 시설(없으면 1순위)
 */
export function evidenceChartsView(regionCharts, items, selected = null) {
  const planCounts = PLAN_TYPES.map((type) => ({ label: type, value: (items ?? []).filter((i) => i.planType === type).length }))
    .filter((p) => p.value > 0);
  const pick = selected ?? items?.[0] ?? null;
  const cases = pick?.planCases ?? [];
  // 고른 방안의 근거가 사례가 아닌 추정이면(사례 묶음 0곳) 사례 그래프 자리에 그 사실만 적는다.
  const noCase = pick && Array.isArray(pick.planCases) && pick.options?.length ? "고른 방안과 같은 유형의 비교 가능한 사례가 없습니다." : null;
  // 상시 운영 사례는 월평균, 기간이 짧은 행사 사례는 행사 기간 방문객 원값을 그 기간과 함께 그린다(월평균으로 환산해 보이지 않는다).
  const hasEvent = cases.some((c) => Number.isFinite(c.event_days));
  const caseData = cases.length
    ? {
      scope: `${pick.planType} ${pick.caseTier === "range" ? "사례" : "참고 사례"} · ${hasEvent ? "기간별 방문" : "월평균 방문"}`,
      unit: "명",
      points: cases.map((c) => (Number.isFinite(c.event_days)
        ? { label: `${c.name}(축제 기간 ${c.event_days}일)`, value: c.event_visitors }
        : { label: `${c.name}(월평균)`, value: c.monthly })),
      label: hasEvent ? "방문객" : "월평균 방문",
      note: `출처: ${cases.map((c) => c.sources[0].title).join(" · ")}`
    }
    : null;
  const derived = planCounts.length
    ? { scope: `추천 시설 ${items.length}곳 · 활용 방안별 시설 수`, unit: "곳", points: planCounts, label: "시설 수", note: "기준: 위 추천 결과" }
    : null;
  const dataOf = { population: regionCharts?.population, case: caseData, plans: derived };
  return EVIDENCE_CHART_SLOTS.map((slot) => {
    const data = dataOf[slot.id] ?? null;
    const { labels, series } = data ? seriesOf(data) : { labels: [], series: [] };
    return {
      id: slot.id,
      title: slot.title,
      scope: data ? `${data.scope} (${data.unit})` : "",
      unit: data?.unit ?? "",
      labels,
      series,
      note: data ? (data.note ?? `출처: ${data.source_text}`) : "",
      empty: labels.length === 0 ? (slot.id === "case" && noCase ? noCase : slot.guide) : null
    };
  });
}
