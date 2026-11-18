// 정책 추천 ④ 화면·보고서 문구 — 추천 행(facilityRows)을 화면 글자로 바꾼다. 두 모드 같은 자리, 값이 없으면 행동 안내.
import { facilityShortName, livingText, settleText } from "./facilityRecommendation.js";

export const CHART_ITEMS = 5; // 추천 지수 차트에 그리는 순위 수(= 고를 수 있는 사업)
export const GUIDE = {
  registry: "시설 대장을 등록하면 추천 사업이 표시됩니다.",
  pending: "시설 대장 등록 후 산정",
  usage: "시설 이용 실적을 등록하면 표시됩니다.",
  trends: "생활인구 통계를 연결하면 표시됩니다.",
  derive: "[정책 추천 도출]을 누르면 표시됩니다.",
  handoff: "추천 사업이 있어야 보고서로 보낼 수 있습니다."
};

const num = (v) => Number(v).toLocaleString("ko-KR");
const isSource = (basis) => typeof basis === "string" && basis.startsWith("source");
const STAGE_WORD = { 운영: "운영 중", 조성: "조성 중", 계획: "계획" };

export const chosenOf = (item) => item?.options?.find((o) => o.chosen) ?? null;
/** 방안 문구의 "검토" 꼬리를 뗀다(화면은 사업 이름으로 읽힌다). */
export const noReview = (text) => String(text).replace(/\s*검토$/, "");
export const shortName = (item, regionName) => facilityShortName(item, regionName);
/** 사업 이름 — 시설 짧은 이름 → 고른 방안 문구. */
export const businessName = (item, regionName) => {
  const c = chosenOf(item);
  return c ? `${shortName(item, regionName)} → ${noReview(c.text)}` : item.name;
};
export const stageWord = (f) => (isSource(f.basis?.stage) && f.stage ? STAGE_WORD[f.stage] ?? f.stage : "");

/** 운영 현황 — 운영 단계와 출처 규모(수용·호수·연면적·월평균 입장). */
export function stageText(f) {
  const scale = [];
  if (Number.isFinite(f.capacity_people) && isSource(f.basis?.capacity_people)) {
    const kind = f.capacity_kind ?? "";
    scale.push(kind.startsWith("일 최대 방문") ? `하루 최대 방문 ${num(f.capacity_people)}명` : kind.includes("유치 목표") ? `유치 목표 ${num(f.capacity_people)}명` : `수용 ${num(f.capacity_people)}명`);
  }
  if (Number.isFinite(f.supply_units) && isSource(f.basis?.supply_units)) scale.push(`${num(f.supply_units)}호`);
  if (Number.isFinite(f.scale_area_m2) && isSource(f.basis?.scale_area_m2)) scale.push(`연면적 ${num(f.scale_area_m2)}㎡`);
  if (Number.isFinite(f.visitors_monthly) && isSource(f.basis?.visitors_monthly)) scale.push(`2025년 월평균 입장 ${num(f.visitors_monthly)}명`);
  return [stageWord(f), ...scale].filter(Boolean).join(" · ");
}
export const locationText = (f) => (f.address && isSource(f.basis?.address) ? f.address : "");

export const pctRange = (r) => (r.low === r.high ? `${r.low}%` : `${r.low}~${r.high}%`);
const eok = (v) => (v < 10 ? v.toFixed(1) : String(Math.round(v)));
export const budgetText = (b) => (b ? `${eok(b.low)}~${eok(b.high)}억 원` : "");
export const utilText = (item, o) => `${item.util}% → ${pctRange(o.utilAfter)}`;
export const sensText = (o) => `월 +${num(o.budgetSens.low)}~${num(o.budgetSens.high)}명`;
const isSettlePlan = (o) => o.kind === "house" || o.kind === "job";
/** 성과 목표 = 효과 하한(사용률·정주 또는 생활인구). */
export const kpiText = (c) => `사용률 ${c.utilAfter.low}% 이상 · ${isSettlePlan(c) ? `연 정주인구 +${num(c.settle.low)}명` : `월 생활인구 +${num(c.living.low)}명`} 이상`;

/**
 * 선정 사유 — 시뮬레이션 입력을 조절하면 달라지는 근거와 방안 기준. 값이 없으면 같은 자리에 행동 안내.
 * @param {object|null} item 고른 추천 행
 * @param {{ objective: string, budgetTotal: number, welfare: number, industry: number, housing: number, period: string|null }} inputs
 */
export function reasonRows(item, inputs) {
  const c = chosenOf(item);
  return [
    ["목적함수", inputs.objective || GUIDE.pending],
    ["반영 조건", `총 예산 ${num(inputs.budgetTotal)}억 원 · 정책 강조 복지 ${inputs.welfare} · 산업 ${inputs.industry} · 주거 ${inputs.housing}`],
    ["분석 기간", inputs.period ?? GUIDE.trends],
    ["예산 ±20%", c ? `생활인구 ${sensText(c)}` : GUIDE.pending, Boolean(c)],
    ["방안 기준", item?.choiceRule ?? GUIDE.pending]
  ];
}
/** 사업 개요 — 위치·운영 현황·소요 예산. */
export function overviewRows(item) {
  const f = item?.facility;
  return [
    ["위치", f ? locationText(f) : GUIDE.pending],
    ["운영 현황", f ? stageText(f) : GUIDE.pending],
    ["소요 예산", item?.budget ? budgetText(item.budget) : GUIDE.pending, Boolean(item?.budget)]
  ];
}

/** 정책 효과 전망 머리 — [사용률, 생활인구, 정주인구] 각 { value, sub }. */
export function effectHeads(item) {
  const c = chosenOf(item);
  if (!c) return { util: null, population: null, settle: null };
  return {
    util: { value: utilText(item, c), sub: `목표 ${c.utilAfter.low}% 이상` },
    population: { value: livingText(c.living), sub: `목표 월 +${num(c.living.low)}명 이상` },
    settle: { value: settleText(c), sub: c.settle ? `목표 연 +${num(c.settle.low)}명 이상` : "" }
  };
}

/** 반영 정책 칩 — 1~3순위(고른 사업이 4·5순위면 그 사업을 하나 더). */
export const chipItems = (items, selected) => items.slice(0, 3).concat(selected && selected.rank > 3 ? [selected] : []);

/**
 * 보고서로 넘길 고른 사업 — 시행계획 중점과제 한 줄과 기금 투자계획 사업 한 건.
 * @param {object} item 고른 추천 행
 * @param {object} region 지역(id·name)
 * @param {object} inputs reasonRows와 같은 시뮬레이션 입력
 */
export function reportHandoffOf(item, region, inputs) {
  const c = chosenOf(item);
  const name = businessName(item, region.name);
  const sens = `총 예산 ${num(inputs.budgetTotal)}억 원 기준, 예산 ±20%일 때 생활인구 ${sensText(c)} · ${item.choiceRule}`;
  return {
    regionId: region.id,
    business: name,
    task: `${name} — ${kpiText(c)}`,
    fund: [
      ["사업명", name],
      ["위치", locationText(item.facility)],
      ["사업 내용", `${item.name} · ${stageText(item.facility)} · ${c.planType} · ${noReview(c.text)}`],
      ["기대 효과", `현재 사용률 ${item.util}% → 전환 후 ${pctRange(c.utilAfter)} · 생활인구 ${livingText(c.living)} · 정주 ${settleText(c)}`],
      ["성과지표", kpiText(c)],
      ["사업비", budgetText(item.budget)],
      ["선정 근거", sens]
    ],
    effect: { living: c.living, settleValue: c.settleValue, settle: settleText(c) }
  };
}
