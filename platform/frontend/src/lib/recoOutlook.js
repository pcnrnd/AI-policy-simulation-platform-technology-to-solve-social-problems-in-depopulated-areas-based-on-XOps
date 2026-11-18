// 정책 효과 전망 — 생활인구·정주인구 두 그래프의 값. 관측 8분기 → 현재 → 전망 12분기.
// 관측: 플랫폼 공표 통계 API(분기 등록·체류 인구, 만 명). 데모 표시만 관측이 8분기보다 짧으면 앞 분기를
//   전년 동기 평균 변화율로 거꾸로 채운다(합성). 실데이터 표시는 있는 분기만 그린다.
// 반영 전: populationForecast의 정책 미반영 값(추세 감쇠). 반영 후: 반영 전 + 고른 방안 효과 × 도입 곡선(recoSynthetic).
import { FORECAST_QUARTERS, populationForecast } from "./facilityRecommendation.js";

export const OUTLOOK_OBSERVED = 8;
const PERSONS_PER_MAN = 10000;
const round1 = (v) => Math.round(v * 10) / 10;
const keyOf = (p) => p.year * 4 + p.quarter;
const fromKey = (k) => { const year = Math.floor((k - 1) / 4); return { year, quarter: k - year * 4 }; };
const quarterText = (p) => `${String(p.year).slice(2)}년 ${p.quarter}분기`;

/** 두 그래프 자리 — 두 모드 같은 자리, 값이 없으면 같은 자리에 행동 안내. */
export const OUTLOOK_CHARTS = [
  { id: "population", title: "생활인구 전망", observed: (p) => round1(p.registered + p.stay), forecast: (p) => p.registered + p.stay, path: "livingPath" },
  { id: "settle", title: "정주인구 전망", observed: (p) => p.registered, forecast: (p) => p.registered, path: "settlePath" }
];

// 같은 분기 전년 대비 평균 변화율(관측 쌍이 없으면 null)
function meanGrowth(byKey, field) {
  const g = [...byKey.values()].filter((p) => byKey.has(keyOf(p) - 4)).map((p) => p[field] / byKey.get(keyOf(p) - 4)[field] - 1);
  return g.length ? g.reduce((a, b) => a + b, 0) / g.length : null;
}

// 앞 분기 거꾸로 채우기(데모 합성) — 1년 뒤 같은 분기 ÷ (1 + 평균 전년 동기 변화율)
function backcast(points, need) {
  const byKey = new Map(points.map((p) => [keyOf(p), p]));
  const gReg = meanGrowth(byKey, "registered");
  const gStay = meanGrowth(byKey, "stay");
  if (gReg === null || gStay === null) return [];
  const first = keyOf(points[0]);
  const out = [];
  for (let back = need; back >= 1; back -= 1) {
    const next = byKey.get(first - back + 4) ?? out.find((p) => keyOf(p) === first - back + 4);
    if (!next) continue;
    out.push({ ...fromKey(first - back), registered: round1(next.registered / (1 + gReg)), stay: round1(next.stay / (1 + gStay)) });
  }
  return out;
}

/**
 * 고른 사업과 관계없는 전망 바탕 — 관측 분기·반영 전 전망. 공표 통계가 없으면 null.
 * @param {{ population: { unit: string, points: object[] } | null } | null} trends usePublicTrends 결과
 * @param {{ backcast: boolean }} opts 데모 표시만 true(앞 분기 합성)
 */
export function outlookBase(trends, { backcast: fill }) {
  const data = trends?.population;
  if (!data?.points?.length) return null;
  const points = [...data.points].sort((a, b) => keyOf(a) - keyOf(b));
  const obsPoints = [...(fill ? backcast(points, Math.max(0, OUTLOOK_OBSERVED - points.length)) : []), ...points].slice(-OUTLOOK_OBSERVED);
  const charts = OUTLOOK_CHARTS.map((def) => {
    const fc = populationForecast({ unit: data.unit, points }, def.forecast, null);
    return { id: def.id, obs: obsPoints.map(def.observed), base: fc.series[0].values.slice(fc.observed), fcLabels: fc.labels.slice(fc.observed) };
  });
  return { obsLabels: obsPoints.map(quarterText), charts };
}

/**
 * 그래프 하나의 값 — 고른 사업이 있으면 반영 후 선(반영 전 + 효과 추이)과 끝 값·증가분.
 * @param {object|null} base outlookBase 결과
 * @param {string} id "population" | "settle"
 * @param {object|null} item 고른 추천 행(forecast 포함), 없으면 반영 후 선 없음
 */
export function outlookChart(base, id, item) {
  const def = OUTLOOK_CHARTS.find((c) => c.id === id);
  const chart = base?.charts.find((c) => c.id === id);
  if (!chart) return { id, title: def.title, empty: true };
  const path = item?.forecast?.[def.path] ?? null;
  const reflect = path ? chart.base.map((v, k) => v + path[k + 1] / PERSONS_PER_MAN) : null;
  return {
    id,
    title: def.title,
    empty: false,
    labels: [...base.obsLabels, ...chart.fcLabels],
    obs: chart.obs,
    base: chart.base,
    reflect,
    end: reflect ? reflect[reflect.length - 1] : chart.base[chart.base.length - 1],
    increase: path ? path[FORECAST_QUARTERS] : null,
    increases: path ? path.slice(1) : null
  };
}
