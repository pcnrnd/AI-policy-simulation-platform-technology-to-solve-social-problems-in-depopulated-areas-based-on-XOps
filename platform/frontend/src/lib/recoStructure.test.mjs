// 정책 추천 ④ 구조·문구·전망·보고서 연계 검사 (vitest 미도입 → node 직접 실행).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { FORECAST_QUARTERS, facilityRows, populationForecast } from "./facilityRecommendation.js";
import { OUTLOOK_OBSERVED, outlookBase, outlookChart } from "./recoOutlook.js";
import {
  CHART_ITEMS, GUIDE, businessName, chipItems, effectHeads, kpiText, overviewRows, reasonRows, reportHandoffOf, shortName, stageText
} from "./recoText.js";
import { buildReportBlocks, buildReportRows } from "./reportContent.js";

const here = dirname(fileURLToPath(import.meta.url));
const read = (...path) => readFileSync(join(here, "..", ...path), "utf-8");
const seed = JSON.parse(read("assets", "mock_data.json"));
const conversion = { ...seed.conversion, regions: {
  namwon: { stay_monthly: 250000, inflow_total: 6000, inflow_external: 3000, persons_per_household: 2 },
  shinan: { stay_monthly: 150000, inflow_total: 7500, inflow_external: 7000, persons_per_household: 2 }
} };
const region = (id) => seed.regions.find((r) => r.id === id);
const rows = Object.fromEntries(["namwon", "shinan"].map((id) => [id, facilityRows(id, seed.facilities, [], {}, conversion)]));
const inputs = { objective: "청년 정주 전환 극대화", budgetTotal: 600, welfare: 50, industry: 50, housing: 50, period: "2025-07~2026-06" };

// ── 정책 효과 전망: 관측 8분기(앞 분기 채우기는 데모만) → 현재 → 전망 12분기, 반영 후 = 반영 전 + 효과 × 도입 곡선 ──
// 아래 값은 API 응답과 모양만 같은 합성 숫자다(공표 값은 저장소에 두지 않는다).
const points = [[2024, 3], [2024, 4], [2025, 1], [2025, 2], [2025, 3], [2025, 4]].map(([year, quarter], i) => ({ year, quarter, registered: 7 - i * 0.05, stay: 20 + (i % 2) * 3 + i * 0.1 }));
const trends = { population: { unit: "만 명", points } };
const demoBase = outlookBase(trends, { backcast: true });
const realBase = outlookBase(trends, { backcast: false });
assert.equal(demoBase.obsLabels.length, OUTLOOK_OBSERVED);
assert.deepEqual(demoBase.obsLabels.slice(0, 3), ["24년 1분기", "24년 2분기", "24년 3분기"]);
assert.equal(realBase.obsLabels.length, points.length, "실데이터 표시는 있는 분기만(합성 없음)");
assert.deepEqual(realBase.charts[0].obs, points.map((p) => Math.round((p.registered + p.stay) * 10) / 10));
// 거꾸로 채운 분기 = 1년 뒤 같은 분기 ÷ (1 + 평균 전년 동기 변화율)
const g = [[4, 0], [5, 1]].map(([a, b]) => points[a].registered / points[b].registered - 1);
assert.equal(demoBase.charts[1].obs[0], Math.round((points[2].registered / (1 + (g[0] + g[1]) / 2)) * 10) / 10);
const fc = populationForecast(trends.population, (p) => p.registered, null);
assert.deepEqual(realBase.charts[1].base, fc.series[0].values.slice(fc.observed));
assert.equal(realBase.charts[1].fcLabels.length, FORECAST_QUARTERS);
assert.equal(outlookBase(null, { backcast: true }), null);
const pick = rows.namwon[0];
const pop = outlookChart(demoBase, "population", pick);
assert.equal(pop.labels.length, OUTLOOK_OBSERVED + FORECAST_QUARTERS);
pop.reflect.forEach((v, k) => assert.ok(Math.abs(v - (pop.base[k] + pick.forecast.livingPath[k + 1] / 10000)) < 1e-9));
assert.deepEqual([pop.increase, pop.end], [pick.forecast.living, pop.reflect.at(-1)]);
const plain = outlookChart(realBase, "settle", null);
assert.deepEqual([plain.reflect, plain.increase, plain.end], [null, null, plain.base.at(-1)]);
assert.deepEqual(outlookChart(null, "population", pick), { id: "population", title: "생활인구 전망", empty: true });

// ── 화면 문구: 상세(사업 개요·선정 사유), 전망 머리 숫자, 칩, 보고서 연계 ──
const regionName = region("namwon").name;
assert.deepEqual(overviewRows(pick).map((r) => r[0]), ["위치", "운영 현황", "소요 예산"]);
assert.deepEqual(reasonRows(pick, inputs).map((r) => r[0]), ["목적함수", "반영 조건", "분석 기간", "예산 ±20%", "방안 기준"]);
assert.equal(reasonRows(pick, inputs)[0][1], "청년 정주 전환 극대화");
assert.equal(reasonRows(pick, inputs)[2][1], "2025-07~2026-06");
assert.equal(reasonRows(pick, { ...inputs, period: null })[2][1], GUIDE.trends);
// 사업이 없으면(실데이터 표시) 같은 자리·같은 이름에 행동 안내.
assert.deepEqual(overviewRows(null).map((r) => r[1]), [GUIDE.pending, GUIDE.pending, GUIDE.pending]);
assert.deepEqual(reasonRows(null, inputs).map((r) => r[0]), ["목적함수", "반영 조건", "분석 기간", "예산 ±20%", "방안 기준"]);
assert.deepEqual(effectHeads(null), { util: null, population: null, settle: null });
const heads = effectHeads(pick);
const c = pick.options.find((o) => o.chosen);
assert.equal(heads.util.value, `${pick.util}% → ${c.utilAfter.low}~${c.utilAfter.high}%`);
assert.equal(heads.population.sub, `목표 월 +${c.living.low.toLocaleString("ko-KR")}명 이상`); // 성과 목표 = 효과 하한
// 같은 효과 값은 전망 머리에만 — 상세(사업 개요·선정 사유)에는 효과 머리 값이 없다.
const detailText = [...overviewRows(pick), ...reasonRows(pick, inputs)].map((r) => r[1]).join(" ");
for (const h of Object.values(heads)) assert.ok(!detailText.includes(h.value), h.value);
// 칩: 1~3순위, 고른 사업이 4·5순위면 하나 더.
assert.deepEqual(chipItems(rows.namwon, rows.namwon[0]).map((r) => r.rank), [1, 2, 3]);
assert.deepEqual(chipItems(rows.namwon, rows.namwon[4]).map((r) => r.rank), [1, 2, 3, 5]);
assert.equal(CHART_ITEMS, 5);
assert.equal(shortName(rows.shinan.find((r) => r.id === "fac_sa_museum"), region("shinan").name), "이세돌바둑기념관");
assert.equal(stageText(seed.facilities.find((f) => f.id === "fac_sa_fossil")), "운영 중 · 연면적 2,190㎡ · 2025년 월평균 입장 736명");

// 보고서로 보내기: 시행계획 중점과제 한 줄 + 기금 투자계획 사업 한 건, 고정 비율(0.95) 없이 고른 사업 계산값.
const handoff = reportHandoffOf(pick, region("namwon"), inputs);
assert.equal(handoff.regionId, "namwon");
assert.equal(handoff.task, `${businessName(pick, regionName)} — ${kpiText(c)}`);
assert.deepEqual(handoff.fund.map((r) => r[0]), ["사업명", "위치", "사업 내용", "기대 효과", "성과지표", "사업비", "선정 근거"]);
assert.doesNotMatch(read("lib", "reportContent.js") + read("pages", "ReporterPage.jsx"), /0\.95/);
const template = seed.report_templates.find((t) => !["template_task_order", "template_vital_population", "template_smartfarm", "template_settlement"].includes(t.id)) ?? seed.report_templates[0];
const blocksOf = (extra) => buildReportBlocks(region("namwon"), template, { allowSeed: true, generatedAt: "2026-11-18T08:00:00Z", ...extra });
const withHandoff = blocksOf({ handoff });
const text = (blocks) => blocks.map((b) => b.text ?? (b.items ?? []).join(" ")).join("\n");
assert.match(text(withHandoff), /시행계획 중점과제/);
assert.ok(text(withHandoff).includes(handoff.task));
assert.ok(handoff.fund.every(([label, value]) => text(withHandoff).includes(`${label}: ${value}`)));
assert.doesNotMatch(text(blocksOf({})), /시행계획 중점과제|정책적용|정책 적용 시 10년/);
assert.doesNotMatch(text(blocksOf({ handoff: { ...handoff, regionId: "shinan" } })), /시행계획 중점과제/, "다른 지역 사업은 싣지 않는다");
const xlsx = buildReportRows(region("namwon"), template, { allowSeed: true, generatedAt: "2026-11-18T08:00:00Z", handoff });
assert.ok(xlsx.some((r) => r[1] === "추천 사업 정주인구 변화(연)" && r[2] === c.settleValue));
assert.ok(!buildReportRows(region("namwon"), template, { allowSeed: true }).some((r) => String(r[1]).includes("정책적용")));

// ── 화면 문구 규칙: 출처 각주·괄호 꼬리표·설명 부제·모델 이름·내부 표기 없음, 기업·기관명·개인정보·내부 주소 없음 ──
const screen = [
  ...Object.values(GUIDE),
  ...["namwon", "shinan"].flatMap((id) => rows[id].flatMap((row) => {
    const name = region(id).name;
    return [shortName(row, name), businessName(row, name), ...overviewRows(row).map((r) => r[1]), ...reasonRows(row, inputs).map((r) => r[1]),
      ...Object.values(effectHeads(row)).flatMap((h) => [h.value, h.sub])];
  }))
];
const COMPONENTS = ["PolicyRecommendation.jsx", "reco/RecoResult.jsx", "reco/EffectOutlook.jsx", "reco/OutlookChart.jsx", "reco/FacilityPlanTable.jsx", "reco/ReportHandoff.jsx"];
const componentSrc = COMPONENTS.map((f) => read("components", ...f.split("/"))).join("\n");
const literals = [...componentSrc.matchAll(/>([^<>{}\n]*[가-힣][^<>{}\n]*)</g), ...componentSrc.matchAll(/"([^"\n]*[가-힣][^"\n]*)"/g)].map((m) => m[1]);
const FORBIDDEN = /데모|예시|합성|시드|mock|synthetic|demo|RICE|가중|비율 범위|로그 척도|모델|출처|기준:|\(공식|\(연 환산\)/i;
const ORG_OR_PERSONAL = /주식회사|㈜|\bKT\b|BC카드|@|\d{2,3}-\d{3,4}-\d{4}|\b\d{1,3}(\.\d{1,3}){3}\b|localhost/i;
for (const t of [...screen, ...literals]) {
  assert.doesNotMatch(t, FORBIDDEN, t);
  assert.doesNotMatch(t, ORG_OR_PERSONAL, t);
}

// ── 구조: 구역 4개(두 모드 같은 구성), 선택 하나로 지도·차트·칩·표·상세·전망 연동 ──
const panel = read("components", "PolicyRecommendation.jsx");
const order = ["<RecoResult", "<EffectOutlook", "<FacilityPlanTable", "<ReportHandoff"].map((tag) => panel.indexOf(tag));
assert.ok(order.every((i, k) => i > 0 && (k === 0 || i > order[k - 1])), "추천 결과 → 정책 효과 전망 → 세부 비교 → 보고서로 보내기");
assert.doesNotMatch(panel, /demo\s*(\?|&&)\s*\(?\s*</, "모드에 따라 다른 구역을 그리지 않는다");
const slots = ["sim.report.result", "sim.report.outlook", "sim.report.charts", "sim.report.compare", "sim.report.handoff"];
for (const slot of slots) assert.equal(componentSrc.split(`data-slot="${slot}"`).length - 1, 1, slot);
for (const title of ["추천 결과", "정책 효과 전망", "세부 비교", "보고서로 보내기"]) assert.ok(componentSrc.includes(`title="${title}"`), title);
const page = read("pages", "SimulatorPage.jsx");
// 데모 ④는 ③ 설정(목적함수·총 예산·정책 강조·시설 변수)이 바뀌면 바로 다시 계산, 실데이터는 시설 대장이 없어 빈 행(결정 72).
assert.match(page, /const recoLive = useMemo\(\(\) => \(demo && analysisDone && recommendation \? deriveFacilities\(currentRegion\)\.items : \[\]\),/);
assert.match(page, /\[demo, analysisDone, recommendation, currentRegion, controlValues, regionStay, budgetTotal, welfareWeight, industryWeight, housingWeight\]/);
assert.match(page, /const recoContext = \{ objective: currentRegion\.case\?\.simulation\?\.objective \?\? "", budgetTotal,\s*policyWeights: \{ welfare: welfareWeight, industry: industryWeight, housing: housingWeight \} \};/);
// ③ 10년 예측 ↔ ④: 정책 적용 선 = ③ 정책 계산 + 고른 사업 정주 효과(도입 곡선), 카드에 "고른 사업 반영분".
assert.match(page, /const simTrend = useMemo\(\(\) => policyTrend\.map\(\(v, t\) => v \+ recoAdd\[t\]\)/);
assert.match(page, /recoSelected\?\.forecast\?\.settlePath\[Math\.min\(4 \* \(t \+ 1\)/);
assert.match(page, />고른 사업 반영분</);
assert.equal(seed.regions.find((r) => r.id === "namwon").case.simulation.factors.supply, "전환 가능 시설 수용 규모");
assert.match(page, /reco=\{\{ items: recoItems\.slice\(0, CHART_ITEMS\), selected: recoSelected, onPick: setRecoPick \}\}/); // 지도 추천 층
assert.match(page, /selected=\{recoSelected\}\s*onPick=\{setRecoPick\}/);
const map = read("components", "SimulatorMap.jsx");
assert.match(map, /\.on\("click", \(\) => onPick\(it\.id\)\)/);
assert.match(map, /사용률 \$\{it\.util\}% → \$\{pctRange\(c\.utilAfter\)\}/);
assert.match(map, /OpenStreetMap/); // 좌표 출처는 접히는 ⓘ 안
for (const f of ["reco/RecoResult.jsx", "reco/EffectOutlook.jsx", "reco/FacilityPlanTable.jsx"]) assert.match(read("components", ...f.split("/")), /onPick\(/, f);

// ③ 분포 카드: 전문 용어 제목·축, ⓘ에 산식·등급 구간(RISK_GRADES 값 그대로)·점 색의 뜻, 기관 이름 없음.
assert.match(page, /title=\{<>지방소멸위험지수·인구 규모 분포<InfoTip text=\{RISK_INDEX_TIP\}/);
assert.match(page, /text: "지방소멸위험지수"/);
assert.match(page, /text: "인구 규모\(명\)"/);
const tip = page.slice(page.indexOf("const RISK_INDEX_TIP"), page.indexOf("const SIM_DEBOUNCE_MS"));
assert.match(tip, /20~39세 여성 인구 ÷ 65세 이상 인구/);
assert.match(tip, /RISK_GRADES\.map\(\(g\) => `\$\{g\.label\} \$\{g\.rangeLabel\}`\)/);
assert.match(tip, /회색 인구감소 지역, 주황 시범 지자체, 파랑 선택 지자체/);
assert.doesNotMatch(tip, /정보원|연구원|통계청|국가데이터처|행정안전부/);

console.log("recoStructure:전망 관측 8분기·앞 분기 채우기(데모만)·반영 후 = 반영 전 + 효과 추이·상세 두 칸·머리 숫자 한 번·칩·보고서 연계(0.95 제거)·화면 문구·기관명·개인정보 0·구역 4개 구조·선택 연동 PASS");
