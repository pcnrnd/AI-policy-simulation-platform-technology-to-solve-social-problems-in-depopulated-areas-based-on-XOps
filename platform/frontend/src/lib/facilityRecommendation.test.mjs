// 시설 × 전환 방안 추천 순수 계산 검사 (vitest 미도입 → node 직접 실행).
// 계산 검사는 실명 지역 시드 통계와 독립된 합성 지역 통계로 한다(공표 통계 값을 시험 기대값에 쓰지 않는다).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  FORECAST_QUARTERS, INDEX_MAX, PLAN_TYPES, capacityCap, comparableCases, facilityRows, facilityShortName, linkedControls, livingText,
  optionCandidates, policyShortName, populationForecast, recoControlSig, scaleGrade, settleText, unlinkedControlNote, withRegionStay
} from "./facilityRecommendation.js";
import { adoption, baseUtil, budgetOf, effectPath, leaseUpOccupancy, rngFor, scaleFactor } from "./recoSynthetic.js";

const here = dirname(fileURLToPath(import.meta.url));
const read = (...path) => readFileSync(join(here, "..", ...path), "utf-8");
const seed = JSON.parse(read("assets", "mock_data.json"));
const SYNTH_REGIONS = {
  namwon: { stay_monthly: 250000, inflow_total: 6000, inflow_external: 3000, persons_per_household: 2 },
  shinan: { stay_monthly: 150000, inflow_total: 7500, inflow_external: 7000, persons_per_household: 2 }
};
const conversion = { ...seed.conversion, regions: SYNTH_REGIONS };
const rules = conversion.synthetic.rules;
const SEED = conversion.synthetic.random_seed;
const controlsOf = (id) => seed.regions.find((r) => r.id === id).case.simulation.controls;
const rowsOf = (id, values = {}, conv = conversion) => facilityRows(id, seed.facilities, controlsOf(id), values, conv);
const facility = (id) => seed.facilities.find((f) => f.id === id);
const namwon = rowsOf("namwon");
const shinan = rowsOf("shinan");
const all = [...namwon, ...shinan];
const chosen = (row) => row.options.find((o) => o.chosen);

// ── 시드: 실존 시설 10곳(지어낸 데모 이름·시설물이 아닌 사업은 뺐다), 신원 값은 출처, 좌표는 주소 검색 ──
assert.deepEqual(seed.facilities.map((f) => f.id), [
  "fac_nw_fiorium", "fac_nw_glocal", "fac_nw_dudong", "fac_sa_museum",
  "fac_sa_fossil", "fac_sa_eros", "fac_sa_birdcraft", "fac_sa_aphae_housing", "fac_nw_smartfarm", "fac_nw_bogeumjari"
]);
const BASIS = new Set(["source", "source_plan_target"]);
const validSource = (s) => /^https:\/\//.test(s.url) && /^\d{4}-\d{2}-\d{2}$/.test(s.checked) && s.title && ["facility", "location"].includes(s.scope);
for (const f of seed.facilities) {
  assert.ok(f.name && f.address && ["운영", "조성", "계획"].includes(f.stage), f.id);
  for (const [field, b] of Object.entries(f.basis)) {
    assert.ok(BASIS.has(b), `${f.id}.${field} 출처 값만`);
    assert.ok(field === "location" ? Number.isFinite(f.lat) && Number.isFinite(f.lng) : f[field] !== undefined, `${f.id} basis ${field}`);
  }
  assert.ok(f.sources.every(validSource), f.id);
  assert.ok(f.sources.some((s) => s.scope === "facility") && f.sources.some((s) => s.scope === "location"), `${f.id} 시설·좌표 출처`);
  assert.ok(Number.isFinite(f.lat) && Number.isFinite(f.lng) && ["road", "ri", "dong", "town", "approx"].includes(f.geo_precision), `${f.id} 좌표`);
  assert.ok(!("reco_inputs" in f) && !("operating_budget" in f), `${f.id} 근거 없는 예시값 없음`);
  if (f.use_plan) assert.ok(PLAN_TYPES.includes(f.use_plan.type), f.id);
}
assert.deepEqual(seed.facilities.filter((f) => f.open_space).map((f) => [f.id, f.open_space]), [["fac_nw_fiorium", "달빛정원·광장"], ["fac_sa_birdcraft", "새조각공원 8,500㎡"]]);
assert.deepEqual([facility("fac_sa_aphae_housing").supply_units, facility("fac_nw_bogeumjari").supply_units, facility("fac_sa_fossil").visitors_monthly], [90, 28, 736]);
assert.doesNotMatch(read("lib", "facilityRecommendation.js") + read("lib", "recoSynthetic.js"), /fac_(nw|sa)_/); // 시설 ID로 결과를 고르는 분기 없음

// 사례: 시설 전환 뒤 이용 실적만(섬 축제 3건 제외), 추이 참고 2건에 관광지점 실적, 세계화석광물박물관 사례 추가.
assert.deepEqual(conversion.cases.map((c) => c.id), ["case_fiorium", "case_gangju", "case_daesu", "case_bonsai", "case_yeonhong", "case_ypark", "case_fossil", "case_sangju"]);
assert.deepEqual(["case_yeonhong", "case_ypark", "case_fossil"].map((id) => conversion.cases.find((c) => c.id === id).monthly), [14083, 3951, 736]);
for (const c of conversion.cases) {
  assert.ok(!c.trend_only && Number.isFinite(c.monthly) && c.monthly > 0 && c.measure && c.method && c.basis === "source", c.id);
  assert.ok(c.depopulation_area === true && ["섬", "농촌", "소도시"].includes(c.region_type) && PLAN_TYPES.includes(c.plan_type), c.id);
  assert.ok(c.sources.length && c.sources.every((s) => /^https:\/\//.test(s.url) && /^\d{4}-\d{2}-\d{2}$/.test(s.checked)), c.id);
}
// 합성 규칙: 고정 시드, 규칙마다 생성 방법.
const synthetic = conversion.synthetic;
assert.equal(synthetic.basis, "synthetic");
assert.equal(SEED, 20261118);
assert.deepEqual(Object.keys(synthetic.method).sort(), [...Object.keys(rules), "random_seed"].sort());
for (const type of PLAN_TYPES) assert.ok(rules.visit_share[type] && rules.utilization_pp[type] && rules.unit_cost_manwon_per_m2[type] && rules.adoption_quarters[type] && rules.budget_elasticity[type], type);
// 지역 통계는 출처 값, 월평균 체류인구는 시드에 없다(공표 통계 API).
for (const [id, stats] of Object.entries(seed.conversion.regions)) assert.ok(!("stay_monthly" in stats) && stats.basis.inflow_external === "source", id);
// 신안 임대주택 공급 호수 조절변수는 실존 시설(출처 90호)에 연결, 기본값이 출처 값.
assert.deepEqual(linkedControls(controlsOf("shinan")).map((c) => [c.facility_id, c.field, c.default]), [["fac_sa_aphae_housing", "supply_units", 90]]);
assert.deepEqual(linkedControls(controlsOf("namwon")), []);

// ── 방안 후보 규칙 ──
const optionTypes = (id) => optionCandidates(facility(id)).map((o) => `${o.planType}/${o.kind}`);
assert.deepEqual(optionTypes("fac_nw_dudong"), ["복합화/visit", "용도 전환/visit", "용도 전환/house"]); // 계획 단계 폐교 → 전환형
assert.deepEqual(optionTypes("fac_sa_museum"), ["복합화/visit", "운영 개선/visit", "개방·공유/visit"]); // 운영 중 폐교 → 운영형
assert.deepEqual(optionTypes("fac_sa_aphae_housing"), ["용도 전환/stay", "운영 개선/house"]);
assert.deepEqual(optionTypes("fac_nw_smartfarm"), ["운영 개선/job", "개방·공유/visit"]);
assert.deepEqual(optionCandidates({ type: "알 수 없음" }).map((o) => o.planType), ["운영 개선", "개방·공유"]);

// ── 규모 등급·수용 상한·비교 사례(사례·통계 계산 단계) ──
assert.deepEqual(["fac_sa_museum", "fac_nw_fiorium", "fac_nw_dudong", "fac_nw_glocal", "fac_sa_fossil", "fac_nw_smartfarm"].map((id) => scaleGrade(facility(id), rules)),
  ["소형", "중형", "소형", "대형", "중형", "중형"]);
assert.equal(capacityCap({ type: "섬 관광" }, rules), 11081 * 25);
assert.equal(capacityCap({ type: "문화관광", capacity_people: 100 }, rules), 100 * 50);
assert.equal(capacityCap(facility("fac_nw_glocal"), rules), 1000); // 계획 목표 수용 인원(등록형)
const tierOf = (id, planType) => comparableCases(optionCandidates(facility(id)).find((o) => o.planType === planType), facility(id), conversion.cases, rules);
assert.deepEqual([tierOf("fac_nw_dudong", "용도 전환"), tierOf("fac_sa_fossil", "복합화"), tierOf("fac_sa_museum", "복합화"), tierOf("fac_sa_museum", "운영 개선")]
  .map((t) => [t.tier, t.cases.map((c) => c.id)]), [
  ["range", ["case_gangju", "case_daesu", "case_yeonhong"]], ["reference", ["case_fiorium", "case_ypark"]], ["single", ["case_fossil"]], ["none", []]
]); // 세계화석광물박물관은 자기 사례를 쓰지 않는다

// ── 추천 행: 결정적 합성, 시설 여건 보정, 추천 지수 만점 80 ──
assert.deepEqual([namwon.length, shinan.length], [5, 5]);
assert.deepEqual(facilityRows("gunwi", seed.facilities, [], {}, conversion), []);
assert.deepEqual(facilityRows("namwon", [], [], {}, conversion), []); // 실데이터 표시(시설 대장 없음)
assert.deepEqual(rowsOf("namwon"), namwon, "같은 시드면 같은 결과");
const reseeded = rowsOf("namwon", {}, { ...conversion, synthetic: { ...synthetic, random_seed: 7 } });
assert.notDeepEqual(reseeded.map((r) => r.util), namwon.map((r) => r.util), "시드가 바뀌면 합성 값도 바뀐다");
assert.equal(INDEX_MAX, 80);
assert.deepEqual(namwon.map((r) => r.rank), [1, 2, 3, 4, 5]);
for (const rows of [namwon, shinan]) for (let i = 1; i < rows.length; i += 1) assert.ok(rows[i - 1].score >= rows[i].score, "점수 높은 순");
for (const row of all) {
  const c = chosen(row);
  const region = SYNTH_REGIONS[row.facility.region_id];
  assert.equal(row.options.filter((o) => o.chosen).length, 1, row.id);
  assert.ok(row.index === Math.round(c.score) && row.index > 0 && row.index <= INDEX_MAX, `${row.id} 지수`);
  assert.equal(row.planType, c.planType);
  assert.deepEqual(row.budget, c.budget);
  assert.ok(Number.isInteger(row.util) && row.util >= 0 && row.util <= 100, `${row.id} 현재 사용률`);
  for (const o of row.options) {
    // 판단: 고른 방안 추천, 점수 80% 이상 검토, 그 아래 보류.
    const expected = o === c && o.living.high > 0 ? "추천" : o.living.high > 0 && o.score >= c.score * 0.8 ? "검토" : "보류";
    assert.equal(o.verdict, expected, `${row.id} ${o.planType} 판단`);
    // 점수 = 생활인구 0.6 × 로그 척도 + 정주 0.2 × (연 정주 ÷ 외부 전입 1%) — 추진 여건 없음.
    const living = 0.6 * Math.min(100, (100 * Math.log10(1 + (o.living.low + o.living.high) / 2)) / Math.log10(1 + region.stay_monthly * rules.visit_cap_share));
    const settle = 0.2 * Math.min(100, (100 * o.settleValue) / (region.inflow_external * 0.01));
    assert.ok(Math.abs(o.score - (living + settle)) < 1e-9, `${row.id} 점수 식`);
    assert.deepEqual(o.utilAfter, { low: Math.min(100, row.util + o.utilization.low), high: Math.min(100, row.util + o.utilization.high) });
    const e = rules.budget_elasticity[o.planType] * rules.budget_change;
    assert.deepEqual(o.budgetSens, { low: Math.round(o.living.low * (1 - e)), high: Math.round(o.living.high * (1 + e)) });
    assert.ok(o.budget.low > 0 && o.budget.low <= o.budget.high, `${row.id} 소요 예산`);
    assert.ok(o.settleValue >= (o.living.high > 0 ? 1 : 0), `${row.id} 정주 값`);
  }
  // 전망 효과 추이: 0에서 시작해 방안별 도입 분기 뒤 정상 상태(생활인구 효과 중간값, 연 정주).
  assert.equal(row.forecast.livingPath.length, FORECAST_QUARTERS + 1);
  assert.deepEqual([row.forecast.livingPath[0], row.forecast.livingPath.at(-1), row.forecast.settlePath.at(-1)], [0, row.forecast.living, c.settleValue]);
}
// 같은 등급 시설도 같은 값이 되지 않는다(연면적·시설별 흔들림으로 규모 보정).
assert.equal(new Set(all.map((r) => r.scaleFactor)).size, all.length);
const sf = (id) => scaleFactor(facility(id), scaleGrade(facility(id), rules), SEED, rules);
assert.ok(sf("fac_sa_fossil") >= 0.7 * 0.9 && sf("fac_sa_fossil") <= 1.4 * 1.1);
// 현재 사용률: 계획·조성 단계는 유휴 범위, 운영 중은 유형 범위, 입주율이 있으면 그 값.
for (const row of all) {
  const f = row.facility;
  const range = Number.isFinite(f.occupancy_rate) ? [f.occupancy_rate, f.occupancy_rate] : ["계획", "조성"].includes(f.stage) ? rules.base_util_range["기타|유휴"] : rules.base_util_range[`${f.type}|운영`] ?? rules.base_util_range["기타|운영"];
  assert.ok(row.util >= range[0] && row.util <= range[1], `${row.id} 사용률 범위`);
}
assert.equal(baseUtil({ id: "x", occupancy_rate: 84 }, SEED, rules), 84);
// 계획 단계 주거(공급 호수만 출처): 초기 입주율 합성(80~92%), 출처 입주율이 있으면 만들지 않는다.
const lease = leaseUpOccupancy(facility("fac_sa_aphae_housing"), SEED, rules);
assert.ok(lease >= 80 && lease <= 92);
assert.equal(leaseUpOccupancy({ ...facility("fac_sa_aphae_housing"), occupancy_rate: 70 }, SEED, rules), null);
assert.equal(shinan.find((r) => r.id === "fac_sa_aphae_housing").facility.basis.occupancy_rate, "synthetic");
// 방안 선택 규칙: 주거 → 정착형, 열린 공간 → 개방·공유, 사용률 30% 미만 → 운영 개선, 계획·조성 → 용도 전환, 그 밖은 근거 등급·점수.
const ruleOf = (rows, id) => [rows.find((r) => r.id === id).planType, rows.find((r) => r.id === id).choiceRule];
assert.deepEqual(ruleOf(shinan, "fac_sa_aphae_housing"), ["운영 개선", "주거시설 정착"]);
assert.deepEqual(ruleOf(namwon, "fac_nw_bogeumjari"), ["운영 개선", "주거시설 정착"]);
assert.deepEqual(ruleOf(namwon, "fac_nw_fiorium"), ["개방·공유", "열린 공간 달빛정원·광장"]);
assert.deepEqual(ruleOf(shinan, "fac_sa_birdcraft"), ["개방·공유", "열린 공간 새조각공원 8,500㎡"]);
assert.deepEqual(ruleOf(namwon, "fac_nw_dudong"), ["용도 전환", "계획 단계"]);
assert.deepEqual(ruleOf(namwon, "fac_nw_glocal"), ["용도 전환", "조성 중 단계"]);
for (const row of all.filter((r) => r.util < rules.low_util_pct && !r.choiceRule.startsWith("열린") && r.facility.type !== "임대주택")) {
  if (row.options.some((o) => o.planType === "운영 개선")) assert.equal(row.planType, "운영 개선", `${row.id} 사용률 낮음`);
}
// 주거 정착형 정주 = 공급 호수 × 입주율 × 세대당 인구 × 외부 전입 비율(± band) × 규모 보정.
const aphae = shinan.find((r) => r.id === "fac_sa_aphae_housing");
const house = Math.round(90 * (lease / 100) * 2 * (7000 / 7500));
const houseOpt = aphae.options.find((o) => o.kind === "house");
assert.deepEqual([houseOpt.settle.low, houseOpt.settle.high], [Math.round(Math.round(house * 0.9) * aphae.scaleFactor), Math.round(Math.round(house * 1.1) * aphae.scaleFactor)]);
assert.equal(houseOpt.settleSynthetic, false);
// 조절변수: 연결된 공급 호수를 늘리면 정착 효과가 커진다. 연결되지 않은 변수는 행을 바꾸지 않는다.
const more = rowsOf("shinan", { rental_units: 200 }).find((r) => r.id === "fac_sa_aphae_housing");
assert.ok(more.options.find((o) => o.kind === "house").settle.low > houseOpt.settle.low);
assert.deepEqual(rowsOf("shinan", { staff: 200 }), shinan);
assert.notEqual(recoControlSig(controlsOf("shinan"), { rental_units: 200 }), recoControlSig(controlsOf("shinan"), {}));
assert.equal(recoControlSig(controlsOf("shinan"), { staff: 200 }), recoControlSig(controlsOf("shinan"), {}));
assert.equal(unlinkedControlNote(controlsOf("namwon"), { farm_scale: 30 }), "스마트팜 단지 규모: 시설 자료를 등록하면 추천에 반영됩니다.");
assert.equal(unlinkedControlNote(controlsOf("namwon"), {}), null);
// 지역 통계(월평균 체류인구)가 없으면 효과를 만들지 않는다 — 지수 빈 값, 같은 자리에 안내.
const noStay = facilityRows("namwon", seed.facilities, [], {}, seed.conversion);
assert.ok(noStay.every((r) => r.options === null && r.index === null && r.forecast === null));
assert.equal(withRegionStay(seed.conversion, "namwon", 0), seed.conversion);
assert.equal(withRegionStay(seed.conversion, "namwon", 1000).regions.namwon.stay_monthly, 1000);
assert.equal(seed.conversion.regions.namwon.stay_monthly, undefined); // 시드 원본 불변
// 도입 곡선·예산(합성 규칙 단위).
assert.deepEqual([adoption("운영 개선", 2, rules), adoption("용도 전환", 8, rules), adoption("용도 전환", 0, rules)], [1, 1, 0]);
assert.ok(adoption("용도 전환", 2, rules) < adoption("운영 개선", 1, rules));
assert.deepEqual(effectPath(1000, "복합화", 12, rules).slice(0, 2), [0, Math.round(1000 * (1 - (1 - 1 / 6) ** 2))]);
assert.deepEqual(budgetOf({ scale_area_m2: 1000, basis: { scale_area_m2: "source" } }, "운영 개선", "소형", rules), { low: 1, high: 3 });
assert.deepEqual(budgetOf({ occupancy_rate: 80, basis: {} }, "용도 전환", "중형", rules), { low: 18, high: 30 }); // 6,000㎡ × 공실 20%
assert.ok(rngFor(1, "a")() !== rngFor(2, "a")());

// ── 문구 ──
assert.equal(livingText({ low: 10, high: 10 }), "월 +10명");
assert.equal(livingText({ low: 1000, high: 2500 }), "월 +1,000~2,500명");
assert.equal(settleText({ settleSynthetic: true, settleValue: 12 }), "연 +12명");
assert.equal(settleText({ settleSynthetic: false, settle: { low: 3, high: 5 } }), "연 +3~5명");
const regionNames = { namwon: "전북특별자치도 남원시", shinan: "전남광주통합특별시 신안군" };
assert.equal(facilityShortName({ name: "신안 이세돌바둑기념관(1도1뮤지엄)" }, regionNames.shinan), "이세돌바둑기념관");
assert.equal(policyShortName({ name: "남원 옛 두동초 발효·체류시설" }, { planType: "용도 전환" }, regionNames.namwon), "옛 두동초 발효·체류시설 용도 전환");

// ── 인구 전망(정책 미반영 추세 감쇠) — 합성 분기 값 ──
const synthPopulation = { unit: "만 명", points: [2024, 2025].flatMap((y, yi) => [1, 2, 3, 4].map((q) => ({ year: y, quarter: q, registered: 1 + yi * 0.4 + q / 10, stay: 3 + yi + q }))) };
const fc = populationForecast(synthPopulation, (p) => p.registered + p.stay, null);
assert.equal(fc.labels.length, 16);
assert.equal(fc.observed, 4);
assert.deepEqual([fc.series[0].values[4], fc.series[0].values[8], fc.series[0].values[12], fc.series[0].values[15]], [7.9, 8.6, 8.95, 12.25]);
const short = populationForecast({ unit: "만 명", points: synthPopulation.points.slice(0, 4) }, (p) => p.registered + p.stay, null);
assert.deepEqual(short.series[0].values, [5.1, 6.2, 7.3, ...Array(13).fill(8.4)]);

console.log("facilityRecommendation: 실존 시설 시드·출처·좌표·사례·합성 규칙(고정 시드)·방안 후보·비교 사례·결정적 합성·추천 지수(만점 80)·판단·사용률·예산 ±20%·소요 예산·방안 선택 규칙·주거 정착·조절변수 연결·지역 통계 없음 안내·도입 곡선·추세 전망 PASS");
