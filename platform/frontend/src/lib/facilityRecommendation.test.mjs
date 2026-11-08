// 시설 × 전환 방안 추천 순수 계산 검사 (vitest 미도입 → node 직접 실행).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  COMPARE_COLUMNS, DEFAULT_CRITERION, EVIDENCE_CHART_SLOTS, PLAN_TYPES, SORT_CRITERIA,
  conversionCompareView, evidenceChartsView, facilityRecommendationView, facilityRows, linkedControls, optionCandidates,
  capacityCap, comparableCases, recoControlSig, scaleGrade, sortFacilityRows, unlinkedControlNote
} from "./facilityRecommendation.js";

const here = dirname(fileURLToPath(import.meta.url));
const read = (...path) => readFileSync(join(here, "..", ...path), "utf-8");
const seed = JSON.parse(read("assets", "mock_data.json"));
const conversion = seed.conversion;
const regionOf = (id) => seed.regions.find((r) => r.id === id);
const controlsOf = (id) => regionOf(id).case.simulation.controls;
const ids = (list) => list.map((r) => r.id);
const rowsOf = (id, values = {}) => facilityRows(id, seed.facilities, controlsOf(id), values, conversion);

// 시드 범위(r4): r3 시드 6건은 그대로 두고, 공개 출처가 있는 실제 시설·사업만 더한다(남원 6·신안 5).
const R3_IDS = ["fac_nw_farm01", "fac_nw_farm02", "fac_nw_rental", "fac_sa_rental", "fac_sa_solar", "fac_sa_garden"];
for (const id of R3_IDS) assert.ok(seed.facilities.some((f) => f.id === id), id);
assert.equal(seed.facilities.length, 11);
const namwon = rowsOf("namwon");
const shinan = rowsOf("shinan");
assert.deepEqual([namwon.length, shinan.length], [6, 5]);
assert.deepEqual([...ids(namwon), ...ids(shinan)].sort(), ids(seed.facilities).sort());
assert.deepEqual(facilityRows("gunwi", seed.facilities, [], {}, conversion), []);
assert.deepEqual(facilityRows("namwon", undefined), []);

// 출처·값 구분: 시설 값 필드마다 basis(source | demo_example | source_plan_target)가 있고, 출처 값이면 시설 출처 URL·확인일이 있다.
// source_plan_target = 출처에 적힌 계획 목표 값(실제 정원·실적 아님, 3차 보정) — 수용 인원에만 쓰고 수용 종류에도 "계획 목표"를 적는다.
const BASIS = new Set(["source", "demo_example", "source_plan_target"]);
for (const f of seed.facilities) {
  for (const [field, b] of Object.entries(f.basis ?? {})) if (b === "source_plan_target") assert.ok(field === "capacity_people" && /^계획 목표/.test(f.capacity_kind), `${f.id}.${field}`);
}
const VALUE_FIELDS = ["metric", "operating_budget", "supply_units", "occupancy_rate", "reco_inputs", "address", "stage", "scale_area_m2", "capacity_people"];
const validSource = (s) => /^https:\/\//.test(s.url) && /^\d{4}-\d{2}-\d{2}$/.test(s.checked) && s.title && ["facility", "type"].includes(s.scope);
for (const f of seed.facilities) {
  const basis = f.basis ?? {};
  for (const field of VALUE_FIELDS) if (f[field] !== undefined) assert.ok(BASIS.has(basis[field]), `${f.id}.${field}`);
  if (f.lat !== undefined || f.lng !== undefined) assert.ok(BASIS.has(basis.location), `${f.id}.location`);
  for (const key of Object.keys(basis)) assert.ok(key === "location" ? f.lat !== undefined : f[key] !== undefined, `${f.id} basis ${key}`);
  assert.ok(Array.isArray(f.sources), f.id);
  for (const s of f.sources) assert.ok(validSource(s), `${f.id} ${s.url}`);
  const hasFacilitySource = f.sources.some((s) => s.scope === "facility");
  if (Object.values(basis).includes("source")) assert.ok(hasFacilitySource, `${f.id} source 값에 시설 출처 없음`);
  if (!R3_IDS.includes(f.id)) assert.ok(hasFacilitySource, `${f.id} r4 추가 시설 출처 없음`);
  if (f.use_plan) assert.ok(PLAN_TYPES.includes(f.use_plan.type), f.id);
  assert.equal(basis.reco_inputs, "demo_example", f.id); // 추진 여건 값은 출처 수치가 아니다
  assert.deepEqual(Object.keys(f.reco_inputs), ["readiness"], f.id); // 1단계부터 추천 지수 입력은 추진 여건만(나머지는 효과 계산)
}
// 운영 단계·연면적은 시설 출처 기사 기준(source): 옛 두동초 2019년 계획, 글로컬캠퍼스 2027 개교 목표(조성), 피오리움·1도1뮤지엄·퍼플섬 운영 중.
assert.deepEqual(seed.facilities.filter((f) => f.stage).map((f) => [f.id, f.stage, f.basis.stage]), [
  ["fac_nw_fiorium", "운영", "source"], ["fac_nw_glocal", "조성", "source"], ["fac_nw_dudong", "계획", "source"], ["fac_sa_museum", "운영", "source"], ["fac_sa_purple", "운영", "source"]
]);
assert.deepEqual(seed.facilities.filter((f) => f.scale_area_m2).map((f) => [f.id, f.scale_area_m2, f.basis.scale_area_m2]), [["fac_nw_fiorium", 3741, "source"], ["fac_sa_museum", 1094, "source"]]);
// 수정 라운드 1(6번): 실명 시설 2곳(신재생 이익공유 발전소·정원수 사회적협동조합)의 예시 운영 예산은 시드에서 값과 basis 키를 지웠다.
// 일반 이름 시설(스마트팜 1·2단지, 임대주택 2곳)의 예시 예산은 그대로다.
for (const id of ["fac_sa_solar", "fac_sa_garden"]) {
  const f = seed.facilities.find((x) => x.id === id);
  assert.equal(f.operating_budget, undefined, id);
  assert.equal(f.basis.operating_budget, undefined, id);
}
assert.deepEqual(["fac_nw_farm01", "fac_nw_farm02", "fac_nw_rental", "fac_sa_rental"].map((id) => seed.facilities.find((f) => f.id === id).operating_budget), [42, 28, 15, 21]);
assert.equal(seed.facilities.find((f) => f.id === "fac_nw_farm01").metric, "재배면적 12㏊ · 고용 청년 38명");
for (const id of ["fac_sa_solar", "fac_sa_garden"]) assert.equal(seed.facilities.find((f) => f.id === id).basis.metric, "source");

// 전환 효과 시드: 지역 통계·사례는 출처 값(source, 원 페이지 URL·확인일), 합성 규칙은 synthetic + 규칙별 생성 방법, 난수 없음.
const https = (s) => /^https:\/\//.test(s.url) && /^\d{4}-\d{2}-\d{2}$/.test(s.checked) && Boolean(s.title);
const STAT_FIELDS = ["stay_monthly", "inflow_total", "inflow_external", "persons_per_household"];
assert.deepEqual(Object.keys(conversion.regions), ["namwon", "shinan"]);
for (const [id, stats] of Object.entries(conversion.regions)) {
  for (const field of STAT_FIELDS) {
    assert.ok(Number.isFinite(stats[field]) && stats[field] > 0, `${id}.${field}`);
    assert.equal(stats.basis[field], "source", `${id}.${field}`);
    assert.ok(stats.periods[field], `${id}.${field} 기준 시점`);
    const cited = stats.sources.filter((s) => s.fields.includes(field));
    assert.equal(cited.length, 1, `${id}.${field} 출처`);
    assert.ok(https(cited[0]) && /mois\.go\.kr|kosis\.kr/.test(cited[0].url), `${id}.${field}`);
  }
  assert.ok(stats.inflow_external <= stats.inflow_total);
}
// 원표 값(2026-11-06 확인): 체류인구 2025-07~2026-06 월평균, 2025년 전입(KOSIS 시군구별 이동자수), 세대당 인구(주민등록 2025-12).
assert.deepEqual(STAT_FIELDS.map((f) => conversion.regions.namwon[f]), [263632, 6643, 3528, 1.87]);
assert.deepEqual(STAT_FIELDS.map((f) => conversion.regions.shinan[f]), [145542, 7564, 7077, 1.87]);
// 사례: 비교 가능한 사례만(인구감소지역 안 사례, 원 파일 89개 시군구로 확인) — 전주 하얀양옥집은 뺐다. 섬 운영 개선 사례 2건(청산도·임자도)을 더했다.
// 2차 보정: 같은 시설 유형의 섬 정원 사례(1004섬 분재정원)를 더했다. 사례마다 지역 유형(섬·농촌·소도시·도시)을 적었다.
assert.deepEqual(conversion.cases.map((c) => c.id), ["case_fiorium", "case_gangju", "case_daesu", "case_purple_festival", "case_cheongsan", "case_imja", "case_bonsai", "case_sangju"]);
assert.deepEqual(conversion.cases.map((c) => [c.id, c.region_type, c.facility_type]), [
  ["case_fiorium", "소도시", "문화관광"], ["case_gangju", "농촌", "폐교 활용"], ["case_daesu", "농촌", "폐교 활용"], ["case_purple_festival", "섬", "섬 관광"],
  ["case_cheongsan", "섬", "섬 관광"], ["case_imja", "섬", "섬 관광"], ["case_bonsai", "섬", "정원수"], ["case_sangju", "소도시", "스마트팜"]
]);
// 행사 사례: 행사 기간(일수·기간)과 기간 방문객 원값을 두고, 연간 계산용 월 값은 기간 방문객 ÷ 12(연 1회 행사)다.
assert.deepEqual(conversion.cases.filter((c) => Number.isFinite(c.event_days)).map((c) => [c.id, c.event_days, c.event_period, c.event_visitors, c.monthly]), [
  ["case_purple_festival", 11, "2026-05-15~25", 50000, 4167], ["case_cheongsan", 30, "2025-04-05~05-04", 50198, 4183], ["case_imja", 4, "2026-02-27~03-02", 20000, 1667]
]);
for (const c of conversion.cases.filter((x) => Number.isFinite(x.event_days))) assert.equal(c.monthly, Math.round(c.event_visitors / 12), c.id);
for (const c of conversion.cases) {
  assert.equal(c.basis, "source", c.id);
  assert.ok(PLAN_TYPES.includes(c.plan_type), c.id);
  assert.ok(c.sources.length && c.sources.every(https), c.id);
  assert.ok(Number.isFinite(c.monthly) && c.monthly > 0 && c.measure && c.method, c.id);
  assert.equal(c.depopulation_area, true, c.id);
  assert.ok(c.region && c.facility_type && ["new", "added"].includes(c.opening) && ["섬", "농촌", "소도시", "도시"].includes(c.region_type), c.id);
  assert.doesNotMatch(c.region, /전주|부산|경주/, c.id);
}
assert.deepEqual(conversion.cases.filter((c) => c.plan_type === "운영 개선").map((c) => [c.id, c.monthly]), [["case_purple_festival", 4167], ["case_cheongsan", 4183], ["case_imja", 1667]]);
assert.equal(conversion.cases.find((c) => c.id === "case_fiorium").scale_area_m2, 3741);
for (const type of PLAN_TYPES) assert.ok(conversion.cases.some((c) => c.plan_type === type), `${type} 사례`);
const synthetic = conversion.synthetic;
assert.equal(synthetic.basis, "synthetic");
assert.equal(synthetic.random_seed, null); // 난수 없음 — 같은 시드면 항상 같은 결과
assert.deepEqual(Object.keys(synthetic.method).sort(), [...Object.keys(synthetic.rules), "random_seed"].sort());
for (const type of PLAN_TYPES) assert.ok(synthetic.rules.visit_share[type] && synthetic.rules.utilization_pp[type], type);

// 방안 후보 규칙: 시설마다 2~3개, 네 가지 방안 유형 안에서. 폐교 활용은 운영 단계로 후보가 갈리고, use_plan 문구는 같은 유형 후보를 바꾼다.
for (const f of seed.facilities) {
  const options = optionCandidates(f);
  assert.ok(options.length >= 2 && options.length <= 3, f.id);
  for (const o of options) assert.ok(PLAN_TYPES.includes(o.planType) && o.text && ["visit", "stay", "house", "job"].includes(o.kind), f.id);
  if (f.use_plan) assert.equal(options.filter((o) => o.text === f.use_plan.text && o.planType === f.use_plan.type).length, 1, f.id);
}
const optionTypes = (id) => optionCandidates(seed.facilities.find((f) => f.id === id)).map((o) => `${o.planType}/${o.kind}`);
assert.deepEqual(optionTypes("fac_nw_dudong"), ["복합화/visit", "용도 전환/visit", "용도 전환/house"]); // 계획 단계 폐교 → 전환형
assert.deepEqual(optionTypes("fac_sa_museum"), ["복합화/visit", "운영 개선/visit", "개방·공유/visit"]); // 운영 중 폐교 → 운영형
assert.deepEqual(optionTypes("fac_nw_rental"), ["용도 전환/stay", "운영 개선/house"]);
assert.deepEqual(optionTypes("fac_nw_farm01"), ["운영 개선/job", "개방·공유/visit"]);
assert.deepEqual(optionCandidates({ type: "알 수 없음" }).map((o) => o.planType), ["운영 개선", "개방·공유"]);

// 효과 값마다 basis: 사용률·정주는 합성(synthetic), 생활인구는 출처 사례 값 그대로면 source·규칙이나 지역 상한이 들어가면 synthetic.
// 정주인구는 주거·일자리형 방안에만. 시설마다 추천 1개(고른 방안)이고 판단은 추천·검토·보류.
const caseById = Object.fromEntries(conversion.cases.map((c) => [c.id, c]));
for (const row of [...namwon, ...shinan]) {
  const region = conversion.regions[seed.facilities.find((f) => f.id === row.id).region_id];
  const cap = region.stay_monthly * synthetic.rules.visit_cap_share;
  assert.equal(row.options.filter((o) => o.chosen).length, 1, row.id);
  assert.equal(row.options.filter((o) => o.verdict === "추천").length, 1, row.id);
  assert.equal(row.options.find((o) => o.chosen).verdict, "추천");
  assert.equal(row.planType, row.options.find((o) => o.chosen).planType);
  for (const o of row.options) {
    assert.ok(["추천", "검토", "보류"].includes(o.verdict));
    assert.equal(o.utilization.basis, "synthetic", row.id);
    assert.ok(["source", "synthetic"].includes(o.living.basis), row.id);
    assert.ok(o.living.low <= o.living.high && o.living.low >= 0 && o.living.high <= Math.round(cap), `${row.id} 지역 상한`);
    assert.equal(Boolean(o.settle), o.kind === "house" || o.kind === "job", `${row.id} 정주는 주거·일자리형만`);
    if (o.settle) assert.equal(o.settle.basis, "synthetic");
    const f = seed.facilities.find((x) => x.id === row.id);
    if (o.kind === "visit") assert.ok(o.living.high <= capacityCap(f, synthetic.rules), `${row.id} 시설 수용 상한`);
    // 사례값 범위(출처 값 그대로·"유사 사례 N곳")는 같은 시설 유형 사례가 2곳 이상일 때만. 1곳이거나 다른 유형 참고 사례면 규모 보정 추정(합성).
    if (o.living.basis === "source") {
      const values = o.cases.map((id) => caseById[id].monthly);
      assert.ok(o.cases.length >= 2, `${row.id} 출처 범위는 2곳 이상`);
      assert.deepEqual([o.living.low, o.living.high], [Math.min(...values), Math.max(...values)], `${row.id} 사례 값 그대로`);
      assert.match(o.note, new RegExp(`^유사 사례 ${o.cases.length}곳 기준(·연 1회 행사 방문을 12개월로 나눈 값)?$`));
    }
    const event = o.cases.some((id) => Number.isFinite(caseById[id].event_days));
    assert.equal(Boolean(o.annualized), event, `${row.id} 연 환산 표시는 행사 사례를 쓸 때만`);
    // 3차 보정: 계획 목표 수용 인원(실제 정원 출처 없음)으로 잘린 값이면 근거 문장 끝에 "계획 수용 인원 기준".
    const planned = o.kind === "visit" && /계획/.test(f.capacity_kind ?? "") && o.living.high === capacityCap(f, synthetic.rules) ? "·계획 수용 인원 기준" : "";
    assert.equal(o.note.endsWith("·계획 수용 인원 기준"), Boolean(planned), `${row.id} 계획 수용 인원 문구`);
    if (o.cases.length >= 2 && o.caseTier === "range") assert.match(o.note, new RegExp(`^유사 사례 ${o.cases.length}곳 기준(·규모 반영)?${event ? "·연 1회 행사 방문을 12개월로 나눈 값" : ""}${planned}$`));
    if (o.caseTier === "single") assert.equal(o.note, `같은 유형 시설 규모 기준 추정${event ? "·연 1회 행사 방문을 12개월로 나눈 값" : ""}${planned}`);
    if (o.caseTier === "reference") assert.equal(o.note, `유사 방안 사례 규모 기준 추정${event ? "·연 1회 행사 방문을 12개월로 나눈 값" : ""}${planned}`);
    if (o.caseTier !== "range") assert.ok(!/^유사 사례/.test(o.note), `${row.id} 1곳·참고 사례는 사례값 범위로 보이지 않음`);
    for (const id of o.cases) {
      const c = caseById[id];
      assert.notEqual(c.facility_id, row.id, `${row.id} 자기 사례 제외`);
      assert.equal(c.depopulation_area, true, `${row.id} 비교 가능 사례만`);
      assert.ok(["섬", "농촌", "소도시"].includes(c.region_type), `${row.id} 비슷한 지역 유형만`);
      if (o.caseTier === "range" || o.caseTier === "single") assert.equal(c.facility_type, f.type, `${row.id} 같은 시설 유형`);
      if (o.caseTier === "reference") assert.ok(c.applies_to.includes(f.type) && !conversion.cases.some((x) => x.plan_type === o.planType && x.facility_type === f.type && x.facility_id !== f.id), `${row.id} 같은 유형이 없을 때만 참고`);
    }
  }
}
// 규모 등급: 출처 연면적이 있으면 그 값(2,000㎡ 미만 소형 · 10,000㎡ 미만 중형 · 그 이상 대형), 없으면 유형 기본 등급(합성).
const rules = synthetic.rules;
const facility = (id) => seed.facilities.find((f) => f.id === id);
const kase = (id) => conversion.cases.find((c) => c.id === id);
assert.deepEqual(["fac_sa_museum", "fac_nw_fiorium", "fac_nw_dudong", "fac_nw_glocal", "fac_sa_purple", "fac_sa_garden", "fac_nw_farm01"].map((id) => scaleGrade(facility(id), rules)),
  ["소형", "중형", "소형", "대형", "대형", "중형", "중형"]);
assert.deepEqual(["case_fiorium", "case_gangju", "case_cheongsan", "case_bonsai", "case_sangju"].map((id) => scaleGrade(kase(id), rules)), ["중형", "소형", "대형", "중형", "중형"]);
assert.deepEqual(rules.scale_factor, { 소형: 1, 중형: 2, 대형: 4 });
assert.deepEqual(rules.operating_increment, [0.25, 0.5]);
// 시설 수용 상한(월): 출처 수용 인원(등록형)이면 그 인원 × 1, 아니면 (출처 연면적 또는 등급 대표 연면적) ÷ 4.6㎡ × 일 2회전 × 월 25일.
assert.deepEqual([rules.capacity_area_per_person_m2, rules.capacity_grade_area_m2, rules.capacity_turnover], [4.6, { 소형: 1000, 중형: 6000, 대형: 20000 }, { daily: 2, open_days: 25, enrolled: 1 }]);
assert.ok(synthetic.rule_sources.some((s) => s.rules.includes("capacity_area_per_person_m2") && /^https:\/\/www\.law\.go\.kr\//.test(s.url) && /\[별표 7\] 수용인원의 산정 방법/.test(s.title)));
// 3차 보정(글로컬캠퍼스): 실제 정원 공식 출처를 찾지 못해 1,000명(최종 외국인 유학생 유치 목표)을 유지하고, 수용 종류에 계획 목표임을 적는다.
const glocal = facility("fac_nw_glocal");
assert.deepEqual([glocal.capacity_people, glocal.capacity_kind, glocal.basis.capacity_people], [1000, "계획 목표(유치 목표·등록형)", "source_plan_target"]);
assert.ok(glocal.sources.some((s) => s.url === "https://www.jjan.kr/article/20260929500077" && /1000명 유치 목표/.test(s.title) && s.checked === "2026-11-06"));
// 3차 보정(퍼플섬): 1일 입도 제한·도선 수송 능력 공식 값은 찾지 못했고, 출처 관측 일 최대 방문 1만1,081명(2026-05-24) × 월 25일을 시설 수용 상한으로 쓴다.
const purple = facility("fac_sa_purple");
assert.deepEqual([purple.capacity_people, purple.capacity_kind, purple.basis.capacity_people], [11081, "일 최대 방문(관측)", "source"]);
assert.ok(purple.sources.some((s) => s.url === "https://www.ktotalchannel.com/news/articleView.html?idxno=12801" && /1만1,081명/.test(s.title) && s.checked === "2026-11-06"));
// 유형 규칙: 섬 관광 시설에 수용 인원 자료가 없으면 같은 유형 출처 관측값을 일 방문 상한으로 쓴다(시설 ID가 아니라 유형으로).
assert.deepEqual(rules.capacity_daily_by_type, { "섬 관광": 11081 });
assert.ok(synthetic.rule_sources.some((s) => s.rules.includes("capacity_daily_by_type") && s.url === "https://www.ktotalchannel.com/news/articleView.html?idxno=12801"));
assert.equal(capacityCap({ type: "섬 관광" }, rules), 11081 * 25);
assert.equal(capacityCap({ type: "섬 관광", scale_area_m2: 3000 }, rules), 11081 * 25);
assert.equal(capacityCap({ type: "문화관광", capacity_people: 100 }, rules), 100 * 50);
assert.deepEqual(["fac_nw_glocal", "fac_sa_museum", "fac_nw_fiorium", "fac_nw_dudong", "fac_sa_purple"].map((id) => capacityCap(facility(id), rules)),
  [1000, Math.round(1094 / 4.6) * 50, Math.round(3741 / 4.6) * 50, Math.round(1000 / 4.6) * 50, 11081 * 25]);
// 비교 가능 사례 단계(시설·사례 속성만): 같은 시설 유형 2곳 이상 → 범위, 1곳 → 규모 보정 추정, 없으면 다른 유형 참고 사례, 그것도 없으면 없음.
const tierOf = (id, planType) => comparableCases(optionCandidates(facility(id)).find((o) => o.planType === planType), facility(id), conversion.cases, rules);
assert.deepEqual(Object.values({
  dudongConvert: tierOf("fac_nw_dudong", "용도 전환"), gardenMix: tierOf("fac_sa_garden", "복합화"), farmOpen: tierOf("fac_nw_farm01", "개방·공유"),
  glocalMix: tierOf("fac_nw_glocal", "복합화"), purpleOps: tierOf("fac_sa_purple", "운영 개선"), museumOps: tierOf("fac_sa_museum", "운영 개선")
}).map((t) => [t.tier, t.cases.map((c) => c.id)]), [
  ["range", ["case_gangju", "case_daesu"]], ["single", ["case_bonsai"]], ["single", ["case_sangju"]],
  ["reference", ["case_fiorium"]], ["range", ["case_cheongsan", "case_imja"]], ["none", []]
]);
assert.doesNotMatch(read("lib", "facilityRecommendation.js"), /fac_(nw|sa)_/); // 시설 ID로 결과를 고르는 분기 없음
// 규모 보정이 실제로 적용된다(다른 시설의 전체 방문객을 그대로 변화로 옮기지 않는다). 2차 보정부터 큰 시설에도 배수를 건다:
// - 글로컬캠퍼스(대형, 조성) 복합화 = 피오리움(중형) 10,833 × 2 = 21,666 → 시설 수용 상한 1,000(계획 수용 인원)으로 자름
// - 옛 두동초(소형, 계획) 복합화 = 10,833 × 1/2 = 5,417 / 1도1뮤지엄(소형, 운영 중) = 10,833 × 1/2 × 25~50% = 1,354~2,708
// - 퍼플섬(대형, 운영 중) 복합화 = 10,833 × 2 × 25~50% = 5,417~10,833 → 지역 상한 7,277로 자름
// - 정원수(중형, 운영 중) 복합화 = 분재정원(중형) 9,804 × 1 × 25~50% = 2,451~4,902(같은 유형 1곳 → 추정)
// - 피오리움(중형, 운영 중) 운영 개선 = 섬 축제 3곳(대형, 참고 사례) × 1/2 → 834~2,092
const opt = (rowsList, id, planType) => rowsList.find((r) => r.id === id).options.find((o) => o.planType === planType);
assert.deepEqual([opt(namwon, "fac_nw_glocal", "복합화").living, opt(namwon, "fac_nw_glocal", "복합화").note], [{ low: 1000, high: 1000, basis: "synthetic" }, "유사 방안 사례 규모 기준 추정·계획 수용 인원 기준"]);
assert.deepEqual([opt(namwon, "fac_nw_dudong", "복합화").living, opt(namwon, "fac_nw_dudong", "복합화").note], [{ low: 5417, high: 5417, basis: "synthetic" }, "유사 방안 사례 규모 기준 추정"]);
assert.deepEqual(opt(shinan, "fac_sa_museum", "복합화").living, { low: Math.round(10833 * 0.5 * 0.25), high: Math.round(10833 * 0.5 * 0.5), basis: "synthetic" });
assert.deepEqual(opt(shinan, "fac_sa_purple", "복합화").living, { low: Math.round(10833 * 2 * 0.25), high: Math.round(145542 * 0.05), basis: "synthetic" });
assert.deepEqual([opt(shinan, "fac_sa_garden", "복합화").living, opt(shinan, "fac_sa_garden", "복합화").note], [{ low: Math.round(9804 * 0.25), high: Math.round(9804 * 0.5), basis: "synthetic" }, "같은 유형 시설 규모 기준 추정"]);
assert.deepEqual([opt(namwon, "fac_nw_fiorium", "운영 개선").living, opt(namwon, "fac_nw_fiorium", "운영 개선").cases, opt(namwon, "fac_nw_fiorium", "운영 개선").note],
  [{ low: Math.round(1667 * 0.5), high: Math.round(4183 * 0.5), basis: "synthetic" }, ["case_purple_festival", "case_cheongsan", "case_imja"], "유사 방안 사례 규모 기준 추정·연 1회 행사 방문을 12개월로 나눈 값"]);
assert.deepEqual([opt(namwon, "fac_nw_farm01", "개방·공유").living, opt(namwon, "fac_nw_farm01", "개방·공유").note], [{ low: 152, high: 152, basis: "synthetic" }, "같은 유형 시설 규모 기준 추정"]);
assert.deepEqual([opt(namwon, "fac_nw_dudong", "용도 전환").living, opt(namwon, "fac_nw_dudong", "용도 전환").note], [{ low: 700, high: 1250, basis: "source" }, "유사 사례 2곳 기준"]);
// 퍼플섬 운영 개선은 자기 사례를 빼고 같은 유형 섬 축제 2곳 값 그대로(출처 값), 근거 문장에 연간 환산을 드러낸다.
// 3차 보정: 같은 유형 2곳 범위라 근거 등급이 가장 높아 추천, 값이 더 큰 복합화(다른 유형 참고 사례 보정)는 검토.
assert.deepEqual([opt(shinan, "fac_sa_purple", "운영 개선").living, opt(shinan, "fac_sa_purple", "운영 개선").cases, opt(shinan, "fac_sa_purple", "운영 개선").verdict, opt(shinan, "fac_sa_purple", "운영 개선").note],
  [{ low: 1667, high: 4183, basis: "source" }, ["case_cheongsan", "case_imja"], "추천", "유사 사례 2곳 기준·연 1회 행사 방문을 12개월로 나눈 값"]);
assert.equal(opt(shinan, "fac_sa_purple", "복합화").verdict, "검토");

// 근거 등급(3차 보정): 같은 유형 2곳 이상 범위 > 같은 유형 1곳 보정 > 다른 유형 참고 보정 > 통계 기준 추정.
// 모든 시설에서 고른 방안은 시설 안 최고 등급이고, 같은 등급 안에서만 점수가 가장 크다. 낮은 등급 방안은 값이 커도 추천이 아니다.
const GRADE = { range: 4, single: 3, reference: 2, none: 1 };
for (const row of [...namwon, ...shinan]) {
  const chosen = row.options.find((o) => o.chosen);
  const top = Math.max(...row.options.map((o) => GRADE[o.caseTier]));
  assert.equal(GRADE[chosen.caseTier], top, `${row.id} 최고 등급 방안을 고름`);
  for (const o of row.options.filter((x) => GRADE[x.caseTier] === top)) assert.ok(o.score <= chosen.score, `${row.id} 같은 등급 안 최고 점수`);
  for (const o of row.options.filter((x) => GRADE[x.caseTier] < top)) assert.notEqual(o.verdict, "추천", `${row.id} 낮은 등급은 추천 아님`);
}
// 판단 규칙 전수 대조(3차 수정 라운드 1): 고른 방안 = 추천, 나머지는 생활인구 효과가 있고 점수가 고른 방안 점수 × 0.8 이상이면 검토, 아니면 보류.
const expectedVerdict = (o, chosen) => (o === chosen && o.living.high > 0 ? "추천" : o.living.high > 0 && o.score >= chosen.score * 0.8 ? "검토" : "보류");
const assertVerdicts = (rows, label) => {
  for (const row of rows) {
    const chosen = row.options.find((o) => o.chosen);
    for (const o of row.options) assert.equal(o.verdict, expectedVerdict(o, chosen), `${label} ${row.id} ${o.planType}/${o.kind} 판단`);
  }
};
assertVerdicts([...namwon, ...shinan], "기본");
// 경계 바로 아래·위 사례를 고정한다 — 기준을 0.798~0.848 사이로 묶는다.
const ratioOf = (rows, id, planType) => Math.round((1000 * opt(rows, id, planType).score) / rows.find((r) => r.id === id).options.find((o) => o.chosen).score) / 1000;
assert.deepEqual([ratioOf(shinan, "fac_sa_museum", "운영 개선"), opt(shinan, "fac_sa_museum", "운영 개선").verdict], [0.798, "보류"]); // 1도1뮤지엄 운영 개선
assert.deepEqual([ratioOf(namwon, "fac_nw_farm01", "운영 개선"), opt(namwon, "fac_nw_farm01", "운영 개선").verdict], [0.848, "검토"]); // 스마트팜 1단지 운영 개선
// 옛 두동초: 복합화(참고 사례 보정 월 +5,417명)가 값은 크지만 캠핑·체험 전환(같은 유형 2곳 범위 월 +700~1,250명)이 추천, 복합화는 검토.
assert.ok(opt(namwon, "fac_nw_dudong", "복합화").score > opt(namwon, "fac_nw_dudong", "용도 전환").score);
assert.deepEqual(namwon.find((r) => r.id === "fac_nw_dudong").options.map((o) => [o.planType, o.caseTier, o.verdict]),
  [["복합화", "reference", "검토"], ["용도 전환", "range", "추천"], ["용도 전환", "none", "보류"]]);

// 합성 규칙 재계산(난수 없음): 사례 없는 방문형 = 지역 월평균 체류인구 × 비율 범위, 주거 공실 = (공급 − 입주) 호수 기준.
const nwOption = (id, planType) => namwon.find((r) => r.id === id).options.find((o) => o.planType === planType);
assert.deepEqual([nwOption("fac_nw_fiorium", "개방·공유").living.low, nwOption("fac_nw_fiorium", "개방·공유").living.high], [Math.round(263632 * 0.0005), Math.round(263632 * 0.001)]);
assert.equal(nwOption("fac_nw_fiorium", "개방·공유").note, "지역 생활인구 규모 기준 추정");
const nwRental = nwOption("fac_nw_rental", "용도 전환"); // 공실 120 × 9% = 10.8호
assert.deepEqual([nwRental.living.low, nwRental.living.high, nwRental.note], [Math.round(10.8 * 4), Math.round(10.8 * 8), "공실 규모 기준 추정"]);
const nwSettle = nwOption("fac_nw_rental", "운영 개선"); // 10.8호 × 채움 30~50% × 세대당 1.87명 × 다른 시군구 전입 3,528/6,643
assert.deepEqual([nwSettle.settle.low, nwSettle.settle.high], [Math.round(10.8 * 0.3 * 1.87 * 3528 / 6643), Math.round(10.8 * 0.5 * 1.87 * 3528 / 6643)]);
assert.deepEqual([nwSettle.utilization.low, nwSettle.utilization.high], [Math.round(9 * 0.3), Math.round(9 * 0.5)]);
const glocalJob = nwOption("fac_nw_glocal", "용도 전환"); // 일자리형: 연간 다른 시군구 전입 × 0.3~0.6%
assert.deepEqual([glocalJob.settle.low, glocalJob.settle.high, glocalJob.note], [Math.round(3528 * 0.003), Math.round(3528 * 0.006), "전입 통계 기준 추정"]);
assert.deepEqual([glocalJob.living.low, glocalJob.living.high], [glocalJob.settle.low, glocalJob.settle.high]);
// 지역 1~3순위 고른 방안의 생활인구 효과 합 ÷ 지역 월평균 체류인구(보고서 18절): 남원 1.0~1.6%, 신안 3.8~8.1%.
const top3Share = (rowsList, id) => {
  const chosen = rowsList.slice(0, 3).map((r) => r.options.find((o) => o.chosen).living);
  const stay = conversion.regions[id].stay_monthly;
  return [chosen.reduce((s, r) => s + r.low, 0), chosen.reduce((s, r) => s + r.high, 0)].map((v) => Math.round((1000 * v) / stay) / 10);
};
assert.deepEqual(top3Share(namwon, "namwon"), [1, 1.6]);
assert.deepEqual(top3Share(shinan, "shinan"), [3.8, 8.1]);

// 활용도: 주거 시설 입주율만 값이 있고, 나머지는 빈 값(화면에서 행동 안내).
const byId = Object.fromEntries([...namwon, ...shinan].map((r) => [r.id, r]));
assert.equal(byId.fac_nw_rental.utilization, 91);
assert.equal(byId.fac_sa_rental.utilization, 84);
assert.equal(byId.fac_nw_rental.utilizationLabel, "입주율");
for (const row of [...namwon, ...shinan]) if (!/rental/.test(row.id)) assert.equal(row.utilization, null, row.id);
for (const row of [...namwon, ...shinan]) {
  assert.ok(PLAN_TYPES.includes(row.planType), row.id);
  assert.doesNotMatch(`${row.planType} ${row.plan}`, /통합/);
}

// 추천 순위(1단계 재정의): 생활인구 효과 중심 + 정주 효과·추진 여건 보조. 모든 시설에 값이 있다(주 그래프 빈 막대 0).
assert.deepEqual(ids(namwon), ["fac_nw_fiorium", "fac_nw_glocal", "fac_nw_dudong", "fac_nw_farm01", "fac_nw_farm02", "fac_nw_rental"]);
assert.deepEqual(namwon.map((r) => r.index), [60, 55, 52, 46, 44, 42]);
assert.deepEqual(ids(shinan), ["fac_sa_purple", "fac_sa_garden", "fac_sa_museum", "fac_sa_rental", "fac_sa_solar"]);
assert.deepEqual(shinan.map((r) => r.index), [69, 68, 65, 53, 48]);
// 고른 방안 분포(3차 보정 재도출 결과 — 규칙을 결과에 맞추지 않았다): 남원 운영 개선 1·복합화 1·용도 전환 2·개방·공유 2, 신안 운영 개선 1·복합화 2·용도 전환 1·개방·공유 1.
assert.deepEqual(namwon.map((r) => r.planType), ["운영 개선", "복합화", "용도 전환", "개방·공유", "개방·공유", "용도 전환"]);
assert.deepEqual(shinan.map((r) => r.planType), ["운영 개선", "복합화", "복합화", "용도 전환", "개방·공유"]);
// 시설 자신의 출처 계획(글로컬캠퍼스 교육·외국인 지원 거점 전환)은 후보에 있고, 일자리형이라 정주 기준 효과만 잡혀 보류다.
assert.deepEqual([opt(namwon, "fac_nw_glocal", "용도 전환").text, opt(namwon, "fac_nw_glocal", "용도 전환").kind, opt(namwon, "fac_nw_glocal", "용도 전환").verdict],
  ["폐교 대학 시설을 교육·외국인 지원 거점으로 전환 검토", "job", "보류"]);
assert.deepEqual([...namwon, ...shinan].map((r) => r.rank), [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5]);
// 지역 통계가 없으면 효과를 만들지 않는다 — 지수 빈 값(자료 등록 후 산정), 방안 비교는 행동 안내.
const noStats = facilityRows("namwon", seed.facilities, controlsOf("namwon"), {});
assert.ok(noStats.every((r) => r.index === null && r.options === null && r.planCases.length === 0));

// 지역별 유형: 두 지역의 시설 유형 집합과 Top3 유형 목록이 다르고, Top3 임대주택은 최대 1건(기존 시험 유지).
const types = (rows) => rows.map((r) => r.type);
assert.notDeepEqual([...new Set(types(namwon))].sort(), [...new Set(types(shinan))].sort());
assert.deepEqual(types(namwon.slice(0, 3)), ["문화관광", "교육", "폐교 활용"]);
assert.deepEqual(types(shinan.slice(0, 3)), ["섬 관광", "정원수", "폐교 활용"]);
assert.notDeepEqual([...types(namwon.slice(0, 3))].sort(), [...types(shinan.slice(0, 3))].sort());
// 조절변수 전 범위(신안 임대주택 공급 호수 50~1,500호)에서도 Top3 임대주택 ≤1, 모든 막대에 값.
const saRental = controlsOf("shinan").find((c) => c.key === "rental_units");
for (let supply = saRental.min; supply <= saRental.max; supply += saRental.step) {
  assertVerdicts(rowsOf("shinan", { rental_units: supply }), `supply ${supply}`);
  const view = facilityRecommendationView(rowsOf("shinan", { rental_units: supply }));
  assert.ok(view.items.slice(0, 3).filter((i) => i.type === "임대주택").length <= 1, `supply ${supply}`);
  for (const item of view.items) assert.ok(Number.isFinite(item.score) && item.score >= 0 && item.score <= 100, `${supply} ${item.id}`);
}
for (const rows of [namwon, shinan]) assert.ok(facilityRecommendationView(rows).items.slice(0, 3).filter((i) => i.type === "임대주택").length <= 1);

// 조절변수 연결: 신안 임대주택 공급 호수만 시설 값에 연결되고, 남원 두 변수는 연결 대상이 없다.
assert.deepEqual(linkedControls(controlsOf("shinan")).map((c) => c.key), ["rental_units"]);
assert.deepEqual(linkedControls(controlsOf("namwon")), []);
assert.deepEqual(linkedControls(undefined), []);
const shinanRental = (supply) => rowsOf("shinan", { rental_units: supply }).find((r) => r.id === "fac_sa_rental");
assert.equal(shinanRental(300).utilization, 84); // 기본값 = 시설 자료 공급 호수 → 관측값 그대로
assert.equal(shinanRental(600).utilization, 42); // 공급을 늘리면 현재 입주 호수 기준 입주율이 낮아진다
assert.equal(shinanRental(50).utilization, 100); // 공급보다 많이 입주할 수 없다
// 공실이 늘면 단기 체류·정착 효과가 커져 추천 지수·순위도 바뀐다(재도출 시 반영).
assert.deepEqual([shinanRental(300).index, shinanRental(600).index, shinanRental(1500).index], [53, 72, 81]);
assert.deepEqual([shinanRental(300).rank, shinanRental(600).rank, shinanRental(1500).rank], [4, 1, 1]);
assert.ok(shinanRental(50).options.every((o) => o.verdict === "보류" || o.chosen)); // 공실 0호면 효과 0
// 연결되지 않은 변수(운영 인력)는 행을 바꾸지 않는다.
assert.deepEqual(rowsOf("shinan", { staff: 300 }), shinan);
assert.deepEqual(rowsOf("namwon", { farm_scale: 80, staff: 300 }), namwon);

// 재도출 판정: 값이 있는 시설 변수(신안 임대주택 공급 호수)만 서명에 들어가 바뀌면 재도출, 값이 없는 변수는 서명 불변.
const nwControls = controlsOf("namwon");
const saControls = controlsOf("shinan");
assert.equal(recoControlSig(nwControls, {}), "");
assert.equal(recoControlSig(nwControls, { farm_scale: 80, staff: 300 }), recoControlSig(nwControls, {}));
assert.equal(recoControlSig(saControls, {}), "rental_units:300");
assert.notEqual(recoControlSig(saControls, { rental_units: 600 }), recoControlSig(saControls, {}));
assert.equal(recoControlSig(saControls, { staff: 300 }), recoControlSig(saControls, {}));
const saView = (values) => facilityRecommendationView(rowsOf("shinan", values));
assert.equal(saView({}).items.find((i) => i.id === "fac_sa_rental").cells[1], "입주율 84%");
assert.equal(saView({ rental_units: 600 }).items.find((i) => i.id === "fac_sa_rental").cells[1], "입주율 42%");

// 값이 없는 변수(남원 스마트팜 단지 규모·운영 인력)를 바꾸면 재도출 안내 대신 같은 자리의 행동 안내만.
assert.equal(unlinkedControlNote(nwControls, {}), null);
assert.equal(unlinkedControlNote(nwControls, { farm_scale: 18, staff: 60 }), null);
assert.equal(unlinkedControlNote(nwControls, { farm_scale: 30 }), "스마트팜 단지 규모: 시설 자료를 등록하면 추천에 반영됩니다.");
assert.equal(unlinkedControlNote(nwControls, { farm_scale: 30, staff: 100 }), "스마트팜 단지 규모·운영 인력: 시설 자료를 등록하면 추천에 반영됩니다.");
assert.equal(unlinkedControlNote(saControls, { rental_units: 600 }), null);
assert.equal(unlinkedControlNote(undefined, {}), null);
// 화면 연결: 데모 재도출은 시드 conversion을 넘기고, 서명은 연결 변수만, 행동 안내는 재도출 안내와 같은 <p> 자리.
const pageSrc = read("pages", "SimulatorPage.jsx");
const dockSrc = read("components", "SimulatorDock.jsx");
assert.match(pageSrc, /facilityRows\(region\.id, appData\.facilities, controls, controlValues, appData\.conversion\)/);
assert.match(pageSrc, /const recoInputSig = demo \? `\$\{contextKey\}\|\$\{recoControlSig\(controls, controlValues\)\}` : currentSig;/);
assert.match(pageSrc, /const recoStale = recommendation !== null && recoSig !== recoInputSig;/);
assert.match(pageSrc, /const controlNote = demo \? unlinkedControlNote\(controls, controlValues\) : null;/);
assert.match(pageSrc, /controlNote=\{controlNote\}/);
assert.equal(dockSrc.match(/\{\(stale \|\| controlNote\) && \(/g)?.length, 1);
assert.match(dockSrc, /\{stale \? "변수가 변경되었습니다\. 추천을 다시 도출하세요\." : controlNote\}/);

// 조절변수 재도출 — 같은 지역 주거 시설 두 곳 중 연결된 시설의 공급 호수를 늘리면 공실이 늘어 순서가 바뀐다.
const twinConversion = { regions: { r: { stay_monthly: 100000, inflow_total: 2000, inflow_external: 1000, persons_per_household: 2 } }, cases: [], synthetic };
const twin = [
  { id: "h1", name: "주거 1", type: "임대주택", region_id: "r", supply_units: 100, occupancy_rate: 80, reco_inputs: { readiness: 50 } },
  { id: "h2", name: "주거 2", type: "임대주택", region_id: "r", supply_units: 100, occupancy_rate: 70, reco_inputs: { readiness: 50 } }
];
const twinControls = [{ key: "units", default: 100, facility_id: "h1", field: "supply_units" }];
const order = (values) => ids(facilityRecommendationView(facilityRows("r", twin, twinControls, values, twinConversion)).items);
assert.deepEqual(order({ units: 100 }), ["h2", "h1"]);
assert.deepEqual(order({ units: 200 }), ["h1", "h2"]);
// 추진 여건 값이 없으면 지수는 비고(자료 등록 후 산정) 순위 뒤로 간다.
const partial = facilityRows("r", [{ id: "x", name: "x", type: "임대주택", region_id: "r" }, twin[0]], [], {}, twinConversion);
assert.deepEqual(partial.map((r) => [r.id, r.index]), [["h1", 44], ["x", null]]);

// 표 정렬: 기본은 추천 순위, 순위·활용도·운영 예산 열 제목으로 바꾼다. 값 없는 시설은 방향과 관계없이 뒤, 같은 값은 입력 순서 유지.
assert.equal(DEFAULT_CRITERION, "rank");
assert.deepEqual(Object.keys(SORT_CRITERIA), ["rank", "utilization", "budget"]);
assert.deepEqual(ids(sortFacilityRows(namwon, "utilization")), ["fac_nw_rental", "fac_nw_fiorium", "fac_nw_glocal", "fac_nw_dudong", "fac_nw_farm01", "fac_nw_farm02"]);
assert.deepEqual(ids(sortFacilityRows(namwon, "budget")), ["fac_nw_farm01", "fac_nw_farm02", "fac_nw_rental", "fac_nw_fiorium", "fac_nw_glocal", "fac_nw_dudong"]);
assert.deepEqual(sortFacilityRows(shinan, "budget").map((r) => r.budget), [21, null, null, null, null]);
const sample = [
  { id: "a", utilization: 70, budget: 5 }, { id: "b", utilization: null, budget: 9 },
  { id: "c", utilization: 40, budget: 5 }, { id: "d", utilization: 70, budget: null }
];
assert.deepEqual(ids(sortFacilityRows(sample, "utilization")), ["c", "a", "d", "b"]);
assert.deepEqual(ids(sortFacilityRows(sample, "budget")), ["b", "a", "c", "d"]);
assert.deepEqual(ids(sample), ["a", "b", "c", "d"]);

// 표시 형식: 표 9열 그대로(순위 + 6열 + 숫자 열 1 + 검토 구분), 기금 투자 여부·운영 예산·이용실적 칸은 지금처럼(합성 값 없음).
const view = facilityRecommendationView(namwon);
assert.equal(view.columns.length, 6);
assert.equal(1 + view.columns.length + (view.scoreColumn ? 1 : 0) + 1, 9);
for (const item of view.items) assert.equal(item.cells.length + 1, view.columns.length, item.id);
assert.deepEqual(view.columns.map((c) => c.label), ["시설", "활용 방안", "활용도(주거 입주율·개방 시설 이용률)", "생활인구 관련 관측(외지인 방문·소비)", "기금 투자 여부", "자료 기준일"]);
assert.deepEqual([view.rankColumn.label, view.scoreColumn.label], ["순위", "운영 예산(억 원)"]);
for (const item of view.items) assert.deepEqual(item.cells.slice(2), ["사용 승인 후 확인", "투자 대장 확인", "기준일 확인"], item.id);
assert.deepEqual(view.items.map((i) => i.scoreCell), ["예산 자료 등록", "예산 자료 등록", "예산 자료 등록", "42", "28", "15"]);
assert.deepEqual(view.items.map((i) => i.cells[0]), view.items.map((i) => i.options.find((o) => o.chosen).text)); // 활용 방안 = 고른 방안
assert.deepEqual(view.items.map((i) => [i.scoreText, i.scoreUnit]).slice(0, 3), [["60", "추천 지수"], ["55", "추천 지수"], ["52", "추천 지수"]]);
assert.equal(view.items[0].chartLabel, "남원 달빛정원 피오리움 · 운영 개선");
// Top3 카드 효과 줄: 고른 방안 유형과 기대 생활인구 변화 한 줄.
assert.deepEqual(view.items.slice(0, 3).map((i) => i.effectLine), ["운영 개선 · 생활인구 월 +834~2,092명", "복합화 · 생활인구 월 +1,000명", "용도 전환 · 생활인구 월 +700~1,250명"]);
// 6번: 신안 표 운영 예산 칸 — 실명 시설(정원수 협동조합·신재생 발전소)은 같은 칸에 "예산 자료 등록", 임대주택 단지는 예시 값 21 그대로.
const saTable = facilityRecommendationView(shinan);
assert.equal(saTable.scoreColumn.label, "운영 예산(억 원)");
assert.deepEqual(saTable.items.map((i) => [i.id, i.scoreCell]), [["fac_sa_purple", "예산 자료 등록"], ["fac_sa_garden", "예산 자료 등록"], ["fac_sa_museum", "예산 자료 등록"], ["fac_sa_rental", "21"], ["fac_sa_solar", "예산 자료 등록"]]);
assert.deepEqual(view.items.map((i) => view.status(i).text), ["우선 검토", "우선 검토", "우선 검토", "추가 검토", "추가 검토", "추가 검토"]);
assert.deepEqual(view.items.map((i, n) => view.rankText(n)).slice(0, 3), ["1순위", "2순위", "3순위"]);
assert.deepEqual(ids(view.tableItems), ids(view.items));
assert.match(view.footnote, /추천 지수\(0~100\) 높은 순/);
const budgetView = facilityRecommendationView(namwon, "budget");
assert.deepEqual(ids(budgetView.items), ids(view.items));
assert.deepEqual(budgetView.tableItems.map((i) => i.rank), [4, 5, 6, 1, 2, 3]);
assert.match(budgetView.footnote, /운영 예산 큰 순/);
const utilView = facilityRecommendationView(namwon, "utilization");
assert.equal(utilView.tableItems[0].id, "fac_nw_rental");
const missing = facilityRecommendationView(partial);
assert.deepEqual(missing.items.map((i) => [i.scoreText, i.scoreUnit, missing.status(i).text]), [["44", "추천 지수", "우선 검토"], ["", "자료 등록 후 산정", "자료 등록 후 검토"]]);

// 전환 방안 비교: 6열(전환 방안·사용률·생활인구·정주인구·근거·판단), 시설 방안 수만큼 행. 실데이터(방안 없음)는 같은 구조에 행동 안내.
assert.deepEqual(COMPARE_COLUMNS, ["전환 방안", "사용률 변화", "생활인구 변화", "정주인구 변화", "근거", "판단"]);
const views = { namwon: view, shinan: facilityRecommendationView(shinan) };
const compares = Object.values(views).flatMap((v) => v.items.map(conversionCompareView));
for (const c of compares) {
  assert.equal(c.title, "전환 방안 비교");
  assert.ok(c.rows.length >= 2 && c.rows.length <= 3 && c.empty === null);
  for (const row of c.rows) {
    assert.equal(row.cells.length, COMPARE_COLUMNS.length);
    assert.match(row.cells[1], /^\+\d+(~\d+)?%p$/);
    assert.match(row.cells[2], /^월 \+[\d,]+(~[\d,]+)?명(\(연 환산\))?$/);
    // 3차 보정: 연 1회 행사를 12개월로 나눈 값이면 숫자 칸에도 "(연 환산)" — 근거 칸의 행사 문장과 같이 다닌다.
    assert.equal(row.cells[2].endsWith("(연 환산)"), row.cells[4].includes("연 1회 행사 방문을 12개월로 나눈 값"), row.cells.join(" | "));
    assert.match(row.cells[3], /^(연 \+[\d,]+(~[\d,]+)?명|해당 없음)$/);
    assert.match(row.cells[4], /^((유사 사례 \d곳 기준(·규모 반영)?|같은 유형 시설 규모 기준 추정|유사 방안 사례 규모 기준 추정)(·연 1회 행사 방문을 12개월로 나눈 값)?|지역 생활인구 규모 기준 추정|공실 규모 기준 추정|전입 통계 기준 추정)(·계획 수용 인원 기준)?$/);
    assert.equal(row.cells[5], row.verdict);
    assert.ok(["source", "synthetic"].includes(row.basis.living) && row.basis.utilization === "synthetic");
  }
}
const fioriumItem = view.items.find((i) => i.id === "fac_nw_fiorium");
const fiorium = conversionCompareView(fioriumItem);
assert.equal(fiorium.scope, "남원 달빛정원 피오리움 · 전환 뒤 변화 범위");
assert.deepEqual(fiorium.rows.map((r) => r.cells), [
  ["복합화 · 전시·정원·광장 복합 운영으로 체류 시간 확대 검토", "+10~20%p", "월 +791~1,582명", "해당 없음", "지역 생활인구 규모 기준 추정", "검토"],
  ["운영 개선 · 야간·계절 행사 운영 개선 검토", "+5~10%p", "월 +834~2,092명(연 환산)", "해당 없음", "유사 방안 사례 규모 기준 추정·연 1회 행사 방문을 12개월로 나눈 값", "추천"],
  ["개방·공유 · 광장·정원 지역 행사 개방 검토", "+3~8%p", "월 +132~264명", "해당 없음", "지역 생활인구 규모 기준 추정", "보류"]
]);
assert.deepEqual(conversionCompareView(facilityRecommendationView(shinan).items[0]).rows.map((r) => r.cells), [
  ["운영 개선 · 테마 행사·체류 프로그램 운영 개선 검토", "+5~10%p", "월 +1,667~4,183명(연 환산)", "해당 없음", "유사 사례 2곳 기준·연 1회 행사 방문을 12개월로 나눈 값", "추천"],
  ["복합화 · 숙박·체험 기능 복합화 검토", "+10~20%p", "월 +5,417~7,277명", "해당 없음", "유사 방안 사례 규모 기준 추정", "검토"]
]);
// 글로컬캠퍼스 비교 표: 계획 목표 수용 인원으로 잘린 복합화 행의 근거 칸.
assert.deepEqual(conversionCompareView(view.items.find((i) => i.id === "fac_nw_glocal")).rows.map((r) => [r.cells[2], r.cells[4], r.cells[5]]), [
  ["월 +11~21명", "전입 통계 기준 추정", "보류"], ["월 +1,000명", "유사 방안 사례 규모 기준 추정·계획 수용 인원 기준", "추천"]
]);
for (const real of [conversionCompareView(null), conversionCompareView({ id: "s", name: "민감도 항목" })]) {
  assert.deepEqual([real.title, real.rows, real.empty], ["전환 방안 비교", [], "시설 자료를 등록하면 전환 방안을 비교합니다."]);
}

// 근거 그래프 묶음: 세 자리 고정. 생활인구는 공식 분기 값(등록·체류 두 계열), 사례 성과는 고른 시설의 추천 방안 유형 사례(출처 값),
// 활용 방안 구성은 추천 결과. 실데이터(값 없음)는 같은 자리에 행동 안내만.
assert.deepEqual(EVIDENCE_CHART_SLOTS.map((s) => s.id), ["population", "case", "plans"]);
for (const id of ["namwon", "shinan"]) {
  const pop = seed.recommendation_charts[id].population;
  assert.equal(pop.basis, "source");
  assert.ok(pop.sources.every(https) && /mois\.go\.kr/.test(pop.sources[0].url));
  assert.deepEqual(pop.series.map((s) => s.label), ["등록인구", "체류인구"]);
  for (const s of pop.series) assert.equal(s.values.length, pop.labels.length);
  for (const label of pop.labels) assert.match(label, /^\d{2}년 [1-4]분기$/); // 공표된 분기만(전망 없음)
  assert.equal(seed.recommendation_charts[id].case, undefined); // 지역 고정 사례 그래프는 방안 유형 사례로 바뀌었다
  const charts = evidenceChartsView(seed.recommendation_charts[id], views[id].items);
  assert.deepEqual(charts.map((c) => c.id), ["population", "case", "plans"]);
  for (const chart of charts) {
    assert.equal(chart.empty, null, `${id} ${chart.id}`);
    assert.ok(chart.series.length >= 1 && chart.series.every((s) => s.values.length === chart.labels.length && s.values.every(Number.isFinite)));
    assert.ok(chart.scope && chart.note, `${id} ${chart.id}`);
  }
  assert.equal(charts[0].series.length, 2);
  assert.equal(charts[2].series[0].values.reduce((a, b) => a + b, 0), views[id].items.length);
}
// 남원 체류인구 분기 값은 2025년 연평균 26만 명(기존 출처 값)과 맞는다.
assert.equal(Math.round(seed.recommendation_charts.namwon.population.series[1].values.slice(0, 4).reduce((a, b) => a + b, 0) / 4), 26);
// 사례 성과: 고른 시설의 추천 방안 근거와 같은 사례 묶음(자기 사례 제외·비교 가능 사례만) — 그래프 막대 수 = 근거 문장의 "N곳".
const nwCases = evidenceChartsView(seed.recommendation_charts.namwon, view.items)[1];
assert.equal(nwCases.scope, "운영 개선 참고 사례 · 기간별 방문 (명)"); // 1순위 피오리움의 운영 개선 근거는 다른 유형(섬 축제) 참고 사례
assert.deepEqual([nwCases.labels.length, nwCases.series[0].values], [3, [50000, 50198, 20000]]);
// 3순위 옛 두동초는 같은 유형 2곳(캠핑·체험 전환 폐교) 사례 묶음.
const dudongCases = evidenceChartsView(seed.recommendation_charts.namwon, view.items, view.items.find((i) => i.id === "fac_nw_dudong"))[1];
assert.deepEqual([dudongCases.scope, dudongCases.series[0].values], ["용도 전환 사례 · 월평균 방문 (명)", [1250, 700]]);
for (const item of [...view.items, ...facilityRecommendationView(shinan).items]) {
  const chosen = item.options.find((o) => o.chosen);
  const chart = evidenceChartsView(seed.recommendation_charts.namwon, view.items, item)[1];
  // 막대는 근거 사례와 같은 묶음, 각 막대에 기간(월평균 또는 축제 기간 N일)을 붙이고 행사는 기간 방문객 원값을 그린다.
  assert.deepEqual(chart.labels, chosen.cases.map((id) => { const c = caseById[id]; return Number.isFinite(c.event_days) ? `${c.name}(축제 기간 ${c.event_days}일)` : `${c.name}(월평균)`; }), `${item.id} 그래프 = 근거 사례`);
  if (chosen.cases.length) assert.deepEqual(chart.series[0].values, chosen.cases.map((id) => (Number.isFinite(caseById[id].event_days) ? caseById[id].event_visitors : caseById[id].monthly)));
  assert.ok(!chosen.cases.some((id) => caseById[id].facility_id === item.id), `${item.id} 자기 사례를 그리지 않음`);
  if (chosen.caseTier === "range") assert.match(chosen.note, new RegExp(`^유사 사례 ${chart.labels.length}곳`));
  if (chosen.cases.length) assert.match(chart.scope, chosen.caseTier === "range" ? /^\S+( \S+)? 사례 · / : / 참고 사례 · /);
  else assert.equal(chart.empty, "고른 방안과 같은 유형의 비교 가능한 사례가 없습니다.");
}
const fioCases = evidenceChartsView(seed.recommendation_charts.namwon, view.items, fioriumItem)[1];
// 행사 사례는 월평균으로 환산해 보이지 않는다 — "축제 기간 N일"과 기간 방문객 원값, 제목도 "기간별 방문".
assert.deepEqual([fioCases.scope, fioCases.labels, fioCases.series[0].values, fioCases.series[0].label], ["운영 개선 참고 사례 · 기간별 방문 (명)",
  ["신안 퍼플섬 라벤더 축제(축제 기간 11일)", "완도 청산도 슬로걷기 축제(축제 기간 30일)", "신안 임자도 섬 홍매화 축제(축제 기간 4일)"], [50000, 50198, 20000], "방문객"]);
assert.ok(!fioCases.labels.some((l) => /피오리움/.test(l)) && !/월평균/.test(fioCases.scope + fioCases.labels.join("")));
const farmCases = evidenceChartsView(seed.recommendation_charts.namwon, view.items, view.items.find((i) => i.id === "fac_nw_farm01"))[1];
assert.deepEqual([farmCases.scope, farmCases.labels], ["개방·공유 참고 사례 · 월평균 방문 (명)", ["상주 스마트팜 혁신밸리 견학(월평균)"]]);
assert.match(farmCases.note, /^출처: 매일신문 2022-07-25$/);
assert.notDeepEqual(evidenceChartsView(null, views.namwon.items)[2].series, evidenceChartsView(null, views.shinan.items)[2].series);
const realCharts = evidenceChartsView(null, null);
assert.deepEqual(realCharts.map((c) => c.empty), ["생활인구 통계를 연결하면 표시됩니다.", "시설 이용실적을 등록하면 표시됩니다.", "시설 자료를 등록하면 표시됩니다."]);
for (const c of realCharts) assert.deepEqual([c.labels, c.series, c.scope, c.note], [[], [], "", ""]);

// 추천 근거 접이식 제외(11-06 사용자 결정): STAGE ④에 추천 근거 영역·전용 코드 없음, 다른 단계 계산 근거 두 곳은 유지.
const stageSrc = pageSrc.slice(pageSrc.indexOf('id="stage-report"'));
const handoffIdx = stageSrc.indexOf('data-slot="sim.report.handoff"');
assert.ok(handoffIdx > 0);
assert.doesNotMatch(stageSrc.slice(0, handoffIdx), /ModelEvidence|sim\.report\.evidence|추천 근거/);
assert.doesNotMatch(pageSrc, /FACILITY_EVIDENCE_ROWS|facilitySourceRow|recoSim/);
assert.equal(pageSrc.match(/data-slot="sim\.report\.handoff"/g)?.length, 1);
assert.match(stageSrc.slice(0, handoffIdx), /<\/Card>\s*\)\}\s*<div className="pl-report-handoff" $/);
for (const slot of ["sim.factor.evidence", "sim.sim.evidence"]) assert.equal(pageSrc.split(`slot="${slot}"`).length - 1, 1, slot);
assert.doesNotMatch(read("lib", "facilityRecommendation.js"), /추천 근거/);

// 화면 문구: 계산 매개변수·계산식·가중치·내부 표기(데모·예시·합성 등) 없음.
const chartTexts = ["namwon", "shinan"].flatMap((id) => evidenceChartsView(seed.recommendation_charts[id], views[id].items))
  .concat(realCharts, farmCases).flatMap((c) => [c.title, c.scope, c.note, c.empty ?? "", ...c.labels, ...c.series.map((s) => s.label)]);
const compareTexts = compares.concat(conversionCompareView(null)).flatMap((c) => [c.title, c.scope, c.empty ?? "", ...c.rows.flatMap((r) => r.cells)]);
const screenTexts = [
  ...[view, budgetView, utilView, views.shinan].flatMap((v) => [v.title, v.scoreLabel, v.caption, v.basisTip, v.footnote,
    ...v.columns.map((c) => c.label), v.rankColumn.label, v.scoreColumn.label,
    ...v.items.flatMap((i) => [i.name, i.chartLabel, i.category, i.effectLine, i.scoreText, i.scoreUnit, i.scoreCell, ...i.cells])]),
  ...chartTexts,
  ...compareTexts,
  ...COMPARE_COLUMNS,
  unlinkedControlNote(nwControls, { farm_scale: 30, staff: 100 })
];
const FORBIDDEN = /RICE|Reach|Impact|Confidence|Effort|점수|가중|비중|비율 범위|÷|×|√|\d+\s*호|room|link|readiness|데모|예시|시드|seed|mock|demo_example|synthetic|합성|구현|API|KT|BC|로그 척도|상한/i;
for (const text of screenTexts) assert.doesNotMatch(text, FORBIDDEN, text);
// 실명 시설 현재 상태 단정 없음: 전환 방안 비교·카드 효과 줄에는 변화 범위만 — 현재 이용률·예산·기금 투자 같은 현재 상태 값이 없다.
for (const text of [...compareTexts, ...view.items.map((i) => i.effectLine), ...views.shinan.items.map((i) => i.effectLine)]) {
  assert.doesNotMatch(text, /현재|이용률 \d|입주율|예산|기금|투자|억 원/, text);
}

// 패널 구조: 주 그래프 → 전환 방안 비교(새 영역 1개) → 근거 그래프 묶음 → 표 → 쪽 넘김 → 각주, 모드 분기 없이(두 모드 같은 자리·같은 구조).
const panelSrc = read("components", "PolicyRecommendation.jsx");
assert.equal(panelSrc.match(/data-slot="sim\.report\.charts"/g)?.length, 1);
assert.equal(panelSrc.match(/data-slot="sim\.report\.compare"/g)?.length, 1);
assert.equal(panelSrc.match(/<ConversionCompare /g)?.length, 1);
const mainChartIdx = panelSrc.indexOf("<Bar data={chartData}");
const compareIdx = panelSrc.indexOf("<ConversionCompare compare={compare} />");
const chartsIdx = panelSrc.lastIndexOf('data-slot="sim.report.charts"');
const tableIdx = panelSrc.indexOf('<div className={view.criterion ? "table-container');
const footIdx = panelSrc.indexOf("{view.footnote}");
const cardEndIdx = panelSrc.lastIndexOf("</Card>");
assert.ok(mainChartIdx > 0 && mainChartIdx < compareIdx && compareIdx < chartsIdx && chartsIdx < tableIdx, "주 그래프 → 전환 방안 비교 → 그래프 묶음 → 표");
assert.ok(tableIdx < panelSrc.indexOf("<TablePager") && panelSrc.indexOf("<TablePager") < footIdx && footIdx < cardEndIdx, "표 → 쪽 넘김 → 각주 → 카드 끝");
assert.doesNotMatch(panelSrc.slice(mainChartIdx, tableIdx), /facilityRows\s*&&|\?\s*\(?\s*<section|realView\s*&&/);
assert.match(panelSrc, /conversionCompareView\(realView \? null : selected\)/); // 실데이터는 같은 표에 행동 안내
assert.match(panelSrc, /\{chart\.empty && <ChartEmptyNote>\{chart\.empty\}<\/ChartEmptyNote>\}/);
assert.match(pageSrc, /chartSeed=\{appData\.recommendation_charts\?\.\[currentRegion\.id\]\}/);
// 1~3순위 카드: 같은 div 카드에 버튼 역할·탭 순서·눌림 상태·Enter/Space 선택, 카드 안 효과 줄 1개.
const cardSrc = panelSrc.slice(panelSrc.indexOf('className="reco-card"'), panelSrc.indexOf('className="reco-score"'));
for (const attr of ['role="button"', "tabIndex={0}", "aria-pressed={s.id === selected?.id}", "onClick={() => setSelectedId(s.id)}", 'e.key === "Enter" || e.key === " "']) assert.ok(cardSrc.includes(attr), attr);
assert.equal(cardSrc.match(/className="reco-effect"/g)?.length, 1);
// 지도 시설물은 위치 자료가 있는 시설만.
const mapSrc = read("components", "SimulatorMap.jsx");
assert.equal(mapSrc.match(/filter\(hasLocation\)/g)?.length, 2);
// 6번: 지도 시설 팝업도 예산 값이 없으면 같은 줄에 "예산 자료 등록"(실명 시설 2곳은 위치가 있어 팝업이 뜬다).
assert.ok(mapSrc.includes('운영 예산: ${Number.isFinite(f.operating_budget) ? `<strong>${f.operating_budget}억</strong>` : "예산 자료 등록"}'));
assert.equal(seed.facilities.filter((f) => Number.isFinite(f.lat) && Number.isFinite(f.lng)).length, 6);

console.log("facilityRecommendation: 출처·합성 구분·방안 후보 규칙·규모 보정(큰 시설 포함)·시설 수용 상한(섬 관광 일 방문·계획 목표)·근거 등급 순 방안 선택·연 환산 표기·사례 2곳 이상 범위·같은 시설·지역 유형 사례·행사 기간 표기·효과 basis·효과 중심 순위·Top3 임대주택≤1·전환 방안 비교·사례 묶음 일치·실명 시설 예산 안내·근거 그래프·실데이터 안내·화면 문구·카드 선택 PASS");
