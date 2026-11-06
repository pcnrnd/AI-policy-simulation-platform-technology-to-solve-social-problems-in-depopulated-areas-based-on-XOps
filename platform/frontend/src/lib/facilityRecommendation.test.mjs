// 시설 × 활용 방안 추천 순수 계산 검사 (vitest 미도입 → node 직접 실행).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  DEFAULT_CRITERION, EVIDENCE_CHART_SLOTS, FACILITY_EVIDENCE_ROWS, PLAN_TYPES, SORT_CRITERIA,
  evidenceChartsView, facilityRecommendationView, facilityRows, facilitySourceRow, linkedControls, recoControlSig,
  sortFacilityRows, unlinkedControlNote
} from "./facilityRecommendation.js";

const here = dirname(fileURLToPath(import.meta.url));
const read = (...path) => readFileSync(join(here, "..", ...path), "utf-8");
const seed = JSON.parse(read("assets", "mock_data.json"));
const regionOf = (id) => seed.regions.find((r) => r.id === id);
const controlsOf = (id) => regionOf(id).case.simulation.controls;
const ids = (list) => list.map((r) => r.id);

// 시드 범위(r4): r3 시드 6건은 그대로 두고, 공개 출처가 있는 실제 시설·사업만 더한다(남원 6·신안 5).
// r3의 "시드 6건만" 검사는 r4 사용자 지시(남원·신안 각각 다른 사례)로 "출처 있는 시설만 추가"로 바뀌었다.
const R3_IDS = ["fac_nw_farm01", "fac_nw_farm02", "fac_nw_rental", "fac_sa_rental", "fac_sa_solar", "fac_sa_garden"];
for (const id of R3_IDS) assert.ok(seed.facilities.some((f) => f.id === id), id);
assert.equal(seed.facilities.length, 11);
const namwon = facilityRows("namwon", seed.facilities, controlsOf("namwon"), {});
const shinan = facilityRows("shinan", seed.facilities, controlsOf("shinan"), {});
assert.deepEqual([namwon.length, shinan.length], [6, 5]);
assert.deepEqual([...ids(namwon), ...ids(shinan)].sort(), ids(seed.facilities).sort());
assert.deepEqual(facilityRows("gunwi", seed.facilities, [], {}), []);
assert.deepEqual(facilityRows("namwon", undefined), []);

// 출처·값 구분: 화면 값 필드마다 basis(source | demo_example)가 있고, 출처 값이면 시설 출처 URL·확인일이 있다.
// r4에서 더한 시설은 모두 시설 자체 출처(scope facility)가 있다. 출처 없는 값은 모두 demo_example.
const BASIS = new Set(["source", "demo_example"]);
const VALUE_FIELDS = ["metric", "operating_budget", "supply_units", "occupancy_rate", "reco_inputs", "address"];
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
  assert.equal(basis.reco_inputs, "demo_example", f.id); // 추천 지수 구성 값은 출처 수치가 아니다
}
// r3 시드의 출처 미확인 수치는 지우지 않고 demo_example로 남긴다(신재생·정원수는 같은 사업의 출처 수치로 바꿈).
assert.equal(seed.facilities.find((f) => f.id === "fac_nw_farm01").metric, "재배면적 12㏊ · 고용 청년 38명");
assert.equal(seed.facilities.find((f) => f.id === "fac_nw_farm01").basis.metric, "demo_example");
for (const id of ["fac_sa_solar", "fac_sa_garden"]) assert.equal(seed.facilities.find((f) => f.id === id).basis.metric, "source");

// 활용도: 주거 시설 입주율만 값이 있고, 나머지는 빈 값(화면에서 행동 안내).
const byId = Object.fromEntries([...namwon, ...shinan].map((r) => [r.id, r]));
assert.equal(byId.fac_nw_rental.utilization, 91);
assert.equal(byId.fac_sa_rental.utilization, 84);
assert.equal(byId.fac_nw_rental.utilizationLabel, "입주율");
for (const row of [...namwon, ...shinan]) if (!/rental/.test(row.id)) assert.equal(row.utilization, null, row.id);
// 활용 방안은 네 가지 안에서만, 통합 검토는 없다.
for (const row of [...namwon, ...shinan]) {
  assert.ok(PLAN_TYPES.includes(row.planType), row.id);
  assert.doesNotMatch(`${row.planType} ${row.plan}`, /통합/);
}

// 추천 순위: 추천 지수 높은 순, 모든 시설에 값이 있다(주 그래프 빈 막대 0).
assert.deepEqual(ids(namwon), ["fac_nw_fiorium", "fac_nw_glocal", "fac_nw_farm01", "fac_nw_dudong", "fac_nw_farm02", "fac_nw_rental"]);
assert.deepEqual(namwon.map((r) => r.index), [73, 72, 61, 57, 54, 41]);
assert.deepEqual(ids(shinan), ["fac_sa_museum", "fac_sa_purple", "fac_sa_garden", "fac_sa_solar", "fac_sa_rental"]);
assert.deepEqual(shinan.map((r) => r.index), [73, 70, 63, 53, 43]);
assert.deepEqual([...namwon, ...shinan].map((r) => r.rank), [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5]);

// 지역별 유형: 두 지역의 시설 유형 집합과 Top3 유형 목록이 다르고, Top3 임대주택은 최대 1건.
const types = (rows) => rows.map((r) => r.type);
assert.notDeepEqual([...new Set(types(namwon))].sort(), [...new Set(types(shinan))].sort());
assert.deepEqual(types(namwon.slice(0, 3)), ["문화관광", "교육", "스마트팜"]);
assert.deepEqual(types(shinan.slice(0, 3)), ["폐교 활용", "섬 관광", "정원수"]);
assert.notDeepEqual([...types(namwon.slice(0, 3))].sort(), [...types(shinan.slice(0, 3))].sort());
// 조절변수 전 범위(신안 임대주택 공급 호수 50~1,500호)에서도 Top3 임대주택 ≤1, 모든 막대에 값.
const saRental = controlsOf("shinan").find((c) => c.key === "rental_units");
for (let supply = saRental.min; supply <= saRental.max; supply += saRental.step) {
  const view = facilityRecommendationView(facilityRows("shinan", seed.facilities, controlsOf("shinan"), { rental_units: supply }));
  assert.ok(view.items.slice(0, 3).filter((i) => i.type === "임대주택").length <= 1, `supply ${supply}`);
  for (const item of view.items) assert.ok(Number.isFinite(item.score), `${supply} ${item.id}`);
}
for (const rows of [namwon, shinan]) {
  const view = facilityRecommendationView(rows);
  assert.ok(view.items.slice(0, 3).filter((i) => i.type === "임대주택").length <= 1);
  for (const item of view.items) assert.ok(Number.isFinite(item.score) && item.score >= 0 && item.score <= 100, item.id);
}

// 조절변수 연결: 신안 임대주택 공급 호수만 시설 값에 연결되고, 남원 두 변수는 연결 대상이 없다.
assert.deepEqual(linkedControls(controlsOf("shinan")).map((c) => c.key), ["rental_units"]);
assert.deepEqual(linkedControls(controlsOf("namwon")), []);
assert.deepEqual(linkedControls(undefined), []);
const shinanRental = (supply) => facilityRows("shinan", seed.facilities, controlsOf("shinan"), { rental_units: supply })
  .find((r) => r.id === "fac_sa_rental");
assert.equal(shinanRental(300).utilization, 84); // 기본값 = 시설 자료 공급 호수 → 관측값 그대로
assert.equal(shinanRental(600).utilization, 42); // 공급을 늘리면 현재 입주 호수 기준 입주율이 낮아진다
assert.equal(shinanRental(50).utilization, 100); // 공급보다 많이 입주할 수 없다
// 입주율이 바뀌면 추천 지수·순위도 바뀐다(재도출 시 반영).
assert.deepEqual([shinanRental(300).index, shinanRental(600).index, shinanRental(1500).index], [43, 59, 70]);
assert.deepEqual([shinanRental(300).rank, shinanRental(600).rank, shinanRental(1500).rank], [5, 4, 2]);
// 연결되지 않은 변수(운영 인력)는 행을 바꾸지 않는다.
assert.deepEqual(facilityRows("shinan", seed.facilities, controlsOf("shinan"), { staff: 300 }), shinan);
assert.deepEqual(facilityRows("namwon", seed.facilities, controlsOf("namwon"), { farm_scale: 80, staff: 300 }), namwon);

// 재도출 판정: 값이 있는 시설 변수(신안 임대주택 공급 호수)만 서명에 들어가 바뀌면 재도출, 값이 없는 변수는 서명 불변.
const nwControls = controlsOf("namwon");
const saControls = controlsOf("shinan");
assert.equal(recoControlSig(nwControls, {}), "");
assert.equal(recoControlSig(nwControls, { farm_scale: 80, staff: 300 }), recoControlSig(nwControls, {}));
assert.equal(recoControlSig(saControls, {}), "rental_units:300");
assert.notEqual(recoControlSig(saControls, { rental_units: 600 }), recoControlSig(saControls, {}));
assert.equal(recoControlSig(saControls, { staff: 300 }), recoControlSig(saControls, {}));
// 공급 호수를 바꿔 다시 도출하면 표의 입주율이 실제로 바뀐다.
const saView = (values) => facilityRecommendationView(facilityRows("shinan", seed.facilities, saControls, values));
assert.equal(saView({}).items.find((i) => i.id === "fac_sa_rental").cells[1], "입주율 84%");
assert.equal(saView({ rental_units: 600 }).items.find((i) => i.id === "fac_sa_rental").cells[1], "입주율 42%");

// 값이 없는 변수(남원 스마트팜 단지 규모·운영 인력)를 바꾸면 재도출 안내 대신 같은 자리의 행동 안내만.
assert.equal(unlinkedControlNote(nwControls, {}), null);
assert.equal(unlinkedControlNote(nwControls, { farm_scale: 18, staff: 60 }), null); // 기본값 그대로면 안내 없음
assert.equal(unlinkedControlNote(nwControls, { farm_scale: 30 }), "스마트팜 단지 규모: 시설 자료를 등록하면 추천에 반영됩니다.");
assert.equal(unlinkedControlNote(nwControls, { farm_scale: 30, staff: 100 }), "스마트팜 단지 규모·운영 인력: 시설 자료를 등록하면 추천에 반영됩니다.");
assert.equal(unlinkedControlNote(saControls, { rental_units: 600 }), null); // 값이 있는 변수는 재도출 안내 쪽
assert.equal(unlinkedControlNote(undefined, {}), null);
// 화면 연결: 데모 재도출 서명은 연결 변수만, 행동 안내는 재도출 안내와 같은 <p> 자리(재도출 안내 우선).
const pageSrc = read("pages", "SimulatorPage.jsx");
const dockSrc = read("components", "SimulatorDock.jsx");
assert.match(pageSrc, /const recoInputSig = demo \? `\$\{contextKey\}\|\$\{recoControlSig\(controls, controlValues\)\}` : currentSig;/);
assert.match(pageSrc, /const recoStale = recommendation !== null && recoSig !== recoInputSig;/);
assert.match(pageSrc, /const controlNote = demo \? unlinkedControlNote\(controls, controlValues\) : null;/);
assert.match(pageSrc, /controlNote=\{controlNote\}/);
assert.equal(dockSrc.match(/\{\(stale \|\| controlNote\) && \(/g)?.length, 1);
assert.match(dockSrc, /\{stale \? "변수가 변경되었습니다\. 추천을 다시 도출하세요\." : controlNote\}/);
assert.doesNotMatch(dockSrc, /\{stale && \(/);

// 표 정렬: 기본은 추천 순위, 순위·활용도·운영 예산 열 제목으로 바꾼다. 값 없는 시설은 방향과 관계없이 뒤, 같은 값은 입력 순서 유지.
assert.equal(DEFAULT_CRITERION, "rank");
assert.deepEqual(Object.keys(SORT_CRITERIA), ["rank", "utilization", "budget"]);
assert.deepEqual(ids(sortFacilityRows(namwon, "utilization")), ["fac_nw_rental", "fac_nw_fiorium", "fac_nw_glocal", "fac_nw_farm01", "fac_nw_dudong", "fac_nw_farm02"]);
assert.deepEqual(ids(sortFacilityRows(namwon, "budget")), ["fac_nw_farm01", "fac_nw_farm02", "fac_nw_rental", "fac_nw_fiorium", "fac_nw_glocal", "fac_nw_dudong"]);
assert.deepEqual(sortFacilityRows(shinan, "budget").map((r) => r.budget), [33, 21, 9, null, null]);
const rows = [
  { id: "a", utilization: 70, budget: 5 }, { id: "b", utilization: null, budget: 9 },
  { id: "c", utilization: 40, budget: 5 }, { id: "d", utilization: 70, budget: null }
];
assert.deepEqual(ids(sortFacilityRows(rows, "utilization")), ["c", "a", "d", "b"]);
assert.deepEqual(ids(sortFacilityRows(rows, "budget")), ["b", "a", "c", "d"]);
assert.deepEqual(ids(rows), ["a", "b", "c", "d"]); // 원본 배열은 그대로

// 조절변수가 바뀌면 순위가 다시 도출된다 — 같은 지역 주거 시설 두 곳 중 연결된 시설의 공급 호수를 늘리면 순서가 바뀐다.
const twin = [
  { id: "h1", name: "주거 1", type: "임대주택", region_id: "r", operating_budget: 10, supply_units: 100, occupancy_rate: 80, reco_inputs: { link: 50, readiness: 50 } },
  { id: "h2", name: "주거 2", type: "임대주택", region_id: "r", operating_budget: 10, supply_units: 100, occupancy_rate: 70, reco_inputs: { link: 50, readiness: 50 } }
];
const twinControls = [{ key: "units", default: 100, facility_id: "h1", field: "supply_units" }];
const order = (values) => ids(facilityRecommendationView(facilityRows("r", twin, twinControls, values)).items);
assert.deepEqual(order({ units: 100 }), ["h2", "h1"]);
assert.deepEqual(order({ units: 200 }), ["h1", "h2"]); // 입주율 80% → 40%, 활용 여지가 커져 지수가 오른다
// 추천 지수 구성 값이 없으면 지수는 비고(자료 등록 후 산정) 순위 뒤로 간다.
const partial = facilityRows("r", [{ id: "x", name: "x", region_id: "r" }, twin[0]], [], {});
assert.deepEqual(partial.map((r) => [r.id, r.index]), [["h1", 38], ["x", null]]);

// 표시 형식: 순위 + 6열 + 숫자 열 1 + 검토 구분 = 9열(기존 추천 표와 같은 열 수), 칸 수 일치.
const view = facilityRecommendationView(namwon);
assert.equal(view.columns.length, 6);
assert.equal(1 + view.columns.length + (view.scoreColumn ? 1 : 0) + 1, 9);
for (const item of view.items) assert.equal(item.cells.length + 1, view.columns.length, item.id);
assert.deepEqual([view.rankColumn, ...view.columns, view.scoreColumn].filter((c) => c?.sortKey).map((c) => c.sortKey), ["rank", "utilization", "budget"]);
assert.equal(view.rankColumn.label, "순위");
assert.equal(view.scoreColumn.label, "운영 예산(억 원)");
assert.match(view.columns[2].label, /^활용도\(주거 입주율·개방 시설 이용률\)$/);
assert.equal(view.columns[3].label, "생활인구 관련 관측(외지인 방문·소비)");
// Top3·주 그래프는 같은 값(추천 지수): 큰 글씨는 숫자만, 지표 이름은 작은 글씨 칸.
assert.deepEqual(view.items.map((i) => [i.scoreText, i.scoreUnit]).slice(0, 3), [["73", "추천 지수"], ["72", "추천 지수"], ["61", "추천 지수"]]);
for (const item of view.items) assert.match(item.scoreText, /^\d+$/);
assert.deepEqual(view.items.map((i) => i.score), [73, 72, 61, 57, 54, 41]);
assert.equal(view.items[0].chartLabel, "남원 달빛정원 피오리움 · 복합화");
assert.deepEqual([view.scoreLabel, view.scoreMax], ["추천 지수(0~100)", 100]);
assert.deepEqual(view.items.map((i) => i.scoreCell), ["예산 자료 등록", "예산 자료 등록", "42", "예산 자료 등록", "28", "15"]);
assert.deepEqual(view.items.map((i) => view.status(i).text), ["우선 검토", "우선 검토", "우선 검토", "추가 검토", "추가 검토", "추가 검토"]);
assert.deepEqual(view.items.map((i, n) => view.rankText(n)).slice(0, 3), ["1순위", "2순위", "3순위"]);
assert.deepEqual(ids(view.tableItems), ids(view.items)); // 기본 표 순서 = 추천 순위
assert.match(view.footnote, /추천 지수\(0~100\) 높은 순/);
assert.match(view.footnote, /표 정렬 기준: 추천 순위/);
// 표만 다른 기준으로 정렬해도 Top3·주 그래프·순위 칸·검토 구분은 추천 순위 그대로.
const budgetView = facilityRecommendationView(namwon, "budget");
assert.deepEqual(ids(budgetView.items), ids(view.items));
assert.deepEqual(ids(budgetView.tableItems), ["fac_nw_farm01", "fac_nw_farm02", "fac_nw_rental", "fac_nw_fiorium", "fac_nw_glocal", "fac_nw_dudong"]);
assert.deepEqual(budgetView.tableItems.map((i) => i.rank), [3, 5, 6, 1, 2, 4]);
assert.deepEqual(budgetView.tableItems.map((i) => budgetView.status(i).text), ["우선 검토", "추가 검토", "추가 검토", "우선 검토", "우선 검토", "추가 검토"]);
assert.match(budgetView.footnote, /운영 예산 큰 순/);
assert.match(budgetView.footnote, /총 예산과 합산하지 않습니다/);
const utilView = facilityRecommendationView(namwon, "utilization");
assert.equal(utilView.tableItems[0].id, "fac_nw_rental");
assert.match(utilView.caption, /활용도 낮은 순/);
// 지수 값이 없는 행은 자료 등록 후 검토.
const missing = facilityRecommendationView(partial);
assert.deepEqual(missing.items.map((i) => [i.scoreText, i.scoreUnit, missing.status(i).text]), [["38", "추천 지수", "우선 검토"], ["", "자료 등록 후 산정", "자료 등록 후 검토"]]);

// 근거 그래프 묶음: 세 자리 고정. 데모는 지역 출처 값 두 개 + 추천 결과 구성, 실데이터(값 없음)는 같은 자리에 행동 안내만.
assert.deepEqual(EVIDENCE_CHART_SLOTS.map((s) => s.id), ["population", "case", "plans"]);
const views = { namwon: view, shinan: facilityRecommendationView(shinan) };
for (const id of ["namwon", "shinan"]) {
  const charts = evidenceChartsView(seed.recommendation_charts[id], views[id].items);
  assert.deepEqual(charts.map((c) => c.id), ["population", "case", "plans"], id);
  for (const chart of charts) {
    assert.equal(chart.empty, null, `${id} ${chart.id}`);
    assert.ok(chart.values.length >= 2 && chart.values.every(Number.isFinite), `${id} ${chart.id}`);
    assert.equal(chart.labels.length, chart.values.length);
    assert.ok(chart.scope && chart.note, `${id} ${chart.id}`);
  }
  // 출처 값 그래프: 시드에 출처 URL·확인일, 기준 기간을 그래프 범위 문구에 적는다. 전망치(2025·2026)는 섞지 않는다.
  for (const slot of ["population", "case"]) {
    const data = seed.recommendation_charts[id][slot];
    assert.equal(data.basis, "source", `${id} ${slot}`);
    assert.ok(data.sources.length && data.sources.every((s) => /^https:\/\//.test(s.url) && /^\d{4}-\d{2}-\d{2}$/.test(s.checked)), `${id} ${slot}`);
    for (const p of data.points) assert.doesNotMatch(p.label, /2025|2026|전망|목표/);
  }
  // 추천 활용 방안 구성은 추천 결과에서 센 시설 수.
  const plans = charts[2];
  assert.equal(plans.values.reduce((a, b) => a + b, 0), views[id].items.length);
}
assert.match(evidenceChartsView(seed.recommendation_charts.namwon, view.items)[0].scope, /2025년 연평균/);
assert.match(evidenceChartsView(seed.recommendation_charts.shinan, views.shinan.items)[0].scope, /2024년 4분기/);
assert.deepEqual(evidenceChartsView(seed.recommendation_charts.shinan, views.shinan.items)[1].values, [17, 36, 47]);
// 두 지역의 추천 활용 방안 구성이 다르다.
assert.notDeepEqual(evidenceChartsView(null, views.namwon.items)[2].values, evidenceChartsView(null, views.shinan.items)[2].values);
// 실데이터 모드(값 없음): 같은 세 자리, 값 0개, 행동 안내만.
const realCharts = evidenceChartsView(null, null);
assert.deepEqual(realCharts.map((c) => c.id), ["population", "case", "plans"]);
assert.deepEqual(realCharts.map((c) => c.empty), ["생활인구 통계를 연결하면 표시됩니다.", "시설 이용실적을 등록하면 표시됩니다.", "시설 자료를 등록하면 표시됩니다."]);
for (const c of realCharts) assert.deepEqual([c.labels, c.values, c.scope, c.note], [[], [], "", ""]);

// 추천 근거의 시설 사례 출처 행: 시설 자체 출처가 있는 시설만, 출처 제목만(기사 문장 없음).
const nwSourceRow = facilitySourceRow("namwon", seed.facilities);
assert.equal(nwSourceRow[0], "시설 사례 출처");
for (const name of ["남원 달빛정원 피오리움", "전북대 남원 글로컬캠퍼스(구 서남대)", "남원 옛 두동초 발효·체류시설"]) assert.match(nwSourceRow[1], new RegExp(name.replace(/[()]/g, "\\$&")));
assert.doesNotMatch(nwSourceRow[1], /스마트팜|임대주택/); // 유형 출처뿐이거나 출처 미확인 시설은 넣지 않는다
assert.equal(facilitySourceRow("gunwi", seed.facilities), null);

// 화면 문구: 계산 매개변수·계산식·가중치·내부 표기·생활인구 관련 관측 실수치 없음.
const chartTexts = ["namwon", "shinan"].flatMap((id) => evidenceChartsView(seed.recommendation_charts[id], views[id].items))
  .concat(realCharts).flatMap((c) => [c.title, c.scope, c.note, c.empty ?? "", ...c.labels]);
const screenTexts = [
  ...[view, budgetView, utilView, views.shinan].flatMap((v) => [v.title, v.scoreLabel, v.caption, v.basisTip, v.footnote,
    ...v.columns.map((c) => c.label), v.rankColumn.label, v.scoreColumn.label,
    ...v.items.flatMap((i) => [i.name, i.chartLabel, i.category, i.scoreText, i.scoreUnit, i.scoreCell, ...i.cells])]),
  ...FACILITY_EVIDENCE_ROWS.flat(), ...nwSourceRow, ...facilitySourceRow("shinan", seed.facilities), ...chartTexts,
  unlinkedControlNote(nwControls, { farm_scale: 30, staff: 100 })
];
const FORBIDDEN = /RICE|Reach|Impact|Confidence|Effort|점수|가중|비중|÷|×|√|\d+\s*호|room|link|readiness|데모|예시|시드|seed|mock|demo_example|구현|API|KT|BC/i;
for (const text of screenTexts) assert.doesNotMatch(text, FORBIDDEN, text);
for (const item of [...view.items, ...budgetView.items]) assert.equal(item.cells[2], "사용 승인 후 확인");
assert.match(FACILITY_EVIDENCE_ROWS[1][1], /공식 생활인구/);
assert.match(FACILITY_EVIDENCE_ROWS[0][1], /입주율\(입주 호수 기준\)/);
assert.match(FACILITY_EVIDENCE_ROWS[2][1], /추천 지수/);

// 패널 구조: 그래프 묶음은 표·페이지 표시·각주 다음 카드 끝에 한 번, 모드 분기 없이(두 모드 같은 자리·같은 구조).
const panelSrc = read("components", "PolicyRecommendation.jsx");
assert.equal(panelSrc.match(/data-slot="sim\.report\.charts"/g)?.length, 1);
const footIdx = panelSrc.indexOf("{view.footnote}");
const chartsIdx = panelSrc.indexOf('data-slot="sim.report.charts"');
const cardEndIdx = panelSrc.lastIndexOf("</Card>");
assert.ok(panelSrc.indexOf("<TablePager") < footIdx && footIdx < chartsIdx && chartsIdx < cardEndIdx);
assert.doesNotMatch(panelSrc.slice(footIdx, cardEndIdx), /realView|facilityRows\s*&&|\?\s*\(?\s*<section/);
assert.match(panelSrc, /\{chart\.empty && <ChartEmptyNote>\{chart\.empty\}<\/ChartEmptyNote>\}/); // 캔버스는 그대로, 안내만 겹침
assert.match(pageSrc, /chartSeed=\{appData\.recommendation_charts\?\.\[currentRegion\.id\]\}/);
assert.match(pageSrc, /demoRows=\{\[\.\.\.FACILITY_EVIDENCE_ROWS, facilitySourceRow\(currentRegion\.id, appData\.facilities\)\]\.filter\(Boolean\)\}/);
// 지도 시설물은 위치 자료가 있는 시설만(위치 미수집 시설로 지도가 깨지지 않게).
const mapSrc = read("components", "SimulatorMap.jsx");
assert.equal(mapSrc.match(/filter\(hasLocation\)/g)?.length, 2);
assert.equal(seed.facilities.filter((f) => Number.isFinite(f.lat) && Number.isFinite(f.lng)).length, 6);

console.log("facilityRecommendation: 출처·값 구분·지역별 유형·Top3 임대주택≤1·추천 지수 순위·근거 그래프 3자리·실데이터 안내·화면 문구 PASS");
