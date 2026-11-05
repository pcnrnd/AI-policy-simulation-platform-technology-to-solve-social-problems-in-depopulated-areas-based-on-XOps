// 시설 × 활용 방안 추천 순수 계산 검사 (vitest 미도입 → node 직접 실행).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  DEFAULT_CRITERION, FACILITY_EVIDENCE_ROWS, SORT_CRITERIA,
  facilityRecommendationView, facilityRows, linkedControls, recoControlSig, sortFacilityRows, unlinkedControlNote
} from "./facilityRecommendation.js";

const here = dirname(fileURLToPath(import.meta.url));
const read = (...path) => readFileSync(join(here, "..", ...path), "utf-8");
const seed = JSON.parse(read("assets", "mock_data.json"));
const regionOf = (id) => seed.regions.find((r) => r.id === id);
const controlsOf = (id) => regionOf(id).case.simulation.controls;

// 지역 시설 자료만 쓴다 — 남원 3·신안 3, 새 시설을 만들지 않는다.
const namwon = facilityRows("namwon", seed.facilities, controlsOf("namwon"), {});
const shinan = facilityRows("shinan", seed.facilities, controlsOf("shinan"), {});
assert.equal(seed.facilities.length, 6);
assert.deepEqual([namwon.length, shinan.length], [3, 3]);
assert.deepEqual([...namwon, ...shinan].map((r) => r.id).sort(), seed.facilities.map((f) => f.id).sort());
assert.deepEqual(facilityRows("gunwi", seed.facilities, [], {}), []);
assert.deepEqual(facilityRows("namwon", undefined), []);

// 활용도: 주거 시설 입주율만 값이 있고, 나머지는 빈 값(화면에서 행동 안내).
const byId = Object.fromEntries([...namwon, ...shinan].map((r) => [r.id, r]));
assert.equal(byId.fac_nw_rental.utilization, 91);
assert.equal(byId.fac_sa_rental.utilization, 84);
assert.equal(byId.fac_nw_rental.utilizationLabel, "입주율");
for (const id of ["fac_nw_farm01", "fac_nw_farm02", "fac_sa_solar", "fac_sa_garden"]) assert.equal(byId[id].utilization, null, id);
assert.deepEqual(namwon.map((r) => r.budget), [42, 28, 15]);
// 활용 방안에 통합 검토는 없다.
for (const row of [...namwon, ...shinan]) assert.doesNotMatch(`${row.planType} ${row.plan}`, /통합/);

// 조절변수 연결: 신안 임대주택 공급 호수만 시설 값에 연결되고, 남원 두 변수는 연결 대상이 없다.
assert.deepEqual(linkedControls(controlsOf("shinan")).map((c) => c.key), ["rental_units"]);
assert.deepEqual(linkedControls(controlsOf("namwon")), []);
assert.deepEqual(linkedControls(undefined), []);
const shinanRental = (supply) => facilityRows("shinan", seed.facilities, controlsOf("shinan"), { rental_units: supply })
  .find((r) => r.id === "fac_sa_rental").utilization;
assert.equal(shinanRental(300), 84); // 기본값 = 시설 자료 공급 호수 → 관측값 그대로
assert.equal(shinanRental(600), 42); // 공급을 늘리면 현재 입주 호수 기준 입주율이 낮아진다
assert.equal(shinanRental(50), 100); // 공급보다 많이 입주할 수 없다
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
// 공급 호수를 바꿔 다시 도출하면 추천 계산 결과(입주율·표시값)가 실제로 바뀐다.
const saView = (values) => facilityRecommendationView(facilityRows("shinan", seed.facilities, saControls, values));
assert.deepEqual(saView({}).items.map((i) => [i.scoreText, i.scoreUnit, i.cells[1]]), [["84", "%(입주율)", "입주율 84%"], ["", "이용실적 등록", "이용실적 등록"], ["", "이용실적 등록", "이용실적 등록"]]);
assert.deepEqual(saView({ rental_units: 600 }).items.map((i) => [i.scoreText, i.cells[1]]), [["42", "입주율 42%"], ["", "이용실적 등록"], ["", "이용실적 등록"]]);

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

// 정렬: 기준 하나, 값 없는 시설은 방향과 관계없이 뒤, 같은 값은 자료 순서 유지.
assert.equal(DEFAULT_CRITERION, "utilization");
assert.deepEqual(Object.keys(SORT_CRITERIA), ["utilization", "budget"]);
assert.deepEqual(sortFacilityRows(namwon, "utilization").map((r) => r.id), ["fac_nw_rental", "fac_nw_farm01", "fac_nw_farm02"]);
assert.deepEqual(sortFacilityRows(namwon, "budget").map((r) => r.id), ["fac_nw_farm01", "fac_nw_farm02", "fac_nw_rental"]);
assert.deepEqual(sortFacilityRows(shinan, "budget").map((r) => r.budget), [33, 21, 9]);
const rows = [
  { id: "a", utilization: 70, budget: 5 }, { id: "b", utilization: null, budget: 9 },
  { id: "c", utilization: 40, budget: 5 }, { id: "d", utilization: 70, budget: null }
];
assert.deepEqual(sortFacilityRows(rows, "utilization").map((r) => r.id), ["c", "a", "d", "b"]);
assert.deepEqual(sortFacilityRows(rows, "budget").map((r) => r.id), ["b", "a", "c", "d"]);
assert.deepEqual(rows.map((r) => r.id), ["a", "b", "c", "d"]); // 원본 배열은 그대로

// 조절변수가 바뀌면 순위가 다시 도출된다 — 같은 지역 주거 시설 두 곳 중 연결된 시설의 공급 호수를 늘리면 순서가 바뀐다.
const twin = [
  { id: "h1", name: "주거 1", type: "임대주택", region_id: "r", operating_budget: 10, supply_units: 100, occupancy_rate: 80 },
  { id: "h2", name: "주거 2", type: "임대주택", region_id: "r", operating_budget: 10, supply_units: 100, occupancy_rate: 70 }
];
const twinControls = [{ key: "units", default: 100, facility_id: "h1", field: "supply_units" }];
const order = (values) => facilityRecommendationView(facilityRows("r", twin, twinControls, values)).items.map((i) => i.id);
assert.deepEqual(order({ units: 100 }), ["h2", "h1"]);
assert.deepEqual(order({ units: 200 }), ["h1", "h2"]); // 80% → 40%

// 표시 형식: 순위 + 6열 + 숫자 열 1 + 검토 구분 = 9열(기존 추천 표와 같은 열 수), 칸 수 일치.
const view = facilityRecommendationView(namwon);
assert.equal(view.columns.length, 6);
assert.equal(1 + view.columns.length + (view.scoreColumn ? 1 : 0) + 1, 9);
for (const item of view.items) assert.equal(item.cells.length + 1, view.columns.length, item.id);
assert.deepEqual([...view.columns, view.scoreColumn].filter((c) => c.sortKey).map((c) => c.sortKey), ["utilization", "budget"]);
assert.equal(view.scoreColumn.label, "운영 예산(억 원)");
assert.deepEqual(view.items.map((i) => i.scoreCell), ["15", "42", "28"]);
assert.match(view.columns[2].label, /^활용도\(주거 입주율·개방 시설 이용률\)$/);
assert.equal(view.columns[3].label, "생활인구 관련 관측(외지인 방문·소비)");
// Top3 값 줄: 큰 글씨는 숫자만(한글은 작은 글씨 칸) — 기존 "3,061 점" 줄과 같은 높이를 지킨다.
assert.deepEqual(view.items.map((i) => [i.scoreText, i.scoreUnit]), [["91", "%(입주율)"], ["", "이용실적 등록"], ["", "이용실적 등록"]]);
for (const item of view.items) assert.match(item.scoreText, /^[\d,]*$/);
assert.deepEqual(view.items.map((i) => i.score), [91, null, null]);
assert.deepEqual(view.items.map((i, n) => view.status(i, n).text), ["우선 검토", "자료 등록 후 검토", "자료 등록 후 검토"]);
assert.deepEqual(view.items.map((i, n) => view.rankText(n)), ["1순위", "2순위", "3순위"]);
assert.match(view.footnote, /활용도 낮은 순/);
const budgetView = facilityRecommendationView(namwon, "budget");
assert.deepEqual(budgetView.items.map((i) => [i.scoreText, i.scoreUnit]), [["42", "억 원(운영 예산)"], ["28", "억 원(운영 예산)"], ["15", "억 원(운영 예산)"]]);
assert.equal(budgetView.scoreLabel, "운영 예산(억 원)");
assert.match(budgetView.footnote, /운영 예산 큰 순/);
assert.match(budgetView.footnote, /총 예산과 합산하지 않습니다/);
// 4위부터는 추가 검토, 정렬 기준 값이 없으면 순위와 관계없이 자료 등록 후 검토.
const four = facilityRecommendationView([40, 10, null, 30, 20].map((u, n) => ({ id: `f${n}`, name: `f${n}`, plan: "", utilization: u, utilizationLabel: "입주율", budget: 1 })));
assert.deepEqual(four.items.map((i) => i.id), ["f1", "f4", "f3", "f0", "f2"]);
assert.deepEqual(four.items.map((i, n) => four.status(i, n)), [
  { text: "우선 검토", strong: true }, { text: "우선 검토", strong: true }, { text: "우선 검토", strong: true },
  { text: "추가 검토", strong: false }, { text: "자료 등록 후 검토", strong: false }
]);

// 화면 문구: 계산 매개변수·계산식·내부 표기·생활인구 관련 관측 실수치 없음.
const screenTexts = [
  view.title, view.scoreLabel, view.caption, view.basisTip, view.footnote, budgetView.footnote,
  ...view.columns.map((c) => c.label), view.scoreColumn.label, ...view.items.flatMap((i) => [i.name, i.category, i.scoreText, i.scoreUnit, i.scoreCell, ...i.cells]),
  ...budgetView.items.flatMap((i) => [i.scoreText, i.scoreUnit]), ...FACILITY_EVIDENCE_ROWS.flat(),
  unlinkedControlNote(nwControls, { farm_scale: 30, staff: 100 })
];
const FORBIDDEN = /RICE|Reach|Impact|Confidence|Effort|점수|가중|÷|×|√|\d+\s*호|데모|예시|시드|seed|mock|구현|API|KT|BC/i;
for (const text of screenTexts) assert.doesNotMatch(text, FORBIDDEN, text);
for (const item of [...view.items, ...budgetView.items]) assert.equal(item.cells[2], "사용 승인 후 확인");
assert.match(FACILITY_EVIDENCE_ROWS[1][1], /공식 생활인구/);
assert.match(FACILITY_EVIDENCE_ROWS[0][1], /입주율\(입주 호수 기준\)/);

console.log("facilityRecommendation: 시설 6건·활용도 정의·조절변수 연결·단일 기준 정렬·9열 표시·화면 문구 PASS");
