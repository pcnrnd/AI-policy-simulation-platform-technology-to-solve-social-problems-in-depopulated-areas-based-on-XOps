import assert from "node:assert/strict";
import { MAX_PER_REGION, PRESET_SCENARIOS, parseScenarioStore, serializeScenarioStore, withPresetScenarios } from "./scenarioStore.js";

const namwon = { id: "namwon", case: { simulation: { controls: [{ key: "clinic", default: 2 }] } } };
const shinan = { id: "shinan" };
const other = { id: "elsewhere" };
const empty = { scenarios: [], presets: {} };

// 예전 형식(배열)·손상값을 읽는다
assert.deepEqual(parseScenarioStore(JSON.stringify([{ id: "a", regionId: "namwon" }, { bad: 1 }])),
  { scenarios: [{ id: "a", regionId: "namwon" }], presets: {} });
assert.deepEqual(parseScenarioStore("{oops"), empty);
assert.deepEqual(parseScenarioStore(null), empty);

// 빈 저장소: 기본 3개 + 지역 표식을 한 번에, 입력값은 지역 기본 시설 값
const first = withPresetScenarios(empty, namwon, "2026-11-03T00:00:00Z");
assert.equal(first.scenarios.length, PRESET_SCENARIOS.length);
assert.deepEqual(first.presets, { namwon: true });
assert.deepEqual(first.scenarios[0].controlValues, { clinic: 2 });
assert.ok(first.scenarios.every((s) => s.regionId === "namwon" && s.id.startsWith("namwon-preset-")));
// 같은 지역은 다시 넣지 않는다(사용자가 전부 지운 뒤에도 표식이 남아 재주입 금지)
assert.equal(withPresetScenarios({ scenarios: [], presets: { namwon: true } }, namwon), null);
// 대상이 아닌 지역은 넣지 않는다
assert.equal(withPresetScenarios(empty, other), null);
// 기존 행 보존, 7개면 남은 1자리만, 8개면 0개(표식만)
const seven = Array.from({ length: 7 }, (_, i) => ({ id: `u${i}`, regionId: "shinan" }));
const fill = withPresetScenarios({ scenarios: seven, presets: {} }, shinan);
assert.equal(fill.scenarios.filter((s) => s.regionId === "shinan").length, MAX_PER_REGION);
assert.deepEqual(fill.scenarios.slice(0, 7), seven);
const eight = [...seven, { id: "u7", regionId: "shinan" }];
const full = withPresetScenarios({ scenarios: eight, presets: {} }, shinan);
assert.equal(full.scenarios.length, 8);
assert.deepEqual(full.presets, { shinan: true });
// 다른 지역 행은 세지 않는다
assert.equal(withPresetScenarios({ scenarios: eight, presets: {} }, namwon).scenarios.length, 8 + PRESET_SCENARIOS.length);
// 저장 형식: 표식이 없으면 배열(실데이터 키 형식 불변), 있으면 객체 — 왕복 일치
assert.equal(serializeScenarioStore({ scenarios: seven, presets: {} }), JSON.stringify(seven));
assert.deepEqual(parseScenarioStore(serializeScenarioStore(first)), first);
console.log("PASS: 시나리오 저장 형식 호환, 지역별 기본 시나리오 1회·표식·남은 자리·기존 행 보존");
