import test from "node:test";
import assert from "node:assert/strict";
import { AGE_BANDS } from "./ingu.js";
import {
  actionsWithin, applyActions, basisDefaultType, changeVisible, growthByDong, meanRate, settledChange, basisTypeIds, groupBasis, mapFacilities, toggleBasis, capacityOf, INFLOW_SETTINGS, EXAMPLE_COUNTS, exampleInflowFacilities, livingByDong, maxRate, scenarioState, settledByDong, typeResult, usage,
  usageChanges, usageLevel, withInflow
} from "./scenario.js";
import { dbView } from "./ingu.js";
import { accessSummary, distanceKm, targetPop } from "./spatial.js";

// 경도 0.011°≈1km(위도 35.4), 동서로 늘어선 읍면동 4곳. 인구 증감 9구간(합 = 인구)
const IDS = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-64", "65-74", "75-84", "85+"];
const B = (...v) => Object.fromEntries(IDS.map((id, i) => [id, v[i]]));
const U = (id, lng, pop, bands, female) => ({ id, name: id, lat: 35.4, lng, pop, elderlyShare: (bands["65-74"] + bands["75-84"] + bands["85+"]) / pop, childShare: 0, bands, femaleBands: female });
const UNITS = [
  U("d1", 127.0, 555, B(11, 30, 21, 32, 47, 229, 100, 60, 25), { "20-29": 8, "30-39": 15, "40-49": 13 }),
  U("d2", 127.022, 1417, B(42, 80, 54, 72, 91, 508, 300, 200, 70), { "20-29": 17, "30-39": 24, "40-49": 32 }),
  U("d3", 127.044, 100, B(5, 5, 10, 10, 10, 30, 15, 10, 5), { "20-29": 5, "30-39": 5, "40-49": 5 }),
  U("d4", 127.066, 300, B(15, 15, 30, 30, 30, 90, 50, 30, 10), { "20-29": 15, "30-39": 15, "40-49": 15 })
];

test("분석 기준 없음: 결과(판정) 없음, 같은 줄을 다시 누르면 끔", () => {
  const st = scenarioState([], [], UNITS, INFLOW_SETTINGS);
  assert.equal(typeResult(st, null, { radiusKm: 3, growthPct: {}, minPop: 20 }, 1), null);
  assert.equal(toggleBasis(null, "t-a"), "t-a");
  assert.equal(toggleBasis("t-a", "t-a"), null);
  assert.equal(toggleBasis("t-a", "g:visit"), "g:visit");
  assert.deepEqual(basisTypeIds(null), []);
});

test("그룹 분석: 같은 대상이면 그 대상·유형 전부, 섞이면 전체 인구·대상이 다른 유형(산부인과)은 공급에서 뺌, 결과는 공급 유형 시설 전부로", () => {
  const visit = { id: "visit", label: "방문 서비스", color: "#a", types: [{ id: "t-a", target: "elderly" }, { id: "t-b", target: "elderly" }] };
  assert.deepEqual(groupBasis(visit), { id: "g:visit", label: "방문 서비스", color: "#a", target: "elderly", members: ["t-a", "t-b"] });
  const health = { id: "health", label: "의료·보건", color: "#h", types: [{ id: "t-clinic", target: "all" }, { id: "t-obgyn", target: "youthF" }, { id: "t-pharm", target: "all" }] };
  const hb = groupBasis(health);
  assert.equal(hb.target, "all");
  assert.deepEqual(hb.members, ["t-clinic", "t-pharm"]);
  // 그룹 결과 = 공급 유형 시설을 한 유형으로 본 결과와 같다(산부인과 시설은 들어가지 않음)
  const facs = [
    { id: "c1", typeId: "t-clinic", lat: 35.4, lng: 127.0, dong: "d1" },
    { id: "p1", typeId: "t-pharm", lat: 35.4, lng: 127.066, dong: "d4" },
    { id: "o1", typeId: "t-obgyn", lat: 35.4, lng: 127.044, dong: "d3" }
  ];
  const st = scenarioState(facs, [], UNITS, INFLOW_SETTINGS);
  const env = { radiusKm: 1.5, growthPct: {}, minPop: 0 };
  const g = typeResult(st, hb, env, 1);
  const asOne = typeResult({ ...st, facilities: facs.filter((f) => f.id !== "o1").map((f) => ({ ...f, typeId: "one" })) }, { id: "one", target: "all" }, env, 1);
  assert.deepEqual(g.summary, asOne.summary);
  assert.deepEqual(Object.keys(g.usage).sort(), ["c1", "p1"]);
  // 추가·전환 목록 기본값: 그룹이면 시설이 가장 많은 공급 유형
  assert.equal(basisDefaultType(hb, [...facs, { id: "p2", typeId: "t-pharm" }]), "t-pharm");
  assert.equal(basisDefaultType({ id: "t-a" }, facs), "t-a");
});

test("지도 시설 점: 표시 체크가 모두 꺼져도 분석 기준 공급 시설은 보이고, 체크는 기준을 바꾸지 않는다", () => {
  const facs = [{ id: "a", typeId: "t-a" }, { id: "b", typeId: "t-b" }, { id: "c", typeId: "t-c" }];
  const basis = { id: "g:x", members: ["t-a", "t-b"] };
  assert.deepEqual(mapFacilities(facs, new Set(), basis).map((f) => f.id), ["a", "b"]);
  assert.deepEqual(mapFacilities(facs, new Set(["t-c"]), basis).map((f) => f.id), ["a", "b", "c"]);
  assert.deepEqual(mapFacilities(facs, new Set(["t-c"]), null).map((f) => f.id), ["c"]);
});

test("인구 증감 9줄: 9줄에 같은 비율이면 어느 대상이든 대상 인구 × (1 + 비율), 대상 밖 줄을 바꿔도 그 대상 결과는 그대로", () => {
  for (const p of [-30, 0, 25]) {
    const same = Object.fromEntries(IDS.map((id) => [id, p]));
    for (const target of ["all", "elderly", "youth", "youthF", "child"]) {
      for (const u of UNITS) assert.ok(Math.abs(targetPop(u, target, same) - targetPop(u, target, {}) * (1 + p / 100)) < 1e-9, `${target} ${p}`);
    }
  }
  const outside = { elderly: ["0-9", "10-19", "20-29", "30-39", "40-49", "50-64"], youth: ["0-9", "10-19", "50-64", "65-74", "75-84", "85+"], child: IDS.slice(1) };
  for (const [target, ids] of Object.entries(outside)) {
    const g = Object.fromEntries(ids.map((id) => [id, 40]));
    for (const u of UNITS) assert.equal(targetPop(u, target, g), targetPop(u, target, {}), target);
    const env = { radiusKm: 1.5, minPop: 20 };
    const facs = [{ id: "x", lat: 35.4, lng: 127.0 }];
    assert.deepEqual(accessSummary(UNITS, facs, { target }, { ...env, growthPct: g }), accessSummary(UNITS, facs, { target }, { ...env, growthPct: {} }));
  }
  assert.equal(targetPop(UNITS[0], "elderly", {}), 185); // 65세 이상
});
const ALL = { id: "t-all", target: "all" };
const OLD = { id: "t-old", target: "elderly" };
const OB = { id: "t-ob", target: "youthF" };
const BASE = [
  { id: "f1", typeId: "t-all", lat: 35.4, lng: 127.0, dong: "d1" },
  { id: "f2", typeId: "t-all", lat: 35.4, lng: 127.066, dong: "d4" },
  { id: "o1", typeId: "t-old", lat: 35.4, lng: 127.0, dong: "d1" },
  { id: "o2", typeId: "t-old", lat: 35.4, lng: 127.044, dong: "d3" }
];
const ENV = { radiusKm: 1.5, growthPct: {}, minPop: 20 };
const SETTINGS = INFLOW_SETTINGS;

test("정주 유입 배분: 1곳당 정주인구 × 구간 비중(합으로 나눔, 모두 0이면 똑같이), 같은 읍면동은 더함", () => {
  const p = [{ kindId: "housing", dong: "d2" }, { kindId: "housing", dong: "d2" }, { kindId: "jobs", dong: "d3" }];
  const s = { ...SETTINGS, jobs: { settled: 30, living: 0, mix: { "20-29": 0, "30-39": 0, "40-49": 0 } } };
  assert.deepEqual(settledByDong(p, s), { d2: { "20-29": 100, "30-39": 80, "40-49": 20 }, d3: { "20-29": 10, "30-39": 10, "40-49": 10 } });
  assert.deepEqual(settledByDong([{ kindId: "culture", dong: "d1" }], SETTINGS), {}); // 정주 0명 종류
});

test("정주인구 변화 = 인구 증감분(모든 읍면동·모든 구간 인구 × 증감률) + 청년 지원 시설 정주 유입(1곳당 고정 가정값)", () => {
  const g = growthByDong(UNITS, { "20-29": 10, "65-74": -10 });
  assert.deepEqual(g, { d1: 2.1 - 10, d2: 5.4 - 30, d3: 1 - 1.5, d4: 3 - 5 });
  assert.deepEqual(growthByDong(UNITS, {}), {});
  const st = scenarioState(BASE, [{ id: 1, kind: "add", to: "housing", lat: 35.4, lng: 127.0, name: "h" }], UNITS, SETTINGS);
  const ch = settledChange(g, st.settled);
  assert.ok(Math.abs(ch.total - (100 + Object.values(g).reduce((s, n) => s + n, 0))) < 1e-9);
  assert.ok(Math.abs(ch.byDong.d1 - (100 + g.d1)) < 1e-9); // 청년 주택(정주 100명)은 d1(127.0)에
  assert.equal(st.livingTotal, 20);
  assert.deepEqual(settledChange({}, {}), { byDong: {}, total: 0 });
});

test("평균 사용률: 처음 상태는 100%, 시설을 더하면 그 유형 평균이 줄고, 띠는 처음과 다를 때만", () => {
  const capAll = capacityOf(UNITS, BASE.filter((f) => f.typeId === "t-all"), "all");
  const base = scenarioState(BASE, [], UNITS, SETTINGS);
  assert.ok(Math.abs(meanRate(typeResult(base, ALL, ENV, capAll).usage) - 1) < 1e-9);
  const st = scenarioState(BASE, [{ id: 1, kind: "add", to: "t-all", lat: 35.4, lng: 127.03, name: "a" }], UNITS, SETTINGS);
  assert.ok(Math.abs(meanRate(typeResult(st, ALL, ENV, capAll).usage) - 2 / 3) < 1e-9); // 같은 수요를 3곳이 나눔
  assert.equal(meanRate({}), null);
  assert.equal(changeVisible([], {}), false);
  assert.equal(changeVisible([], { "20-29": 0 }), false);
  assert.equal(changeVisible([], { "20-29": 10 }), true);
  assert.equal(changeVisible([{ id: 1 }], {}), true);
});

test("생활인구는 읍면동별로 따로 세고 대상 인구에는 들어가지 않는다", () => {
  const placements = [{ kindId: "culture", dong: "d1" }, { kindId: "jobs", dong: "d1" }];
  assert.deepEqual(livingByDong(placements, SETTINGS), { d1: 800 });
  const st = scenarioState(BASE, [{ id: 1, kind: "add", to: "culture", lat: 35.4, lng: 127.0, name: "c" }], UNITS, SETTINGS);
  assert.equal(st.livingTotal, 500);
  assert.equal(st.settledTotal, 0);
  assert.equal(st.units, UNITS); // 정주 0이면 계산 단위 그대로
});

test("정주 유입이 든 대상 인구: 전체 인구는 인원 그대로, 20~49세 여성은 그 구간 지금 여성 비율만, 증감률은 유입에 곱하지 않음, 노인은 무관", () => {
  const [d1] = withInflow(UNITS, { d1: { "20-29": 150 } });
  assert.equal(targetPop(d1, "all", {}), 555 + 150);
  assert.ok(Math.abs(targetPop(d1, "youthF", {}) - (36 + (150 * 8) / 21)) < 1e-9);
  assert.ok(Math.abs(targetPop(d1, "all", { "20-29": 100 }) - (555 + 21 + 150)) < 1e-9);
  assert.equal(targetPop(d1, "elderly", {}), targetPop(UNITS[0], "elderly", {}));
  const same = Object.fromEntries(AGE_BANDS.all.map((b) => [b.id, 30]));
  for (const u of UNITS) assert.ok(Math.abs(targetPop(u, "all", same) - u.pop * 1.3) < 1e-9); // 4줄 같은 비율 = 전체 × (1+비율)
});

test("사용률: 가장 가까운 시설에 배정, 처음 평균 100%, 추가하면 이웃 시설 사용률이 줄고 새 시설이 받는다", () => {
  const facs = BASE.filter((f) => f.typeId === "t-all");
  const cap = capacityOf(UNITS, facs, "all");
  assert.equal(cap, (555 + 1417 + 100 + 300) / 2);
  const before = usage(UNITS, facs, "all", {}, cap);
  assert.deepEqual(before.f1.dongs, ["d1", "d2"]);
  assert.deepEqual(before.f2.dongs, ["d3", "d4"]);
  assert.ok(Math.abs((before.f1.rate + before.f2.rate) / 2 - 1) < 1e-9);
  const st = scenarioState(BASE, [{ id: 1, kind: "add", to: "t-all", lat: 35.4, lng: 127.02, name: "새" }], UNITS, SETTINGS);
  const after = typeResult(st, ALL, ENV, cap).usage;
  assert.deepEqual(after["act-1"].dongs, ["d2"]);
  assert.ok(after.f1.rate < before.f1.rate);
  const ch = usageChanges(before, after);
  assert.deepEqual(ch.map((c) => c.id).sort(), ["act-1", "f1"]);
  assert.equal(ch.find((c) => c.id === "act-1").before, null);
  assert.deepEqual([usageLevel(0.5), usageLevel(1), usageLevel(1.31), usageLevel(null)], ["low", "mid", "high", null]);
  assert.equal(maxRate(after), Math.max(after.f1.rate, after.f2.rate, after["act-1"].rate));
});

test("전환: 원래 유형은 시설 1곳이 빠지고 새 유형은 같은 자리에 1곳이 생긴다(두 유형 결과가 함께 바뀜)", () => {
  const st = scenarioState(BASE, [{ id: 7, kind: "convert", from: "o2", to: "t-all", name: "전환" }], UNITS, SETTINGS);
  assert.deepEqual(st.facilities.filter((f) => f.typeId === "t-old").map((f) => f.id), ["o1"]);
  const moved = st.facilities.find((f) => f.id === "act-7");
  assert.deepEqual([moved.typeId, moved.lng, moved.dong, moved.converted], ["t-all", 127.044, "d3", true]);
  const base = scenarioState(BASE, [], UNITS, SETTINGS);
  const capOld = capacityOf(UNITS, BASE.filter((f) => f.typeId === "t-old"), "elderly");
  const oldBefore = typeResult(base, OLD, ENV, capOld);
  const oldAfter = typeResult(st, OLD, ENV, capOld);
  assert.ok(oldAfter.summary.meanDistKm > oldBefore.summary.meanDistKm);
  assert.ok(oldAfter.usage.o1.rate > oldBefore.usage.o1.rate);
  // 주소 코드 읍면동이 가장 가까운 대표 지점과 다른 시설도 전환 뒤 같은 읍면동
  const far = [...BASE, { id: "o9", typeId: "t-old", lat: 35.4, lng: 127.044, dong: "d1" }];
  assert.equal(scenarioState(far, [{ id: 9, kind: "convert", from: "o9", to: "t-all", name: "x" }], UNITS, SETTINGS).facilities.find((f) => f.id === "act-9").dong, "d1");
  const toKind = scenarioState(BASE, [{ id: 8, kind: "convert", from: "f2", to: "housing", name: "주택" }], UNITS, SETTINGS);
  assert.equal(toKind.facilities.some((f) => f.id === "f2"), false);
  assert.deepEqual(toKind.placements.map((p) => [p.kindId, p.dong]), [["housing", "d4"]]);
  assert.equal(toKind.settledTotal, 100);
});

test("같은 좌표 시설은 수요를 똑같이 나누고, 한 곳을 전환하면 남은 곳 사용률이 오른다(값이 같은 시설은 영향 시설에 안 들어감)", () => {
  const twin = [
    { id: "o1", typeId: "t-old", lat: 35.4, lng: 127.0, dong: "d1" },
    { id: "o2", typeId: "t-old", lat: 35.4, lng: 127.066, dong: "d4" },
    { id: "o3", typeId: "t-old", lat: 35.4, lng: 127.0, dong: "d1" }
  ];
  const cap = capacityOf(UNITS, twin, "elderly");
  const before = usage(UNITS, twin, "elderly", {}, cap);
  assert.equal(before.o1.rate, before.o3.rate);
  assert.ok(before.o1.rate > 0);
  const st = scenarioState(twin, [{ id: 1, kind: "convert", from: "o1", to: "t-all", name: "x" }], UNITS, SETTINGS);
  const after = typeResult(st, OLD, ENV, cap).usage;
  assert.ok(Math.abs(after.o3.rate - 2 * before.o3.rate) < 1e-9);
  const ch = usageChanges(before, after);
  assert.deepEqual(ch.map((c) => c.id).sort(), ["o1", "o3"]); // o2(먼 시설)는 값이 같아 빠짐
  assert.equal(ch.find((c) => c.id === "o1").after, null);
  const tiny = { a: { rate: 0.361, dongs: ["d1"] } };
  assert.deepEqual(usageChanges(tiny, { a: { rate: 0.3612, dongs: ["d2"] } }), []); // 화면 값(%)이 같으면 영향 아님
});

test("예시 유입 시설: 지역 코드로 고정, 종류별 수, 청년이 사는 읍면동 대표 지점에서 0.3~0.8km, 유형 id = 종류 id, 유입은 더하지 않음", () => {
  const a = exampleInflowFacilities(UNITS, "52190");
  assert.deepEqual(a, exampleInflowFacilities(UNITS, "52190"));
  assert.notDeepEqual(a.map((f) => f.lat), exampleInflowFacilities(UNITS, "52130").map((f) => f.lat));
  const total = Object.values(EXAMPLE_COUNTS).reduce((s, n) => s + n, 0);
  assert.equal(a.length, total);
  for (const f of a) {
    const u = UNITS.find((x) => x.id === f.dong);
    const km = distanceKm(u, f);
    assert.ok(km >= 0.3 - 1e-6 && km <= 0.8 + 1e-6, `${f.name} ${km}`);
    assert.equal(f.group, "youth");
  }
  assert.deepEqual(a.filter((f) => f.typeId === "housing").map((f) => f.name), ["청년 주택 01", "청년 주택 02", "청년 주택 03", "청년 주택 04"]);
  assert.deepEqual(exampleInflowFacilities([], "52190"), []);
  const v = dbView([...BASE.map((f) => ({ type: f.typeId, name: f.id, lat: f.lat, lng: f.lng })), ...a]);
  const youth = v.groups.find((g) => g.id === "youth");
  assert.equal(v.groups.at(-1).id, "youth");
  assert.ok(youth.types.every((t) => t.target === "youth" && EXAMPLE_COUNTS[t.id] === t.count));
  const st = scenarioState(assignBase(v.facilities), [], UNITS, SETTINGS);
  assert.deepEqual([st.placements, st.units, st.settledTotal], [[], UNITS, 0]); // 예시 시설은 유입 0
});
const assignBase = (facs) => facs.map((f) => ({ ...f, dong: f.dong ?? "d1" }));

test("유입 시설 종류로 추가하면 그 유형 시설이 생기고 유입도 생긴다", () => {
  const st = scenarioState(BASE, [{ id: 3, kind: "add", to: "housing", lat: 35.4, lng: 127.022, name: "청년 주택 추가 3" }], UNITS, SETTINGS);
  assert.deepEqual(st.facilities.filter((f) => f.typeId === "housing").map((f) => [f.id, f.dong]), [["act-3", "d2"]]);
  assert.deepEqual(st.placements.map((p) => p.kindId), ["housing"]);
  assert.equal(st.settledTotal, 100);
  assert.equal(targetPop(st.units[1], "youth", {}), 54 + 72 + 91 + 100);
});

test("고른 지역 안 동작만: 추가는 자리가 지역 안, 전환은 원래 시설이 지금 목록에 있을 때(지역 밖은 계산·목록에서 빠짐)", () => {
  const actions = [
    { id: 1, kind: "add", to: "t-all", lat: 35.4, lng: 127.02 },
    { id: 2, kind: "add", to: "t-all", lat: 36.9, lng: 128.5 },
    { id: 3, kind: "convert", from: "o1", to: "t-all" },
    { id: 4, kind: "convert", from: "다른 지역 시설", to: "t-all" }
  ];
  const inside = (p) => p.lat < 36;
  assert.deepEqual(actionsWithin(actions, BASE, inside).map((a) => a.id), [1, 3]);
  assert.deepEqual(actionsWithin([], BASE, inside), []);
});

test("회귀: 동작 0이면 지금과 같음(시설·계산 단위·요약), 유입 0이면 같음", () => {
  const st = scenarioState(BASE, [], UNITS, SETTINGS);
  assert.deepEqual(st.facilities, BASE);
  assert.equal(st.units, UNITS);
  assert.deepEqual(accessSummary(st.units, BASE.filter((f) => f.typeId === "t-all"), ALL, ENV), accessSummary(UNITS, BASE.filter((f) => f.typeId === "t-all"), ALL, ENV));
  assert.equal(withInflow(UNITS, {}), UNITS);
  assert.deepEqual(applyActions(BASE, [], UNITS).placements, []);
  const ob = typeResult(st, OB, ENV, null);
  assert.ok(Object.values(ob.usage).every((v) => v.rate === null)); // 산부인과 시설 없음
});
