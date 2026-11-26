import test from "node:test";
import assert from "node:assert/strict";
import { csvRows } from "./hira.js";
import { aggregateRows, mergeOrphans, populationFor, POP_BANDS, regionList } from "./population.js";

// 1세 열을 0~110세 남녀로 만든 합성 CSV: 나이 a세 남자 = 1명, 여자 = 2명(110세이상 포함). 계 = 111×3 = 333
const ages = Array.from({ length: 110 }, (_, a) => a);
const header = ["행정기관코드", "기준연월", "시도명", "시군구명", "읍면동명", "계", "남자", "여자",
  ...ages.map((a) => `${a}세남자`), "110세이상 남자", ...ages.map((a) => `${a}세여자`), "110세이상 여자"];
const line = (code, sido, sgg, name) => [code, "2026-10-31", sido, sgg, name, "333", "111", "222", ...Array(111).fill("1"), ...Array(111).fill("2")].join(",");
const CSV = `${header.join(",")}\r\n${line("5219011300", "전북특별자치도", "남원시", "동충동")}\r\n${line("5219025000", "전북특별자치도", "남원시", "운봉읍")}\r\n`
  + `${line("5219025100", "전북특별자치도", "남원시", "운봉읍가출장소")}\r\n${line("5213011100", "전북특별자치도", "군산시", "해신동")}\r\n${line("3611000000", "세종특별자치시", "", "세종시")}\r\n`;

test("CSV 읽기·1세 → 9구간(합 = 계)·여성 3구간, 옛 필드(elderly70·70-79·80-89·90+·etc) 없음", () => {
  const { header: h, rows } = csvRows(CSV);
  const pop = aggregateRows(h, rows);
  assert.equal(pop.month, "2026-10-31");
  const d = pop.dongs["5219011300"];
  assert.deepEqual([d.region, d.sido, d.sgg, d.name, d.total], ["52190", "전북특별자치도", "남원시", "동충동", 333]);
  assert.equal(POP_BANDS.reduce((s, b) => s + d.bands[b.id], 0), d.total);
  assert.deepEqual([d.bands["0-9"], d.bands["50-64"], d.bands["65-74"], d.bands["85+"]], [30, 45, 30, 26 * 3]);
  assert.deepEqual(d.femaleBands, { "20-29": 20, "30-39": 20, "40-49": 20 });
  assert.equal(d.child09, 30);
  assert.deepEqual(Object.keys(d.bands), POP_BANDS.map((b) => b.id));
  assert.equal("elderly70" in d, false);
  assert.throws(() => aggregateRows(["행정기관코드"], []));
});

test("시군구 목록·지역 거르기·여러 지역 합치기·대표 지점 붙이기", () => {
  const { header: h, rows } = csvRows(CSV);
  const pop = aggregateRows(h, rows);
  assert.deepEqual(regionList(pop.dongs), [
    { code: "36110", sido: "세종특별자치시", name: "세종특별자치시" }, { code: "52130", sido: "전북특별자치도", name: "군산시" }, { code: "52190", sido: "전북특별자치도", name: "남원시" }
  ]);
  const one = populationFor(pop, ["52190"], { 5219011300: { lat: 35.4, lng: 127.38 } });
  assert.deepEqual(Object.keys(one.population.dongs).sort(), ["5219011300", "5219025000", "5219025100"]);
  assert.deepEqual([one.population.dongs["5219011300"].lat, one.population.dongs["5219025000"].lat], [35.4, undefined]);
  assert.equal(one.names["5219025000"], "운봉읍");
  const two = populationFor(pop, ["52190", "52130"]);
  assert.equal(two.population.total, 333 * 4);
  assert.equal(two.population.bands["20-29"], 30 * 4);
  assert.equal(populationFor(pop, ["11110"]).population, null);
});

test("코드 맞춤: 경계 없는 출장소는 이름이 앞부분인 같은 지역 경계에 합치고, 못 맞추면 남긴다", () => {
  const { header: h, rows } = csvRows(CSV);
  const pop = aggregateRows(h, rows);
  const boundaries = [{ code: "5219011300", name: "동충동" }, { code: "5219025000", name: "운봉읍" }, { code: "5213011100", name: "해신동" }, { code: "5213099900", name: "운봉읍" }];
  const { dongs, merged, unmatched } = mergeOrphans(pop.dongs, boundaries);
  assert.deepEqual(merged, [["5219025100", "5219025000"]]); // 다른 시군구의 같은 이름(5213…)은 보지 않음
  assert.deepEqual(unmatched, ["3611000000"]);
  assert.equal(dongs["5219025000"].total, 666);
  assert.equal(dongs["5219025000"].bands["65-74"], 60);
  assert.equal(dongs["5219025100"], undefined);
  assert.equal(pop.dongs["5219025000"].total, 333); // 원래 자료는 바꾸지 않음
});
