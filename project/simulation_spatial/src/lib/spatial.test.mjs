import test from "node:test";
import assert from "node:assert/strict";
import {
  accessGrid, accessSummary, assignDongs, bandLabels, bandOf, basePopMax, dataBounds, DEFAULT_MIN_POP, DEFAULT_MIN_POP_SCALE, distanceKm, dongStatus,
  dongStatusLabels, dongUnits, minPopScale, padBounds, radiusText, regionStartKm, servedPop, targetPop, withRegions
} from "./spatial.js";

test("거리 시작값: 이름 끝 글자 시·구 3km, 군 5km(광역시 군·일반구·특별자치시 포함), 그 밖은 3km", () => {
  const cases = { 남원시: 3, 순창군: 5, 기장군: 5, "전주시 완산구": 3, 종로구: 3, 세종특별자치시: 3, 제주시: 3, "": 3 };
  for (const [name, km] of Object.entries(cases)) assert.equal(regionStartKm(name), km, name);
});

test("시·군을 섞어 고르면 움직이기 전까지 지역별 기준 거리로 판정, 움직이면 같은 값", () => {
  // 시설에서 4km 떨어진 두 읍면동: 시(11111) 쪽은 3km 기준이라 부족, 군(22222) 쪽은 5km 기준이라 이내
  const at = (code, km) => ({ id: `${code}00000`, name: code, lat: 35, lng: 127 + km * KM_LNG, pop: 100, elderlyShare: 0.5, childShare: 0.1 });
  const us = [at("11111", 4), at("22222", 4)];
  const fac = { lat: 35, lng: 127, dong: "1111100000" };
  const mixed = { radiusKm: 3, radiusByRegion: { 11111: 3, 22222: 5 }, growthPct: 0, minPop: 50 };
  const s = accessSummary(us, [fac], type, mixed);
  assert.deepEqual([s.shortage, s.radii, s.covered], [["1111100000"], [3, 5], 100]);
  assert.deepEqual(us.map((_, i) => dongStatus(s, i, mixed)), ["shortage", "within"]);
  assert.equal(radiusText(mixed), "3·5km");
  assert.deepEqual(bandLabels([3, 5]), ["3·5km 이내", "3·5~6·10km", "6·10~9·15km", "9·15km 초과"]);
  assert.equal(dongStatusLabels([3, 5], 20).shortage, "부족 지역 · 3·5km 밖·20명 이상");
  assert.equal(servedPop(us, fac, type, mixed), 0); // 시설이 시 쪽이라 3km 안만
  // 슬라이더를 움직이면 radiusByRegion이 없어져 모든 지역이 같은 값
  for (const r of [3, 4.5]) {
    const same = { ...mixed, radiusKm: r, radiusByRegion: null };
    const t = accessSummary(us, [fac], type, same);
    assert.deepEqual(t.radii, [r, r]);
    assert.equal(t.shortage.length, r < 4 ? 2 : 0);
    assert.equal(radiusText(same), `${r}km`);
  }
  // 격자 칸은 가장 가까운 읍면동의 시군구
  assert.deepEqual(withRegions([{ lat: 35, lng: 127 + 3.9 * KM_LNG }], [us[0], { ...us[1], lng: 127 + 9 * KM_LNG }]).map((c) => c.region), ["11111"]);
});

// 위도 35도 근처, 경도 0.011도 ≈ 1km 간격으로 한 줄 읍면동 3곳(합성)
const KM_LNG = 1 / (111.32 * Math.cos((35 * Math.PI) / 180));
const unit = (i, pop) => ({ id: `d${i}`, name: `읍면동${i}`, lat: 35, lng: 127 + i * KM_LNG, pop, elderlyShare: 0.5, childShare: 0.1 });
const units = [unit(0, 100), unit(1, 100), unit(2, 200)];
const type = { target: "all" };
const env = { radiusKm: 0.6, growthPct: 0, minPop: 50 }; // 1km = 1.67r, 2km = 3.33r (구간 경계를 피함)
const at0 = { lat: 35, lng: 127, dong: "d0" };

test("거리 근사는 1km 간격을 1km로 잰다", () => {
  assert.ok(Math.abs(distanceKm(units[0], units[1]) - 1) < 0.001);
});

test("연령 구간별 증감: Σ 구간 인구 × (1 + 구간 증감률), 없는 구간은 0", () => {
  const u = { id: "x", pop: 1000, elderlyShare: 0.25, childShare: 0.05, bands: { "65-74": 150, "75-84": 80, "85+": 20, "0-9": 50 } };
  assert.equal(targetPop(u, "elderly", {}), 250);
  assert.equal(targetPop(u, "elderly", { "75-84": 30 }), 150 + 80 * 1.3 + 20);
  assert.equal(targetPop(u, "child", { "0-9": -20, "75-84": 30 }), 40);
  assert.equal(targetPop(u, "all", 10), 1100); // 9구간이 다 없는 단위는 인구 × (1 + 증감률)
});

test("회귀: 모든 구간에 같은 증감을 주면 구간 계산과 바꾸기 전 계산(비율×증감)이 같다", () => {
  // 구간 인구 합 = 65세 이상 인구인 단위(구간 있음)와 같은 단위에서 구간을 뺀 것(비율 계산 경로)
  const withBands = units.map((x, i) => ({ ...x, bands: { "65-74": [30, 25, 60][i], "75-84": [15, 20, 30][i], "85+": [5, 5, 10][i], "0-9": [10, 10, 20][i] } }));
  for (const g of [-50, -15, 0, 25, 50]) {
    const all = { "65-74": g, "75-84": g, "85+": g, "0-9": g };
    withBands.forEach((x, i) => {
      assert.ok(Math.abs(targetPop(x, "elderly", all) - targetPop(units[i], "elderly", g)) < 1e-9);
      assert.ok(Math.abs(targetPop(x, "child", all) - targetPop(units[i], "child", g)) < 1e-9);
    });
    const e = { ...env, growthPct: all, minPop: 30 };
    const a = accessSummary(withBands, [at0], { target: "elderly" }, e);
    const b = accessSummary(units, [at0], { target: "elderly" }, { ...e, growthPct: g });
    assert.deepEqual([a.shortage, a.bands, Math.round(a.total)], [b.shortage, b.bands, Math.round(b.total)]);
  }
});

test("대상 인구는 비율과 증감률을 반영한다", () => {
  assert.equal(targetPop(units[0], "elderly", 0), 50);
  assert.equal(targetPop(units[0], "elderly", 20), 60);
  assert.equal(targetPop(units[0], "all", -200), 0);
});

test("행정동 단위: 인구·이름, 대표 지점 = API가 붙인 경계 안 중심점, 없으면 지점 없음", () => {
  const b = (old, kid) => ({ "0-9": kid, "65-74": old, "75-84": 0, "85+": 0 });
  const population = { dongs: { 1111052000: { total: 200, bands: b(50, 20) }, 1111051000: { total: 100, bands: b(10, 0), lat: 37.55, lng: 127.0 } } };
  const u = dongUnits(population, { 1111051000: "가동" });
  assert.deepEqual(u.map((x) => x.id), ["1111051000", "1111052000"]);
  assert.deepEqual([u[0].name, u[0].lat, u[0].lng], ["가동", 37.55, 127.0]);
  assert.equal(u[1].lat, undefined);
  assert.equal(targetPop(u[1], "elderly"), 50);
  assert.equal(targetPop(u[1], "child"), 20);
  assert.deepEqual(dongUnits(null), []);
});

test("부족 읍면동: 기준 거리 밖이고 최소 인구 이상", () => {
  const s = accessSummary(units, [at0], type, env);
  assert.deepEqual(s.shortage, ["d1", "d2"]);
  assert.equal(s.covered, 100);
  assert.deepEqual(s.bands, [100, 100, 0, 200]);
});

test("대표 지점 없는 단위는 부족 판정·기준 거리 밖 합·구간·평균 거리에 안 들어간다(시설을 더해도)", () => {
  const noPoint = [...units, { id: "d9", pop: 800, elderlyShare: 0, childShare: 0 }];
  for (const facs of [[at0], [at0, { lat: 35, lng: units[2].lng }]]) {
    const s = accessSummary(units, facs, type, env);
    const t = accessSummary(noPoint, facs, type, env);
    assert.ok(!t.shortage.includes("d9"));
    assert.equal(t.total - t.covered, s.total - s.covered);
    assert.deepEqual([t.bands, t.meanDistKm, t.coveredPct, t.shortage], [s.bands, s.meanDistKm, s.coveredPct, s.shortage]);
  }
});

test("시설 읍면동: 주소 코드가 있으면 그대로, 없으면 가장 가까운 대표 지점", () => {
  const out = assignDongs([at0, { lat: 35, lng: units[2].lng + 0.001 }, { lat: 35, lng: 127, dong: "없는코드" }], units);
  assert.deepEqual(out.map((f) => f.dong), ["d0", "d2", "d0"]);
});

test("데이터 범위와 여유, 접근 거리 격자", () => {
  const b = dataBounds([{ lat: 37.5, lng: 127 }, { lat: 37.6, lng: 127.2 }, { lat: null, lng: 1 }]);
  assert.deepEqual(b, { south: 37.5, west: 127, north: 37.6, east: 127.2 });
  assert.equal(dataBounds([]), null);
  const p = padBounds(b, 1);
  assert.ok(Math.abs((b.south - p.south) * 110.57 - 1) < 1e-9);
  const grid = accessGrid({ south: 35, west: 127, north: 35 + 2 / 110.57, east: 127 + 2 * KM_LNG }, [at0], 1);
  assert.equal(grid.length, 4);
  assert.ok(Math.abs(grid[0].dist - Math.SQRT1_2) < 0.01); // 첫 칸 중심(0.5km, 0.5km)
  assert.ok(grid.every((c) => Number.isFinite(c.dist)));
  assert.ok(accessGrid(b, [], 0.1, 100).length <= 121); // 칸 수 상한
  assert.ok(accessGrid(b, [], 1).every((c) => c.dist === Infinity));
});

test("인구 자료가 없는 지역(다른 지역 코드): 계산 단위·부족 판정이 비고 요약 값이 없다", () => {
  const u = dongUnits(null, { 5213025000: "가동" });
  assert.deepEqual(u, []);
  const s = accessSummary(u, [at0], type, env);
  assert.deepEqual([s.total, s.shortage.length, s.meanDistKm], [0, 0, null]);
});

test("읍면동 상태: 기준 거리 안 / 부족 / 기준 거리 밖(인구 적음), 부족 판정과 같은 규칙", () => {
  const s = accessSummary(units, [at0], type, { ...env, minPop: 150 });
  assert.deepEqual(units.map((_, i) => dongStatus(s, i, { ...env, minPop: 150 })), ["within", "outsideLow", "shortage"]);
  const statusShortage = units.map((u, i) => (dongStatus(s, i, { ...env, minPop: 150 }) === "shortage" ? u.id : null)).filter(Boolean);
  assert.deepEqual(statusShortage, s.shortage);
  assert.deepEqual(dongStatusLabels(2.9, 1450), { shortage: "부족 지역 · 2.9km 밖·1,450명 이상", within: "2.9km 이내", outsideLow: "2.9km 밖·1,450명 미만" });
});

test("기준 최댓값: 증감률 0일 때 대상 인구 최댓값, 다른 지역 코드 읍면동도 같은 규칙", () => {
  assert.equal(basePopMax(units, "all"), 200);
  assert.equal(basePopMax(units, "elderly"), 100);
  const b = (old, kid) => ({ "0-9": kid, "65-74": old, "75-84": 0, "85+": 0 });
  const other = dongUnits({ dongs: { 1111051500: { total: 4000, bands: b(600, 300) }, 1111053000: { total: 900, bands: b(200, 50) } } });
  assert.equal(basePopMax(other, "elderly"), 600);
  assert.equal(basePopMax(other, "child"), 300);
  assert.equal(basePopMax([], "elderly"), 0);
});

test("최소 대상 인구 슬라이더: 간격은 약 1/100(1·2·5×10ⁿ), 최댓값은 간격×10 단위로 올림", () => {
  assert.deepEqual(minPopScale(839), { max: 900, step: 10 });
  assert.deepEqual(minPopScale(600), { max: 600, step: 5 });
  assert.deepEqual(minPopScale(120), { max: 120, step: 1 });
  assert.deepEqual(minPopScale(30), { max: 30, step: 1 });
  assert.deepEqual(minPopScale(25000), { max: 26000, step: 20 }); // 1/100 간격 200이지만 기본 20명이 눈금에 맞게 20
  assert.deepEqual(minPopScale(0), DEFAULT_MIN_POP_SCALE);
  for (const v of [37, 839, 4321, 98765]) {
    const { max, step } = minPopScale(v);
    assert.ok(max >= v && max % step === 0 && max / step >= 10, `${v}: ${max}/${step}`);
    assert.equal(DEFAULT_MIN_POP % step, 0, `${v}: 기본 20명이 눈금 위`);
  }
});

test("거리 구간: 그래프·격자 공용 4구간, 경계는 아래 구간에 넣고 지점 없음(Infinity)은 3배 초과", () => {
  assert.deepEqual(bandLabels(2.9), ["2.9km 이내", "2.9~5.8km", "5.8~8.7km", "8.7km 초과"]); // 부동소수 꼬리 없음
  assert.deepEqual(bandLabels(3), ["3km 이내", "3~6km", "6~9km", "9km 초과"]);
  assert.deepEqual([0, 3, 3.01, 6, 6.01, 9, 9.01, Infinity].map((d) => bandOf(d, 3)), [0, 0, 1, 1, 2, 2, 3, 3]);
  const grid = accessGrid({ south: 35, west: 127, north: 35.05, east: 127.05 }, [at0]);
  assert.ok(grid.every((c) => bandOf(c.dist, 1) >= 0 && bandOf(c.dist, 1) <= 3));
});
