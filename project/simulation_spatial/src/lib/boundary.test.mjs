import test from "node:test";
import assert from "node:assert/strict";
import { boundariesByRegion, boundaryAnchors, boundaryBounds, boundaryOfPoint, insideAny, interiorPoint, pointInGeometry } from "./boundary.js";
import { dongUnits } from "./spatial.js";

// 합성 경계(좌표 [경도, 위도]): 정사각형, 구멍 난 정사각형, ㄷ자(무게중심이 밖), 두 조각
const square = (x, y, s = 1) => [[x, y], [x + s, y], [x + s, y + s], [x, y + s], [x, y]];
const SQ = { type: "Polygon", coordinates: [square(0, 0, 2)] };
const HOLED = { type: "Polygon", coordinates: [square(0, 0, 4), square(1, 1, 2)] };
const U = { type: "Polygon", coordinates: [[[0, 0], [3, 0], [3, 3], [2, 3], [2, 1], [1, 1], [1, 3], [0, 3], [0, 0]]] };
const MULTI = { type: "MultiPolygon", coordinates: [[square(10, 10, 1)], [square(20, 20, 3)]] };
const feature = (adm_cd2, adm_nm, geometry) => ({ type: "Feature", properties: { adm_cd2, adm_nm }, geometry });
const GEO = {
  type: "FeatureCollection",
  features: [
    feature("5219025000", "가도 나시 가읍", SQ),
    feature("5219031000", "가도 나시 나면", U),
    feature("5219051000", "가도 나시 다동", HOLED),
    feature("1111053000", "서울특별시 종로구 사직동", MULTI)
  ]
};

test("시군구별 경계 묶음·좌표로 행정동 배정·행정동 대표 지점(경계 안 중심점)", () => {
  const map = boundariesByRegion(GEO);
  assert.deepEqual([...map.keys()], ["52190", "11110"]);
  assert.deepEqual(map.get("52190").map((b) => [b.code, b.name]), [["5219025000", "가읍"], ["5219031000", "나면"], ["5219051000", "다동"]]); // 이름은 adm_nm 마지막 낱말
  assert.equal(boundariesByRegion(null).size, 0);
  const nw = map.get("52190");
  const dongOf = boundaryOfPoint(nw);
  assert.equal(dongOf({ lng: 1, lat: 1 }), "5219025000"); // SQ와 HOLED 바깥 고리가 겹치는 자리 — 먼저 나온 경계
  assert.equal(dongOf({ lng: 2.5, lat: 2.5 }), "5219031000");
  assert.equal(dongOf({ lng: 50, lat: 50 }), null);
  const anchors = boundaryAnchors(nw);
  assert.deepEqual(Object.keys(anchors), ["5219025000", "5219031000", "5219051000"]);
  for (const b of nw) assert.ok(pointInGeometry(anchors[b.code], b.geometry), b.code); // ㄷ자·구멍도 경계 안
});

test("점이 경계 안인지: 구멍 안은 밖, 여러 조각은 어느 하나라도 안", () => {
  assert.equal(pointInGeometry({ lng: 1, lat: 1 }, SQ), true);
  assert.equal(pointInGeometry({ lng: 3, lat: 1 }, SQ), false);
  assert.equal(pointInGeometry({ lng: 0.5, lat: 0.5 }, HOLED), true);
  assert.equal(pointInGeometry({ lng: 2, lat: 2 }, HOLED), false);
  assert.equal(pointInGeometry({ lng: 21, lat: 21 }, MULTI), true);
  assert.equal(pointInGeometry({ lng: 15, lat: 15 }, MULTI), false);
});

test("경계 안 중심점: 볼록하면 무게중심, ㄷ자는 경계 안 점으로 보정, 여러 조각은 가장 큰 조각", () => {
  assert.deepEqual(interiorPoint(SQ), { lat: 1, lng: 1 });
  const u = interiorPoint(U);
  assert.equal(pointInGeometry({ lng: 1.5, lat: 1.4 }, U), false); // 무게중심 부근은 파인 곳(밖)
  assert.ok(pointInGeometry(u, U), JSON.stringify(u));
  const m = interiorPoint(MULTI);
  assert.deepEqual([m.lng, m.lat], [21.5, 21.5]);
  assert.ok(pointInGeometry(interiorPoint(HOLED), HOLED));
});

test("계산 단위: 행정동마다 API가 붙인 경계 안 중심점, 경계 없는 행정동은 지점 없음, 65세 이상 비율", () => {
  const anchors = boundaryAnchors(boundariesByRegion(GEO).get("52190"));
  const bands = (old) => ({ "0-9": 5, "65-74": old, "75-84": 0, "85+": 0 });
  const population = { dongs: {
    5219025000: { total: 100, bands: bands(30), ...anchors["5219025000"] },
    5219051000: { total: 50, bands: bands(10), ...anchors["5219051000"] },
    5219099900: { total: 20, bands: bands(5) }
  } };
  const units = dongUnits(population, { 5219025000: "가읍", 5219051000: "다동" });
  assert.deepEqual(units.map((u) => [u.id, u.name, u.lat != null]), [["5219025000", "가읍", true], ["5219051000", "다동", true], ["5219099900", "5219099900", false]]);
  assert.deepEqual([units[0].lat, units[0].lng], [1, 1]);
  assert.equal(units[0].elderlyShare, 0.3);
});

test("지역 범위와 지역 안 판정(격자 칸 거르기)", () => {
  const b = boundariesByRegion(GEO).get("52190");
  assert.deepEqual(boundaryBounds(b), { south: 0, west: 0, north: 4, east: 4 });
  assert.equal(boundaryBounds([]), null);
  const inside = insideAny(b);
  assert.deepEqual([{ lat: 0.5, lng: 0.5 }, { lat: 3.5, lng: 3.5 }, { lat: 5, lng: 5 }].map(inside), [true, true, false]);
  assert.equal(insideAny([])({ lat: 99, lng: 99 }), true); // 경계가 없으면 거르지 않음
});
