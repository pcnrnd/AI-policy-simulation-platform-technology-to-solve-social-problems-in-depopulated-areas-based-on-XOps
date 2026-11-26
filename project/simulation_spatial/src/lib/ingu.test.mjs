import test from "node:test";
import assert from "node:assert/strict";
import { AGE_BANDS, dbView, dongCode, groupOf, mercatorToLatLng, oldRegionCode, TARGET_AGE_LABEL, toFacilities } from "./ingu.js";

// 경도 126.98·위도 37.57 지점의 3857 좌표(합성)
const X = (126.98 * Math.PI / 180) * 6378137;
const Y = Math.log(Math.tan(Math.PI / 4 + (37.57 * Math.PI / 180) / 2)) * 6378137;

test("웹 메르카토르 → 위경도", () => {
  const p = mercatorToLatLng(String(X), String(Y));
  assert.ok(Math.abs(p.lng - 126.98) < 1e-9 && Math.abs(p.lat - 37.57) < 1e-9);
  assert.equal(mercatorToLatLng("0", "0"), null);
  assert.equal(mercatorToLatLng("", "abc"), null);
});

test("옛 코드 규칙: 앞 2자리 52→45, 51→42, 나머지는 같음", () => {
  assert.equal(oldRegionCode("52130"), "45130");
  assert.equal(oldRegionCode("51110"), "42110");
  assert.equal(oldRegionCode("11110"), "11110");
});

test("읍면동 코드: 옛 코드도 새 코드 8자리로 맞춘다", () => {
  assert.equal(dongCode("4513025021"), "52130250");
  assert.equal(dongCode("5213025000"), "52130250");
  assert.equal(dongCode("4211025300"), "51110253");
  assert.equal(dongCode("1111010100"), "11110101");
});

test("인구 증감 9구간과 대상별 구간·이름(노인 65세 이상)", () => {
  assert.deepEqual(AGE_BANDS.all.map((b) => b.label), ["0~9세", "10~19세", "20~29세", "30~39세", "40~49세", "50~64세", "65~74세", "75~84세", "85세 이상"]);
  assert.deepEqual(AGE_BANDS.elderly.map((b) => b.id), ["65-74", "75-84", "85+"]);
  assert.deepEqual(AGE_BANDS.youth.map((b) => b.id), ["20-29", "30-39", "40-49"]);
  assert.deepEqual([TARGET_AGE_LABEL.elderly, TARGET_AGE_LABEL.youthF], ["65세 이상", "20~49세 여성"]);
});

test("시설: 유형+번호 이름, 좌표 없는 행 제외, 읍면동 코드 대응", () => {
  const rows = [
    { fclty_ty: "방문요양", map_x: X, map_y: Y, addr10: "4513025021" },
    { fclty_ty: "방문요양", map_x: "0", map_y: "0", addr10: "4513025021" },
    { fclty_ty: "방문요양 ", map_x: X, map_y: Y, addr10: "4513031000" },
    { fclty_ty: "노인요양시설", map_x: X, map_y: Y, addr10: "4513031000" }
  ];
  const f = toFacilities(rows);
  assert.deepEqual(f.map((r) => r.name), ["방문요양 01", "방문요양 02", "노인요양시설 01"]);
  assert.deepEqual(f.map((r) => r.dong), ["52130250", "52130310", "52130310"]);
});

test("그룹 묶음 규칙: 유형 이름 키워드(공백·가운뎃점 무시), 안 맞으면 그 밖", () => {
  assert.equal(groupOf("방문요양서비스"), "visit");
  assert.equal(groupOf("재가장기요양기관 방문목욕"), "visit");
  assert.equal(groupOf("주·야간보호서비스"), "dayCare");
  assert.equal(groupOf("재가 주야간보호"), "dayCare");
  assert.equal(groupOf("단기보호"), "dayCare");
  assert.equal(groupOf("노인요양시설"), "residential");
  assert.equal(groupOf("노인요양공동생활가정"), "residential");
  assert.equal(groupOf("복지용구"), "other");
  assert.equal(groupOf("노인복지관"), "other");
});

test("DB 시설 → 데이터 기반 그룹(시설 있는 그룹만, 유형은 많은 순, 다른 지역 코드)", () => {
  const db = [
    { type: "방문목욕", name: "방문목욕 01", lat: 37.5, lng: 127, dong: "51110250" },
    { type: "방문요양", name: "방문요양 01", lat: 37.5, lng: 127, dong: "51110250" },
    { type: "방문요양", name: "방문요양 02", lat: 37.5, lng: 127, dong: "51110310" },
    { type: "복지용구", name: "복지용구 01", lat: 37.5, lng: 127, dong: "51110310" }
  ];
  const v = dbView(db);
  assert.deepEqual(v.groups.map((g) => [g.id, g.types.map((t) => t.label)]), [["visit", ["방문요양", "방문목욕"]], ["other", ["복지용구"]]]);
  assert.deepEqual(v.facilities.map((f) => f.typeId), ["t:방문목욕", "t:방문요양", "t:방문요양", "t:복지용구"]);
  assert.deepEqual(v.facilities.map((f) => f.id), ["f:t:방문목욕:37.500000,127.000000", "f:t:방문요양:37.500000,127.000000", "f:t:방문요양:37.500000,127.000000#2", "f:t:복지용구:37.500000,127.000000"]);
  // 같은 시설은 다른 시설과 함께 오거나 순서가 바뀌어도 같은 id(지역을 바꿔도 이어짐)
  assert.equal(dbView([db[3], db[0]]).facilities[0].id, v.facilities[3].id);
  assert.ok(v.types.every((t) => t.target === "elderly" && t.color));
  assert.ok(v.types.every((t) => TARGET_AGE_LABEL[t.target] === "65세 이상")); // 머리줄 대상 연령층
  assert.deepEqual(dbView([]), { groups: [], types: [], facilities: [] });
});

test("의료·보건 시설: 붙어 온 그룹·대상 그대로, 그룹은 기존 그룹 뒤, 첫 유형(기본 ◉)은 노인 돌봄 그대로", () => {
  const db = [
    { type: "방문요양", name: "방문요양 01", lat: 37.5, lng: 127, dong: "51110250" },
    { type: "의원", group: "health", target: "all", name: "의원 01", lat: 37.5, lng: 127 },
    { type: "의원", group: "health", target: "all", name: "의원 02", lat: 37.5, lng: 127 },
    { type: "산부인과", group: "health", target: "youthF", name: "산부인과 01", lat: 37.5, lng: 127 }
  ];
  const v = dbView(db);
  assert.deepEqual(v.groups.map((g) => [g.id, g.label, g.types.map((t) => `${t.label}:${t.target}`)]), [
    ["visit", "방문 서비스", ["방문요양:elderly"]], ["health", "의료·보건", ["의원:all", "산부인과:youthF"]]
  ]);
  assert.equal(v.types[0].label, "방문요양");
  assert.equal(groupOf("의원"), "other"); // 키워드로는 의료·보건이 되지 않음
});
