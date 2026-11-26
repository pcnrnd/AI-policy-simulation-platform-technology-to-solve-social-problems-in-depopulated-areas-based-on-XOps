import test from "node:test";
import assert from "node:assert/strict";
import { csvRows, healthFacilities, hiraGugunNames, inRegion, readCsv, sidoMatches } from "./hira.js";

test("CSV 읽기: 따옴표 안 쉼표·\"\"·줄바꿈, 머리줄 BOM, 거르기", () => {
  const text = '﻿암호화요양기호,주소,좌표(X)\r\nA1,"전북 남원시, 1",127.4\r\nA2,"따옴표 ""안"" 값\n둘째 줄",127.5\r\n\r\nA3,x,\r\n';
  const all = readCsv(text);
  assert.deepEqual(all.map((r) => r.암호화요양기호), ["A1", "A2", "A3"]);
  assert.equal(all[0].주소, "전북 남원시, 1");
  assert.equal(all[1].주소, '따옴표 "안" 값\n둘째 줄');
  assert.deepEqual(readCsv(text, (r) => r["좌표(X)"] !== "").map((r) => r.암호화요양기호), ["A1", "A2"]);
});

test("시도 이름: HIRA 줄임 ↔ 주소 코드표 전체 이름", () => {
  assert.ok(sidoMatches("전북", "전북특별자치도"));
  assert.ok(sidoMatches("전북", "전라북도"));
  assert.ok(sidoMatches("세종시", "세종특별자치시"));
  assert.ok(sidoMatches("경남", "경상남도"));
  assert.ok(sidoMatches("전남광주", "광주광역시"));
  assert.ok(!sidoMatches("전남", "전라북도"));
  assert.ok(!sidoMatches("대구", "대전광역시"));
  assert.ok(!sidoMatches("", "전라북도"));
  assert.ok(sidoMatches("광주", "전남광주통합특별시")); // 통합 시도
  assert.ok(!sidoMatches("경북", "경상남도"));
});

test("시군구 이름 후보: HIRA는 구를 시·광역시 이름과 붙여 쓴다", () => {
  const keep = (sido, gugun) => inRegion({ sido, gugun });
  assert.ok(keep("경기도", "수원시 장안구")(H("a", "의원", "경기", "수원장안구")));
  assert.ok(keep("부산광역시", "서구")(H("a", "의원", "부산", "부산서구")));
  assert.ok(!keep("부산광역시", "서구")(H("a", "의원", "대구", "대구서구")));
  assert.ok(keep("전남광주통합특별시", "서구")(H("a", "의원", "전남광주", "광주서구")));
  assert.ok(keep("세종특별자치시", "세종특별자치시")(H("a", "의원", "세종시", "세종시")));
  assert.ok(keep("서울특별시", "강서구")(H("a", "의원", "서울", "강서구")));
  assert.ok(!keep("부산광역시", "강서구")(H("a", "의원", "서울", "강서구")));
  assert.ok(hiraGugunNames({ sido: "경기도", gugun: "고양시 일산서구" }).includes("고양일산서구"));
});

test("CSV 행 읽기: 머리줄·배열 행, 하나씩 넘기기", () => {
  const { header, rows } = csvRows('a,b\r\n1,"x,y"\r\n\r\n2,z\r\n');
  assert.deepEqual(header, ["a", "b"]);
  assert.deepEqual(rows, [["1", "x,y"], ["2", "z"]]);
  const seen = [];
  assert.deepEqual(csvRows("a\n1\n2", (r) => seen.push(r[0])).rows, []);
  assert.deepEqual(seen, ["1", "2"]);
});

const H = (id, kind, sido, gugun, x = "127.4", y = "35.4") => ({ 암호화요양기호: id, 요양기관명: `기관${id}`, 종별코드명: kind, 시도코드명: sido, 시군구코드명: gugun, "좌표(X)": x, "좌표(Y)": y });

test("지역 행 거르기: 시군구 이름 + 시도(같은 이름 시군구는 시도로 가름)", () => {
  const keep = inRegion({ sido: "전북특별자치도", gugun: "남원시" });
  assert.ok(keep(H("a", "의원", "전북", "남원시")));
  assert.ok(!keep(H("b", "의원", "전북", "전주완산구")));
  assert.ok(!keep(H("c", "의원", "경남", "남원시")));
});

test("유형 나누기: 보건기관(보건소·지소·진료소)·의원·약국·산부인과(진료과목), 좌표 없는 행 제외, 이름은 유형+번호", () => {
  const hospitals = [
    H("h1", "보건소", "전북", "남원시"), H("h2", "보건지소", "전북", "남원시"), H("h3", "보건진료소", "전북", "남원시"),
    H("c1", "의원", "전북", "남원시"), H("c2", "의원", "전북", "남원시", "", ""), H("d1", "치과의원", "전북", "남원시"),
    H("k1", "한의원", "전북", "남원시"), H("b1", "병원", "전북", "남원시")
  ];
  const pharmacies = [H("p1", "약국", "전북", "남원시"), H("p2", "약국", "전북", "남원시")];
  const departments = [
    { 암호화요양기호: "c1", 진료과목코드명: "산부인과" }, { 암호화요양기호: "c1", 진료과목코드명: "내과" },
    { 암호화요양기호: "b1", 진료과목코드명: "산부인과" }, { 암호화요양기호: "k1", 진료과목코드명: "한방부인과" }
  ];
  const out = healthFacilities({ hospitals, pharmacies, departments });
  assert.deepEqual(out.map((f) => f.name), ["보건기관 01", "보건기관 02", "보건기관 03", "의원 01", "약국 01", "약국 02", "산부인과 01", "산부인과 02"]);
  assert.ok(out.every((f) => f.group === "health" && Number.isFinite(f.lat) && Number.isFinite(f.lng)));
  assert.deepEqual([...new Set(out.map((f) => `${f.type}:${f.target}`))], ["보건기관:all", "의원:all", "약국:all", "산부인과:youthF"]);
  assert.ok(out.every((f) => !Object.values(f).some((v) => String(v).startsWith("기관")))); // 기관명 안 씀
  assert.deepEqual(healthFacilities({}), []);
});
