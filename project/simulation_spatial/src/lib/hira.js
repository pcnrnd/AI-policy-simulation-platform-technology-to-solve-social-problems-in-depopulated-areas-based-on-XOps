// 건강보험심사평가원 "전국 병의원 및 약국 현황"(공공데이터포털 15051059, 공공누리 1유형) CSV → 의료·보건 시설.
// 한 시군구 행만 고르고, 기관명·주소·전화는 읽지 않고 유형+번호로 쓴다. DB·파일 접근 없이 시험할 수 있도록 순수 함수만 둔다.

/** CSV 글 → { header, rows }(행은 배열, 따옴표·""·줄바꿈 처리, 빈 줄 뺌). onRow가 있으면 행을 모으지 않고 하나씩 넘긴다. */
export function csvRows(text, onRow) {
  const rows = [];
  let header = null;
  let row = [];
  let cell = "";
  let quoted = false;
  const endRow = () => {
    row.push(cell);
    cell = "";
    if (!header) header = row.map((h) => h.replace(/^﻿/, "").trim());
    else if (row.length > 1 || row[0] !== "") {
      if (onRow) onRow(row, header);
      else rows.push(row);
    }
    row = [];
  };
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') { cell += '"'; i += 1; }
      else if (ch === '"') quoted = false;
      else cell += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") { row.push(cell); cell = ""; }
    else if (ch === "\n") endRow();
    else if (ch !== "\r") cell += ch;
  }
  if (cell !== "" || row.length) endRow();
  return { header: header ?? [], rows };
}

/** CSV 글 → 머리줄 기준 객체 배열. keep(객체)이 참인 행만 남긴다. */
export function readCsv(text, keep = () => true) {
  const out = [];
  csvRows(text, (row, header) => {
    const obj = Object.fromEntries(header.map((h, i) => [h, row[i] ?? ""]));
    if (keep(obj)) out.push(obj);
  });
  return out;
}

/**
 * HIRA 시도 이름(전북·세종시·전남광주 등 줄임) ↔ 주소 코드표 시도 이름(전북특별자치도·전라북도 등).
 * 줄임 이름을 두 글자씩 나눠, 하나라도 전체 이름의 앞 글자이거나 (첫 글자 같고 둘째 글자를 포함)하면 같은 시도로 본다.
 */
export function sidoMatches(short, full) {
  const s = String(short ?? "").replace(/시$/, "");
  const f = String(full ?? "");
  if (!s || !f) return false;
  const parts = s.match(/.{1,2}/g);
  // "광주"(HIRA) ↔ "전남광주통합특별시"처럼 통합 시도는 두 글자가 이름 가운데에 있어도 같은 시도로 본다
  return parts.some((p) => f.startsWith(p) || (p.length === 2 && (f.includes(p) || (f[0] === p[0] && f.includes(p[1])))));
}

/**
 * 인구 파일 시군구명 → HIRA 시군구코드명 후보. HIRA는 구를 시·광역시 이름과 붙여 쓴다(수원시 장안구 → 수원장안구, 부산 서구 → 부산서구,
 * 전남광주 서구 → 광주서구), 세종특별자치시 → 세종시.
 */
export function hiraGugunNames({ sido, gugun }) {
  const g = String(gugun ?? "");
  const parts = String(sido ?? "").match(/.{1,2}/g) ?? [];
  return [...new Set([g, g.replace(/시 /, ""), g.replace(/\s/g, ""), g.replace(/특별자치시$/, "시"), ...parts.slice(0, 2).map((p) => `${p}${g}`)])];
}

// 유형 4개와 대상 인구(ingu.js TARGET_AGE_LABEL). 치과·한의원·병원급은 넣지 않는다(README).
export const HEALTH_TYPES = [
  { type: "보건기관", target: "all" },
  { type: "의원", target: "all" },
  { type: "약국", target: "all" },
  { type: "산부인과", target: "youthF" }
];
const PUBLIC_HEALTH = new Set(["보건소", "보건지소", "보건진료소"]);
const ID = "암호화요양기호";

/** 시군구 행 판정: 시군구코드명이 지역 시군구 이름 후보(hiraGugunNames) 가운데 하나, 시도코드명이 지역 시도와 맞음. HIRA 시군구 코드는 행정 코드가 아니라 이름으로 거른다. */
export const inRegion = (region) => {
  const names = new Set(hiraGugunNames(region));
  return (r) => names.has(r.시군구코드명) && sidoMatches(r.시도코드명, region.sido);
};

/**
 * 지역 행(병의원·약국·진료과목) → 화면 시설 { type, group: "health", target, name: 유형+번호, lat, lng }. 좌표(X=경도·Y=위도, WGS84) 없는 행은 뺀다.
 * 보건기관 = 종별 보건소·보건지소·보건진료소, 의원 = 종별 의원, 약국 = 약국 파일, 산부인과 = 진료과목에 산부인과가 있는 병의원(다른 유형과 겹칠 수 있음).
 */
export function healthFacilities({ hospitals = [], pharmacies = [], departments = [] }) {
  const obgyn = new Set(departments.filter((d) => d.진료과목코드명 === "산부인과").map((d) => d[ID]));
  const byType = {
    보건기관: hospitals.filter((r) => PUBLIC_HEALTH.has(r.종별코드명)),
    의원: hospitals.filter((r) => r.종별코드명 === "의원"),
    약국: pharmacies,
    산부인과: hospitals.filter((r) => obgyn.has(r[ID]))
  };
  return HEALTH_TYPES.flatMap(({ type, target }) => {
    let n = 0;
    return byType[type].flatMap((r) => {
      const lng = Number(r["좌표(X)"]);
      const lat = Number(r["좌표(Y)"]);
      if (!r["좌표(X)"] || !r["좌표(Y)"] || !Number.isFinite(lat) || !Number.isFinite(lng)) return [];
      n += 1;
      return [{ type, group: "health", target, name: `${type} ${String(n).padStart(2, "0")}`, lat, lng }];
    });
  });
}
