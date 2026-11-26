// 행정안전부 「지역별(행정동) 성별 연령별 주민등록 인구수」 CSV(공공데이터포털 15097972) → 행정동별 인구·연령 구간.
// 열: 행정기관코드(10자리), 기준연월, 시도명, 시군구명, 읍면동명, 계, 남자, 여자, {n}세남자 … 110세이상 남자, {n}세여자 … 110세이상 여자.
// DB·파일 접근 없이 시험할 수 있도록 순수 함수만 둔다.

/** 9구간(합 = 전체). 85세 이상은 끝이 없다. */
export const POP_BANDS = [
  { id: "0-9", from: 0, to: 9 }, { id: "10-19", from: 10, to: 19 }, { id: "20-29", from: 20, to: 29 },
  { id: "30-39", from: 30, to: 39 }, { id: "40-49", from: 40, to: 49 }, { id: "50-64", from: 50, to: 64 },
  { id: "65-74", from: 65, to: 74 }, { id: "75-84", from: 75, to: 84 }, { id: "85+", from: 85, to: Infinity }
];
// 여성 구간(산부인과 대상)
export const FEMALE_BANDS = ["20-29", "30-39", "40-49"];

/** 머리줄 → 나이 열 [{ index, age, female }]. "{n}세남자"·"110세이상 남자" 형식만 받는다. */
export function ageColumns(header) {
  return header.flatMap((name, index) => {
    const m = /^(\d+)세(이상)?\s*(남자|여자)$/.exec(String(name).trim());
    return m ? [{ index, age: Number(m[1]), female: m[3] === "여자" }] : [];
  });
}

const num = (v) => Number(String(v ?? "").replace(/,/g, "")) || 0;
const inBand = (age, b) => age >= b.from && age <= b.to;

/**
 * CSV 행(배열, 머리줄 제외) → { month, dongs: { 행정기관코드: { code, region(5자리), sido, sgg, name, total, child09, bands, femaleBands } } }.
 * bands = 9구간(합 = total). total = "계" 열.
 */
export function aggregateRows(header, rows) {
  const col = (name) => header.indexOf(name);
  const [iCode, iMonth, iSido, iSgg, iName, iTotal] = ["행정기관코드", "기준연월", "시도명", "시군구명", "읍면동명", "계"].map(col);
  if ([iCode, iSido, iSgg, iName, iTotal].some((i) => i < 0)) throw new Error("인구 파일 열을 찾지 못했습니다");
  const ages = ageColumns(header);
  if (!ages.length) throw new Error("인구 파일 나이 열을 찾지 못했습니다");
  const dongs = {};
  let month = null;
  for (const r of rows) {
    const code = String(r[iCode] ?? "").trim();
    if (!/^\d{10}$/.test(code)) continue;
    month ??= String(r[iMonth] ?? "").trim() || null;
    const bands = Object.fromEntries(POP_BANDS.map((b) => [b.id, 0]));
    const femaleBands = Object.fromEntries(FEMALE_BANDS.map((id) => [id, 0]));
    for (const { index, age, female } of ages) {
      const n = num(r[index]);
      for (const b of POP_BANDS) if (inBand(age, b)) bands[b.id] += n;
      if (female) for (const id of FEMALE_BANDS) if (inBand(age, POP_BANDS.find((b) => b.id === id))) femaleBands[id] += n;
    }
    dongs[code] = {
      code, region: code.slice(0, 5), sido: String(r[iSido]).trim(), sgg: String(r[iSgg]).trim(), name: String(r[iName]).trim(),
      total: num(r[iTotal]), child09: bands["0-9"], bands, femaleBands
    };
  }
  return { month, dongs };
}

/** 시군구 목록 [{ code(5자리), sido, name }] — 코드 순. 시군구명이 빈 곳(세종 등)은 시도명을 이름으로. */
export function regionList(dongs) {
  const map = new Map();
  for (const d of Object.values(dongs)) if (!map.has(d.region)) map.set(d.region, { code: d.region, sido: d.sido, name: d.sgg || d.sido });
  return [...map.values()].sort((a, b) => a.code.localeCompare(b.code));
}

const addInto = (to, from) => { for (const [k, v] of Object.entries(from)) to[k] = (to[k] ?? 0) + v; };

/**
 * 고른 지역(5자리 코드 목록)의 행정동만 모아 화면 응답 꼴로: { month, total, child09, bands, femaleBands, dongs: { code: { total, child09, bands, femaleBands, lat?, lng? } } }, names.
 * anchors = { code: { lat, lng } }(경계 안 중심점)가 있으면 그 행정동에 붙인다.
 */
export function populationFor({ month, dongs }, regions, anchors = {}) {
  const set = new Set(regions.map(String));
  const out = { month, total: 0, child09: 0, bands: {}, femaleBands: {}, dongs: {} };
  const names = {};
  for (const d of Object.values(dongs)) {
    if (!set.has(d.region)) continue;
    const { total, child09, bands, femaleBands } = d;
    out.dongs[d.code] = { total, child09, bands, femaleBands, ...(anchors[d.code] ? { lat: anchors[d.code].lat, lng: anchors[d.code].lng } : {}) };
    names[d.code] = d.name;
    out.total += total; out.child09 += child09;
    addInto(out.bands, bands);
    addInto(out.femaleBands, femaleBands);
  }
  return { population: Object.keys(out.dongs).length ? out : null, names };
}

/**
 * 경계가 없는 인구 행(출장소 등)을 같은 시군구에서 이름이 그 행 이름의 앞부분인 경계 하나에 합친다(돌산읍우두출장소 → 돌산읍).
 * → { dongs(경계 코드 기준, 합친 인구 포함), merged: [[인구 코드, 경계 코드]], unmatched: [인구 코드] }. 맞는 경계가 없거나 여럿이면 그대로 둔다.
 */
export function mergeOrphans(dongs, boundaries) {
  const codes = new Set(boundaries.map((b) => b.code));
  const out = Object.fromEntries(Object.entries(dongs).map(([c, d]) => [c, { ...d, bands: { ...d.bands }, femaleBands: { ...d.femaleBands } }]));
  const merged = [];
  const unmatched = [];
  for (const [code, d] of Object.entries(dongs)) {
    if (codes.has(code)) continue;
    const parents = boundaries.filter((b) => b.code.startsWith(d.region) && d.name.startsWith(b.name) && b.name.length < d.name.length);
    const target = parents.length === 1 ? out[parents[0].code] : null;
    if (!target) { unmatched.push(code); continue; }
    for (const k of ["total", "child09"]) target[k] += d[k];
    addInto(target.bands, d.bands);
    addInto(target.femaleBands, d.femaleBands);
    delete out[code];
    merged.push([code, parents[0].code]);
  }
  return { dongs: out, merged, unmatched };
}
