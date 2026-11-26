// flows DB ingu_ 표 → 화면 자료 변환. DB 접속 없이 시험할 수 있도록 순수 함수만 둔다.

const EARTH_R = 6378137;
const RAD = 180 / Math.PI;

/** 웹 메르카토르(EPSG:3857) x·y(m) → 위경도. 숫자가 아니거나 0이면 null. */
export function mercatorToLatLng(x, y) {
  const mx = Number(x);
  const my = Number(y);
  if (!Number.isFinite(mx) || !Number.isFinite(my) || (mx === 0 && my === 0)) return null;
  return { lng: (mx / EARTH_R) * RAD, lat: (2 * Math.atan(Math.exp(my / EARTH_R)) - Math.PI / 2) * RAD };
}

// 시도 코드가 바뀐 곳(전북 45→52, 강원 42→51). 시설 표는 옛 코드, 인구 표는 새 코드를 쓴다. 나머지 지역은 같다.
const SIDO_NEW_TO_OLD = { 52: "45", 51: "42" };
const SIDO_OLD_TO_NEW = Object.fromEntries(Object.entries(SIDO_NEW_TO_OLD).map(([n, o]) => [o, n]));

/** 새 시군구 코드(5자리) → 시설 표 옛 코드. */
export function oldRegionCode(region) {
  const code = String(region);
  return (SIDO_NEW_TO_OLD[code.slice(0, 2)] ?? code.slice(0, 2)) + code.slice(2);
}

/** 주소 코드(10자리, 옛·새 코드 모두) → 인구 표 기준 읍면동 코드(8자리). */
export function dongCode(addr10) {
  const code = String(addr10 ?? "");
  return ((SIDO_OLD_TO_NEW[code.slice(0, 2)] ?? code.slice(0, 2)) + code.slice(2)).slice(0, 8);
}

/**
 * 인구 증감 9구간(합 = 전체 인구). 구간 인구는 API가 행정동 인구 파일(population.js POP_BANDS)에서 같은 id로 만든다.
 * 조건 카드는 ◉ 시설과 관계없이 9줄을 늘 보이고, 대상별 구간(AGE_BANDS)은 그 가운데 일부다.
 */
export const ALL_BANDS = [
  { id: "0-9", label: "0~9세" }, { id: "10-19", label: "10~19세" }, { id: "20-29", label: "20~29세" },
  { id: "30-39", label: "30~39세" }, { id: "40-49", label: "40~49세" }, { id: "50-64", label: "50~64세" },
  { id: "65-74", label: "65~74세" }, { id: "75-84", label: "75~84세" }, { id: "85+", label: "85세 이상" }
];
const pickBands = (...ids) => ALL_BANDS.filter((b) => ids.includes(b.id));
/** 대상별 연령 구간: 노인 65세 이상, 청년(youth) 20~49세, 아동 0~9세, 전체 인구 9구간 모두. */
export const AGE_BANDS = {
  elderly: pickBands("65-74", "75-84", "85+"),
  child: pickBands("0-9"),
  youth: pickBands("20-29", "30-39", "40-49"),
  all: ALL_BANDS
};

/**
 * 대상의 증감 구간(AGE_BANDS). youthF(20~49세 여성)는 청년 3구간을 여성 인구로 센다.
 * 증감값은 구간 id로 기억해 같은 구간을 쓰는 유형끼리 공유한다(전체 인구의 20~29세와 여성 20~29세도 같은 값).
 */
export const targetBands = (target) => AGE_BANDS[target === "youthF" ? "youth" : target] ?? [{ id: "all", label: "전체 인구", from: 0 }];

// 대상 연령층 이름(유형 대상 정의 target → 자료 연령 구간). 노인 돌봄 시설은 모두 elderly.
export const TARGET_AGE_LABEL = { all: "전체 인구", elderly: "65세 이상", child: "0~9세", youth: "20~49세", youthF: "20~49세 여성" };
// 유형 공통 기준 거리 기본값(km). 화면에서 바꾸는 시작값이며 법정·행정 기준이 아니다.
const TYPE_RADIUS_KM = 3.0;

// 시설 유형 이름(공백·가운뎃점 제거) 키워드 → 그룹. 위에서부터 처음 맞는 그룹, 어디에도 안 맞으면 "그 밖".
export const GROUP_RULES = [
  { id: "visit", label: "방문 서비스", color: "#b45309", keywords: ["방문"] },
  { id: "dayCare", label: "주·야간·단기 보호", color: "#7c3aed", keywords: ["주야간", "주간보호", "단기보호"] },
  { id: "residential", label: "입소 시설", color: "#0f766e", keywords: ["요양시설", "공동생활", "양로", "입소"] },
  { id: "other", label: "그 밖", color: "#1d4ed8", keywords: [] },
  // 의료·보건(건강보험심사평가원 병의원·약국 파일, hira.js)은 시설에 그룹이 붙어 와서 키워드를 쓰지 않는다.
  { id: "health", label: "의료·보건", color: "#be185d", keywords: [] },
  // 청년 지원 시설(예시 시설, scenario.js exampleInflowFacilities)도 시설에 그룹이 붙어 온다.
  { id: "youth", label: "청년 지원 시설", color: "#0891b2", keywords: [] }
];

/** 시설 유형 이름 → 그룹 id(키워드 그룹, 안 맞으면 "그 밖"). */
export function groupOf(typeName) {
  const key = String(typeName).replace(/[\s·ㆍ.]/g, "");
  return (GROUP_RULES.find((g) => g.keywords.some((k) => key.includes(k))) ?? GROUP_RULES.find((g) => g.id === "other")).id;
}

/**
 * DB·파일 시설 → 화면 구성 { groups(시설 있는 그룹만), types(시설 많은 순), facilities }.
 * 시설에 group·target이 있으면(의료·보건·청년 지원 시설) 그대로 쓰고, 없으면 유형 이름 키워드 그룹·65세 이상(노인 돌봄).
 * 유형 id = 시설의 typeId(청년 지원 시설 = 유입 시설 종류 id) 또는 "t:{유형 이름}", 시설 id = "f:{유형 id}:{위도},{경도}"(같은 자리 여럿이면 #n).
 * 지역을 바꿔도 같은 유형·시설은 같은 id라서 기준 거리·◉·추가·전환 동작이 그대로 이어진다.
 */
export function dbView(dbFacilities) {
  const counts = {};
  const meta = {};
  for (const f of dbFacilities) {
    counts[f.type] = (counts[f.type] ?? 0) + 1;
    meta[f.type] ??= { id: f.typeId ?? `t:${f.type}`, groupId: f.group ?? groupOf(f.type), target: f.target ?? "elderly" };
  }
  const ranked = Object.entries(counts)
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0], "ko"))
    .map(([label, count]) => ({ ...meta[label], label, radiusKm: TYPE_RADIUS_KM, count }));
  const groups = GROUP_RULES
    .map(({ keywords, ...g }) => ({ ...g, types: ranked.filter((t) => t.groupId === g.id).map((t) => ({ ...t, color: g.color })) }))
    .filter((g) => g.types.length);
  const types = groups.flatMap((g) => g.types);
  const idOf = Object.fromEntries(types.map((t) => [t.label, t.id]));
  const seen = {};
  const facilities = dbFacilities.map((f) => {
    const key = `f:${idOf[f.type]}:${f.lat.toFixed(6)},${f.lng.toFixed(6)}`;
    seen[key] = (seen[key] ?? 0) + 1;
    return { id: seen[key] > 1 ? `${key}#${seen[key]}` : key, typeId: idOf[f.type], name: f.name, lat: f.lat, lng: f.lng, dong: f.dong, added: false };
  });
  return { groups, types, facilities };
}

/** 시설 행(fclty_ty·map_x·map_y·addr10) → 화면 시설. 이름은 유형+번호(기관명은 읽지 않음). 좌표 없는 행은 뺀다. */
export function toFacilities(rows) {
  const counts = {};
  return rows.flatMap((row) => {
    const pos = mercatorToLatLng(row.map_x, row.map_y);
    if (!pos) return [];
    const type = String(row.fclty_ty).trim();
    counts[type] = (counts[type] ?? 0) + 1;
    return [{ type, name: `${type} ${String(counts[type]).padStart(2, "0")}`, ...pos, dong: dongCode(row.addr10) }];
  });
}
