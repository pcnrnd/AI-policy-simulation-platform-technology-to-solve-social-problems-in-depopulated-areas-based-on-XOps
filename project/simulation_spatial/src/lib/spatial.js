// 규칙 기반 공간 계산 — 읍면동 단위 접근 거리·부족 지역·시설 추가 영향, 데이터 범위의 접근 거리 격자.
// 예측·최적화 모델을 쓰지 않는다. 모든 식은 README "계산 규칙"에 같은 내용으로 적는다. 기준 거리 시작값(시·구 3km, 군 5km) 말고는 지역 고정값을 두지 않는다.
import { targetBands } from "./ingu.js";

const KM_PER_DEG_LAT = 110.57;
const KM_PER_DEG_LNG_EQ = 111.32;
const kmPerDegLng = (lat) => KM_PER_DEG_LNG_EQ * Math.cos((lat * Math.PI) / 180);

/** 두 지점 거리(km). 등장방형 근사 — 수십 km 범위에서 오차 0.1% 미만, 광역 범위면 하버사인으로 교체. */
export function distanceKm(a, b) {
  const dx = (b.lng - a.lng) * kmPerDegLng((a.lat + b.lat) / 2);
  const dy = (b.lat - a.lat) * KM_PER_DEG_LAT;
  return Math.hypot(dx, dy);
}

/**
 * 대상 인구 = Σ 대상 연령 구간 인구 × (1 + 그 구간 증감률/100). 구간은 인구 증감 9구간(ingu.js ALL_BANDS) 가운데 대상의 것 —
 * 노인 65~74·75~84·85세 이상, 전체 인구 9구간 모두, 20~49세(여성) 3구간. 20~49세 여성(youthF)은 여성 구간 인구.
 * growth = 구간 id별 증감률 { "65-74": 10, … }(없는 구간은 0), 또는 모든 구간에 같은 값을 주는 숫자 하나. 대상 밖 구간 값은 쓰지 않는다.
 * 단위에 그 대상의 구간 인구가 없으면 인구 × 대상 비율 × (1 + 대상 구간 증감률 평균/100)(20~49세 여성은 0).
 * 여기에 정주인구 유입(unit.inflow = { 구간: 명 }, scenario.js withInflow) 가운데 대상 구간 인원을 더한다(증감률은 곱하지 않음).
 * 20~49세 여성 대상은 유입 인원 × 그 읍면동 같은 구간의 지금 여성 비율만 더한다.
 */
export function targetPop(unit, target, growth = 0) {
  const pct = (id) => (typeof growth === "number" ? growth : growth?.[id] ?? 0);
  const bands = targetBands(target);
  const src = target === "youthF" ? unit.femaleBands : unit.bands;
  const added = (id) => {
    const n = unit.inflow?.[id] ?? 0;
    if (!n || target !== "youthF") return n;
    return unit.bands?.[id] > 0 ? (n * (unit.femaleBands?.[id] ?? 0)) / unit.bands[id] : 0;
  };
  const inflow = bands.reduce((s, b) => s + added(b.id), 0);
  if (src && bands.every((b) => b.id in src)) {
    return Math.max(0, bands.reduce((s, b) => s + src[b.id] * (1 + pct(b.id) / 100), 0)) + inflow;
  }
  const share = { elderly: unit.elderlyShare, child: unit.childShare, all: 1 }[target] ?? 0;
  const meanPct = bands.reduce((s, b) => s + pct(b.id), 0) / bands.length;
  return Math.max(0, unit.pop * share * (1 + meanPct / 100)) + inflow;
}

/** 지점마다 가장 가까운 시설까지 거리(km). 지점 위치가 없거나 시설이 없으면 Infinity. */
export function nearestDistances(points, facilities) {
  return points.map((p) => (p.lat == null ? Infinity : facilities.reduce((m, f) => Math.min(m, distanceKm(p, f)), Infinity)));
}

/**
 * 행정동 인구(API) → 계산 단위. 대표 지점 = API가 붙인 그 행정동 경계 안 중심점(lat·lng). 경계가 없는 행정동은 지점이 없다.
 * population = { dongs: { 행정동 10자리: { total, bands, femaleBands, lat?, lng? } } }, names = { 코드: 이름 }.
 * elderlyShare = 65세 이상 비율(구간 인구가 없을 때 targetPop 대비용).
 */
export function dongUnits(population, names = {}) {
  if (!population?.dongs) return [];
  const OLD = ["65-74", "75-84", "85+"];
  return Object.entries(population.dongs)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([code, d]) => {
      const share = (v) => (d.total > 0 ? v / d.total : 0);
      return {
        id: code, name: names[code] ?? code,
        ...(d.lat != null ? { lat: d.lat, lng: d.lng } : {}),
        pop: d.total,
        elderlyShare: share(d.bands ? OLD.reduce((s, k) => s + (d.bands[k] ?? 0), 0) : d.elderly65 ?? 0),
        childShare: share(d.bands?.["0-9"] ?? d.child09 ?? 0),
        ...(d.bands ? { bands: d.bands } : {}), ...(d.femaleBands ? { femaleBands: d.femaleBands } : {})
      };
    });
}

/** 시설의 읍면동: API가 좌표로 정한 행정동이 계산 단위에 있으면 그대로, 없으면(추가 시설 등) 가장 가까운 대표 지점의 읍면동. */
export function assignDongs(facilities, units) {
  const ids = new Set(units.map((u) => u.id));
  const pts = units.filter((u) => u.lat != null);
  if (!pts.length) return facilities;
  return facilities.map((f) => {
    if (ids.has(f.dong)) return f;
    const near = pts.reduce((best, u) => (distanceKm(u, f) < distanceKm(best, f) ? u : best));
    return { ...f, dong: near.id };
  });
}

// 화면 값 표기 — 부동소수 꼬리 없이(2.9×3 → 8.7), 인구는 천 단위 쉼표
const num = (v) => Math.round(v * 100) / 100;
export const kmText = (v) => `${num(v)}km`;
export const popText = (n) => `${Math.round(n).toLocaleString("ko-KR")}명`;

// 기준 거리 시작값(km): 시·구 3km, 군 5km(README "가정" — 국토연구원 2018). 시군구 이름 끝 글자로 정하고, 그 밖은 시로 본다.
export const CITY_START_KM = 3;
export const COUNTY_START_KM = 5;
export const regionStartKm = (name = "") => (name.trim().endsWith("군") ? COUNTY_START_KM : CITY_START_KM);
// 행정동 10자리 코드의 앞 5자리 = 시군구 코드
export const regionOf = (dongId) => String(dongId ?? "").slice(0, 5);
/**
 * 그 시군구의 기준 거리. env.radiusByRegion = { 시군구 코드: km } — 시·군 시작값이 섞인 채 거리 슬라이더를 아직 움직이지 않았을 때만 있고,
 * 없으면(움직였거나 시작값이 하나) 모든 지역이 env.radiusKm.
 */
export const radiusIn = (env, code) => env.radiusByRegion?.[code] ?? env.radiusKm;
/** 화면에 보일 기준 거리 값들(작은 순, 같은 값은 하나) — 섞이면 [3, 5]. */
export const radiusList = (env) => (env.radiusByRegion ? [...new Set(Object.values(env.radiusByRegion))].sort((a, b) => a - b) : [env.radiusKm]);
// 값 하나 또는 여럿을 "3·5" 꼴로(k배)
const kms = (r, k = 1) => [r].flat().map((v) => num(k * v)).join("·");
export const radiusText = (env) => `${kms(radiusList(env))}km`;

// 거리 구간 — 기준 거리(r) 배수로 나눈다: r 이내, r~2r, 2r~3r, 3r 초과. 영향 그래프와 지도 격자·범례가 같이 쓴다. r이 여럿이면 "3·5km 이내"처럼 함께 적는다.
export const bandLabels = (r) => [`${kms(r)}km 이내`, `${kms(r)}~${kms(r, 2)}km`, `${kms(r, 2)}~${kms(r, 3)}km`, `${kms(r, 3)}km 초과`];
export const bandOf = (d, r) => (d <= r ? 0 : d <= 2 * r ? 1 : d <= 3 * r ? 2 : 3);

/**
 * 한 시설 유형의 읍면동 접근 현황. env = { radiusKm, radiusByRegion?, growthPct, minPop } · type = { target }
 * 부족 읍면동 = 대표 지점에서 최근접 거리 > 그 읍면동 시군구의 기준 거리(radiusIn) 이고 대상 인구 ≥ 최소 인구.
 * 대표 지점이 없는 단위는 거리를 알 수 없어 합계·구간·부족 판정·평균 거리에서 모두 뺀다(dist·pops 배열에는 자리만 남김).
 * radii = 읍면동마다 쓴 기준 거리(상태·상세 문구용).
 */
export function accessSummary(units, facilities, type, env) {
  const dist = nearestDistances(units, facilities);
  const pops = units.map((u) => targetPop(u, type.target, env.growthPct));
  const radii = units.map((u) => radiusIn(env, regionOf(u.id)));
  const bands = [0, 0, 0, 0];
  let total = 0;
  let knownPop = 0;
  let weightedDist = 0;
  const shortage = [];
  units.forEach((u, i) => {
    if (u.lat == null) return;
    total += pops[i];
    bands[bandOf(dist[i], radii[i])] += pops[i];
    if (Number.isFinite(dist[i])) { weightedDist += pops[i] * dist[i]; knownPop += pops[i]; }
    if (dist[i] > radii[i] && pops[i] >= env.minPop) shortage.push(u.id);
  });
  const covered = bands[0];
  return {
    dist, pops, radii, total, covered, bands: bands.map(Math.round), shortage,
    coveredPct: total > 0 ? (100 * covered) / total : 0,
    meanDistKm: knownPop > 0 ? weightedDist / knownPop : null
  };
}

// 읍면동 상태 — 지도 범례와 상세 문구가 같이 쓴다. 판정은 accessSummary의 부족 규칙 그대로.
// r = 값 하나 또는 여럿(범례는 [3, 5] → "3·5km")
export const dongStatusLabels = (r, minPop) => ({ shortage: `부족 지역 · ${kms(r)}km 밖·${popText(minPop)} 이상`, within: `${kms(r)}km 이내`, outsideLow: `${kms(r)}km 밖·${popText(minPop)} 미만` });

/** i번째 읍면동 상태: 그 읍면동 기준 거리 안 / 부족(밖이고 대상 인구 ≥ 최소) / 기준 거리 밖(인구 적음). */
export function dongStatus(summary, i, env) {
  if (summary.dist[i] <= (summary.radii?.[i] ?? env.radiusKm)) return "within";
  return summary.pops[i] >= env.minPop ? "shortage" : "outsideLow";
}

/** 증감률 0일 때 읍면동 대상 인구 최댓값(지도 크기·색 기준). 읍면동이 없으면 0. */
export function basePopMax(units, target) {
  return units.reduce((m, u) => Math.max(m, targetPop(u, target, 0)), 0);
}

// 인구 자료가 없을 때의 최소 대상 인구 슬라이더 범위
export const DEFAULT_MIN_POP_SCALE = { max: 200, step: 5 };

/** 1·2·5×10ⁿ 가운데 x에 가장 가까운 값(로그 거리, 1 이상). */
function niceStep(x) {
  if (!(x > 1)) return 1;
  const p = 10 ** Math.floor(Math.log10(x));
  return [1, 2, 5, 10].map((m) => m * p).reduce((best, c) => (Math.abs(Math.log(c / x)) < Math.abs(Math.log(best / x)) ? c : best));
}

// 부족 판정 기준 기본값(명). 슬라이더 간격이 이 값을 나누어야 기본값이 눈금 위에 놓인다.
export const DEFAULT_MIN_POP = 20;

/**
 * 최소 대상 인구 슬라이더 범위: 최댓값 = 기준 최댓값을 (약 1/100 간격(1·2·5×10ⁿ) × 10) 단위로 올림,
 * 간격 = 약 1/100 간격과 기본값(20명) 가운데 작은 쪽(1·2·5·10·20 — 모두 20을 나눔). 여러 지역처럼 최댓값이 커도 20명이 눈금에 맞는다.
 */
export function minPopScale(baseMax) {
  if (!(baseMax > 0)) return DEFAULT_MIN_POP_SCALE;
  const nice = niceStep(baseMax / 100);
  return { max: Math.ceil(baseMax / (nice * 10)) * nice * 10, step: Math.min(nice, DEFAULT_MIN_POP) };
}

/** 시설 한 곳의 기준 거리(그 시설 읍면동 시군구의 값) 안 대상 인구(대표 지점 기준). */
export function servedPop(units, facility, type, env) {
  const r = radiusIn(env, regionOf(facility.dong));
  return units.reduce((s, u) => (u.lat != null && distanceKm(u, facility) <= r ? s + targetPop(u, type.target, env.growthPct) : s), 0);
}

/** 격자 칸마다 시군구(가장 가까운 읍면동 대표 지점의 시군구) — 시·군 기준 거리가 섞일 때 칸 색을 그 지역 값으로 나누려고 쓴다. */
export function withRegions(cells, units) {
  const pts = units.filter((u) => u.lat != null);
  if (!pts.length) return cells;
  return cells.map((c) => ({ ...c, region: regionOf(pts.reduce((b, u) => (distanceKm(u, c) < distanceKm(b, c) ? u : b)).id) }));
}

/** 점 목록의 범위 { south, west, north, east }. 좌표가 있는 점이 없으면 null. */
export function dataBounds(points) {
  const pts = points.filter((p) => Number.isFinite(p.lat) && Number.isFinite(p.lng));
  if (!pts.length) return null;
  const lats = pts.map((p) => p.lat);
  const lngs = pts.map((p) => p.lng);
  return { south: Math.min(...lats), west: Math.min(...lngs), north: Math.max(...lats), east: Math.max(...lngs) };
}

/** 범위를 사방 km만큼 넓힌다. */
export function padBounds(b, km) {
  const dLat = km / KM_PER_DEG_LAT;
  const dLng = km / kmPerDegLng((b.south + b.north) / 2);
  return { south: b.south - dLat, west: b.west - dLng, north: b.north + dLat, east: b.east + dLng };
}

/**
 * 범위를 덮는 접근 거리 격자. 칸마다 가장 가까운 시설까지 거리(km).
 * 칸 수 상한(maxCells)을 넘으면 칸을 키운다 — 칸 × 시설 전수 계산, 시설이 수천 곳이면 공간 색인으로 교체.
 */
export function accessGrid(bounds, facilities, cellKm = 0.8, maxCells = 3600) {
  const hKm = (bounds.north - bounds.south) * KM_PER_DEG_LAT;
  const wKm = (bounds.east - bounds.west) * kmPerDegLng((bounds.south + bounds.north) / 2);
  const size = Math.max(cellKm, Math.sqrt((hKm * wKm) / maxCells));
  const rows = Math.max(1, Math.ceil(hKm / size));
  const cols = Math.max(1, Math.ceil(wKm / size));
  const dLat = (bounds.north - bounds.south) / rows;
  const dLng = (bounds.east - bounds.west) / cols;
  const cells = [];
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const s = bounds.south + r * dLat;
      const w = bounds.west + c * dLng;
      cells.push({
        id: `G${String(r).padStart(2, "0")}-${String(c).padStart(2, "0")}`,
        lat: s + dLat / 2, lng: w + dLng / 2,
        bounds: [[s, w], [s + dLat, w + dLng]]
      });
    }
  }
  const dist = nearestDistances(cells, facilities);
  return cells.map((cell, i) => ({ ...cell, dist: dist[i] }));
}
