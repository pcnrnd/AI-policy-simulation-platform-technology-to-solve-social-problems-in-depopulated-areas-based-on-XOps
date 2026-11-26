// 행정동 경계(GeoJSON, 좌표 [경도, 위도]) 처리 — 시군구별 묶음, 좌표 → 행정동, 경계 안 중심점, 점이 경계 안인지, 범위.
// API(api.js)와 화면이 같이 쓴다. 지역 고정값을 두지 않는다.

const polygonsOf = (geometry) => (geometry.type === "Polygon" ? [geometry.coordinates] : geometry.type === "MultiPolygon" ? geometry.coordinates : []);

/** 점이 다각형(바깥 고리 + 구멍) 안인지 — 짝홀 규칙이라 구멍 안은 밖이 된다. */
function inPolygon({ lat, lng }, rings) {
  let inside = false;
  for (const ring of rings) {
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const [xi, yi] = ring[i];
      const [xj, yj] = ring[j];
      if ((yi > lat) !== (yj > lat) && lng < ((xj - xi) * (lat - yi)) / (yj - yi) + xi) inside = !inside;
    }
  }
  return inside;
}

/** 점 { lat, lng }이 Polygon·MultiPolygon 안인지. */
export function pointInGeometry(pt, geometry) {
  return polygonsOf(geometry).some((rings) => inPolygon(pt, rings));
}

// 고리 넓이(부호 있음)와 무게중심 — 수 km 범위라 경위도 평면 근사로 충분하다.
function ringCentroid(ring) {
  let a = 0;
  let cx = 0;
  let cy = 0;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [x0, y0] = ring[j];
    const [x1, y1] = ring[i];
    const cross = x0 * y1 - x1 * y0;
    a += cross; cx += (x0 + x1) * cross; cy += (y0 + y1) * cross;
  }
  return a === 0 ? { area: 0, lng: ring[0][0], lat: ring[0][1] } : { area: a / 2, lng: cx / (3 * a), lat: cy / (3 * a) };
}

/** 위도 lat 가로줄이 다각형 안을 지나는 가장 긴 구간의 가운데(없으면 null). */
function scanlineMid(rings, lat) {
  const xs = [];
  for (const ring of rings) {
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const [xi, yi] = ring[i];
      const [xj, yj] = ring[j];
      if ((yi > lat) !== (yj > lat)) xs.push(xi + ((lat - yi) * (xj - xi)) / (yj - yi));
    }
  }
  xs.sort((p, q) => p - q);
  let best = null;
  for (let k = 0; k + 1 < xs.length; k += 2) {
    if (!best || xs[k + 1] - xs[k] > best.w) best = { w: xs[k + 1] - xs[k], lng: (xs[k] + xs[k + 1]) / 2 };
  }
  return best && { lat, lng: best.lng };
}

/**
 * 경계 안 중심점 { lat, lng }: 가장 큰 조각의 무게중심, 그 점이 경계 밖이면(오목한 모양) 같은 위도에서 경계 안을 지나는 가장 긴 구간의 가운데.
 */
export function interiorPoint(geometry) {
  const polys = polygonsOf(geometry);
  if (!polys.length) return null;
  const sized = polys.map((rings) => ({ rings, c: ringCentroid(rings[0]) }));
  const { rings, c } = sized.reduce((m, p) => (Math.abs(p.c.area) > Math.abs(m.c.area) ? p : m));
  const centroid = { lat: c.lat, lng: c.lng };
  if (inPolygon(centroid, rings)) return centroid;
  return scanlineMid(rings, c.lat) ?? { lat: rings[0][0][1], lng: rings[0][0][0] };
}

/** 경계 목록의 범위 { south, west, north, east }. 경계가 없으면 null. */
export function boundaryBounds(boundaries) {
  let b = null;
  for (const { geometry } of boundaries) {
    for (const rings of polygonsOf(geometry)) {
      for (const [lng, lat] of rings[0]) {
        b = b
          ? { south: Math.min(b.south, lat), west: Math.min(b.west, lng), north: Math.max(b.north, lat), east: Math.max(b.east, lng) }
          : { south: lat, west: lng, north: lat, east: lng };
      }
    }
  }
  return b;
}

/** 점이 경계 가운데 하나라도 안인지(경계 상자로 먼저 거른다). 경계가 없으면 모두 안으로 본다. */
export function insideAny(boundaries) {
  if (!boundaries.length) return () => true;
  const find = boundaryOfPoint(boundaries);
  return (pt) => find(pt) !== null;
}

/** 점 → 그 점을 담은 경계 코드(행정동 10자리), 없으면 null. 경계 상자로 먼저 거른다. */
export function boundaryOfPoint(boundaries) {
  const boxes = boundaries.map((b) => ({ code: b.code, g: b.geometry, box: boundaryBounds([b]) }));
  return (pt) => boxes.find(({ g, box }) => pt.lat >= box.south && pt.lat <= box.north && pt.lng >= box.west && pt.lng <= box.east && pointInGeometry(pt, g))?.code ?? null;
}

/** GeoJSON 전체 → 시군구(경계 코드 앞 5자리)별 경계 Map(region → [{ code, name, geometry }]). 시작할 때 한 번 만든다. */
export function boundariesByRegion(geojson) {
  const map = new Map();
  for (const f of geojson?.features ?? []) {
    const code = String(f.properties?.adm_cd2 ?? "");
    if (!/^\d{10}$/.test(code) || !f.geometry) continue;
    const region = code.slice(0, 5);
    if (!map.has(region)) map.set(region, []);
    map.get(region).push({ code, name: String(f.properties.adm_nm ?? "").trim().split(/\s+/).pop(), geometry: f.geometry });
  }
  return map;
}

/** 경계마다 대표 지점(경계 안 중심점) { code: { lat, lng } }. 행정동 단위 인구의 대표 지점이 된다. */
export function boundaryAnchors(boundaries) {
  return Object.fromEntries(boundaries.flatMap((b) => {
    const pt = interiorPoint(b.geometry);
    return pt ? [[b.code, pt]] : [];
  }));
}
