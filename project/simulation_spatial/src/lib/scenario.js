// 시설 추가·전환 시나리오 — 동작 적용, 청년 유입(정주·생활) 배분, 시설 사용률. DB·화면 없이 시험할 수 있도록 순수 함수만 둔다.
// 식과 가정값은 README "계산 규칙"·"가정"에 같은 내용으로 적는다.
import { AGE_BANDS } from "./ingu.js";
import { accessSummary, assignDongs, distanceKm, targetPop } from "./spatial.js";

const YOUTH = AGE_BANDS.youth.map((b) => b.id);

/**
 * 유입 시설 종류(일반 이름, 실제 기관·장소 아님)와 기본 가정값: 1곳당 정주인구(명)·생활인구(명), 청년 3구간 비중.
 * 고정 가정값이다(화면에서 바꾸지 않음). 근거는 README "가정".
 */
export const INFLOW_KINDS = [
  { id: "housing", label: "청년 주택", settled: 100, living: 20, mix: { "20-29": 50, "30-39": 40, "40-49": 10 } },
  { id: "jobs", label: "일자리 거점", settled: 30, living: 300, mix: { "20-29": 30, "30-39": 40, "40-49": 30 } },
  { id: "school", label: "교육·연수 시설", settled: 20, living: 200, mix: { "20-29": 70, "30-39": 20, "40-49": 10 } },
  { id: "startup", label: "창업 공간", settled: 10, living: 100, mix: { "20-29": 40, "30-39": 40, "40-49": 20 } },
  { id: "culture", label: "문화·체육 시설", settled: 0, living: 500, mix: { "20-29": 40, "30-39": 30, "40-49": 30 } }
];
/** 종류 id → { settled, living, mix } — 계산(scenarioState)에 넘기는 고정 가정값. */
export const INFLOW_SETTINGS = Object.fromEntries(INFLOW_KINDS.map(({ id, settled, living, mix }) => [id, { settled, living, mix }]));
export const isInflowKind = (id) => INFLOW_KINDS.some((k) => k.id === id);

/**
 * 동작 목록 → { facilities, placements }.
 * 동작: 추가 { id, kind: "add", to, lat, lng, name } · 전환 { id, kind: "convert", from: 기존 시설 id, to, name }. to = 유형 id(유입 시설 종류 id 포함).
 * facilities = 기존 − 전환된 시설 + 추가·전환된 시설(added, actionId). placements = 그 가운데 유입 시설 종류(kindId) — 유입은 이것만 만든다.
 * 기존(예시 포함) 유입 시설은 지금 있는 시설로 보고 유입을 더하지 않는다. 읍면동은 assignDongs(가장 가까운 대표 지점).
 */
export function applyActions(baseFacilities, actions, units) {
  const byId = Object.fromEntries(baseFacilities.map((f) => [f.id, f]));
  const converted = new Set(actions.filter((a) => a.kind === "convert").map((a) => a.from));
  const made = actions.flatMap((a) => {
    const at = a.kind === "convert" ? byId[a.from] : a;
    if (!at) return [];
    // 전환은 원래 시설의 읍면동(주소 코드)을 그대로 쓴다 — 가장 가까운 대표 지점으로 다시 정하면 소재 읍면동이 바뀔 수 있다
    return [{ id: `act-${a.id}`, actionId: a.id, typeId: a.to, name: a.name, lat: at.lat, lng: at.lng, ...(a.kind === "convert" ? { dong: at.dong } : {}), added: true, converted: a.kind === "convert" }];
  });
  const facilities = assignDongs([...baseFacilities.filter((f) => !converted.has(f.id)), ...made], units);
  return { facilities, placements: facilities.filter((f) => f.actionId && isInflowKind(f.typeId)).map((f) => ({ ...f, kindId: f.typeId })) };
}

/**
 * 고른 지역 안 동작만: 추가는 자리가 고른 지역 경계 안(inside), 전환은 원래 시설이 지금 기존 시설 목록(고른 지역)에 있을 때.
 * 지역을 바꿔도 동작 목록은 그대로 두고, 계산·목록에는 이것만 쓴다.
 */
export function actionsWithin(actions, baseFacilities, inside) {
  const ids = new Set(baseFacilities.map((f) => f.id));
  return actions.filter((a) => (a.kind === "convert" ? ids.has(a.from) : inside(a)));
}

// 예시 유입 시설(실제 자료 아님, README "예시 자료"): 지역 코드로 고정한 시드, 종류별 수, 대표 지점에서 0.3~0.8km
export const EXAMPLE_COUNTS = { housing: 4, jobs: 3, school: 2, startup: 3, culture: 4 };
const EXAMPLE_KM = [0.3, 0.8];
const youthPop = (u) => AGE_BANDS.youth.reduce((s, b) => s + (u.bands?.[b.id] ?? 0), 0);
/** 문자열 → 32비트 시드(FNV-1a). */
const seedOf = (text) => [...String(text)].reduce((h, c) => Math.imul(h ^ c.charCodeAt(0), 16777619) >>> 0, 2166136261);
/** 시드 고정 난수 0~1(mulberry32). */
function seeded(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * 예시 유입 시설: 종류마다 EXAMPLE_COUNTS곳, 대표 지점이 있고 청년(20~49세)이 사는 읍면동을 청년 인구에 비례해 뽑아
 * 그 대표 지점에서 0.3~0.8km 떨어진 임의 방향 자리에 둔다. 시드 = "youth-{지역 코드}"라 지역 코드가 같으면 늘 같은 자리.
 * → DB 시설과 같은 꼴 { type, typeId, group: "youth", target: "youth", name: 유형+번호, lat, lng, dong }.
 */
export function exampleInflowFacilities(units, region) {
  const pool = units.filter((u) => u.lat != null && youthPop(u) > 0);
  if (!pool.length || !region) return [];
  const rand = seeded(seedOf(`youth-${region}`));
  const total = pool.reduce((s, u) => s + youthPop(u), 0);
  return INFLOW_KINDS.flatMap((k) => Array.from({ length: EXAMPLE_COUNTS[k.id] }, (_, i) => {
    let x = rand() * total;
    const u = pool.find((p) => (x -= youthPop(p)) < 0) ?? pool[pool.length - 1];
    const km = EXAMPLE_KM[0] + (EXAMPLE_KM[1] - EXAMPLE_KM[0]) * rand();
    const angle = 2 * Math.PI * rand();
    const lat = u.lat + (km * Math.cos(angle)) / 110.57;
    const lng = u.lng + (km * Math.sin(angle)) / (111.32 * Math.cos((u.lat * Math.PI) / 180));
    return { type: k.label, typeId: k.id, group: "youth", target: "youth", name: `${k.label} ${String(i + 1).padStart(2, "0")}`, lat, lng, dong: u.id };
  }));
}

/** 청년 3구간 비중 → 합 1인 비율(모두 0이면 똑같이). */
function shares(mix = {}) {
  const sum = YOUTH.reduce((s, b) => s + Math.max(0, mix[b] ?? 0), 0);
  return Object.fromEntries(YOUTH.map((b) => [b, sum > 0 ? Math.max(0, mix[b] ?? 0) / sum : 1 / YOUTH.length]));
}

/** 정주인구 유입: 읍면동마다 Σ 배치 × 1곳당 정주인구 × 구간 비중 → { 읍면동: { 구간: 명 } }. 0명은 넣지 않는다. */
export function settledByDong(placements, settings) {
  const out = {};
  for (const p of placements) {
    const s = settings[p.kindId];
    if (!s || !(s.settled > 0) || !p.dong) continue;
    const d = (out[p.dong] ??= {});
    for (const [b, r] of Object.entries(shares(s.mix))) d[b] = (d[b] ?? 0) + s.settled * r;
  }
  return out;
}

/** 생활인구 유입: 읍면동마다 Σ 배치 × 1곳당 생활인구 → { 읍면동: 명 }. 대상 인구에는 더하지 않는다. */
export function livingByDong(placements, settings) {
  const out = {};
  for (const p of placements) {
    const n = settings[p.kindId]?.living ?? 0;
    if (n > 0 && p.dong) out[p.dong] = (out[p.dong] ?? 0) + n;
  }
  return out;
}

/** 계산 단위에 정주 유입(inflow: { 구간: 명 })을 붙인다. 유입이 없으면 같은 배열. 대상 인구(targetPop)가 대상 구간 인원을 더한다. */
export function withInflow(units, byDong) {
  if (!Object.keys(byDong).length) return units;
  return units.map((u) => (byDong[u.id] ? { ...u, inflow: byDong[u.id] } : u));
}

/** 시나리오 상태: 동작을 적용한 시설·배치, 정주 유입을 붙인 계산 단위, 읍면동별 정주·생활 유입 합. */
export function scenarioState(baseFacilities, actions, units, settings) {
  const { facilities, placements } = applyActions(baseFacilities, actions, units);
  const settled = settledByDong(placements, settings);
  const living = livingByDong(placements, settings);
  const settledSum = Object.fromEntries(Object.entries(settled).map(([d, b]) => [d, Object.values(b).reduce((s, n) => s + n, 0)]));
  const total = (o) => Object.values(o).reduce((s, n) => s + n, 0);
  return { facilities, placements, units: withInflow(units, settled), settled: settledSum, living, settledTotal: total(settledSum), livingTotal: total(living) };
}

/**
 * 인구 증감으로 늘거나 준 주민 수: 읍면동마다 Σ 9구간 인구 × 그 구간 증감률/100 → { 읍면동: 명 }(0명은 넣지 않음).
 * 분석 기준과 관계없이 모든 읍면동·모든 구간이다. 정주인구 변화 = 이 값 + 청년 지원 시설 정주 유입(scenarioState settled).
 */
export function growthByDong(units, growth = {}) {
  const out = {};
  for (const u of units) {
    const n = Object.entries(u.bands ?? {}).reduce((s, [b, pop]) => s + (pop * (growth[b] ?? 0)) / 100, 0);
    if (n) out[u.id] = n;
  }
  return out;
}

/** 읍면동별 정주인구 변화(증감분 + 정주 유입)와 합 → { byDong: { 읍면동: 명 }, total }. */
export function settledChange(growthDong, settled) {
  const byDong = { ...growthDong };
  for (const [d, n] of Object.entries(settled)) byDong[d] = (byDong[d] ?? 0) + n;
  return { byDong, total: Object.values(byDong).reduce((s, n) => s + n, 0) };
}

/** 분석 기준 시설의 평균 사용률(사용률이 있는 시설 평균, 없으면 null). 처음 상태는 수용을 그렇게 잡았으므로 1(100%). */
export function meanRate(u) {
  const rates = Object.values(u ?? {}).map((v) => v.rate).filter((r) => r != null);
  return rates.length ? rates.reduce((s, r) => s + r, 0) / rates.length : null;
}

/** "변화" 띠를 보일지: 처음 상태(추가·전환 0, 증감 0)와 다를 때만. 입력은 인구 증감과 시설 추가·전환뿐이다. */
export const changeVisible = (actions, growth) => actions.length > 0 || Object.values(growth).some((v) => v !== 0);

/** 1곳당 수용 = 처음 상태(동작·증감·유입 0) 대상 인구 합(대표 지점 있는 읍면동) ÷ 처음 그 유형 시설 수. 시설이 없으면 null. */
export function capacityOf(units, typeFacilities, target) {
  if (!typeFacilities.length) return null;
  const total = units.reduce((s, u) => (u.lat == null ? s : s + targetPop(u, target, 0)), 0);
  return total > 0 ? total / typeFacilities.length : null;
}

// 같은 거리로 보는 차이(km) — 같은 건물·좌표에 시설이 여럿인 경우
const TIE_KM = 1e-6;

/**
 * 시설 사용률: 대표 지점 있는 읍면동을 가장 가까운 그 유형 시설에 배정, 사용률 = 배정 대상 인구 합 ÷ 1곳당 수용.
 * 가장 가까운 시설이 여럿(같은 좌표)이면 수요를 똑같이 나눈다(먼저 나온 한 곳이 다 받고 나머지가 0%가 되지 않게).
 * → { [시설 id]: { demand, rate, dongs: [읍면동 id] } }. 수용이 없으면 rate = null.
 */
export function usage(units, typeFacilities, target, growth, capacity) {
  const out = Object.fromEntries(typeFacilities.map((f) => [f.id, { demand: 0, rate: null, dongs: [] }]));
  if (!typeFacilities.length) return out;
  for (const u of units) {
    if (u.lat == null) continue;
    const dist = typeFacilities.map((f) => distanceKm(u, f));
    const min = Math.min(...dist);
    const near = typeFacilities.filter((_, i) => dist[i] - min <= TIE_KM);
    const share = targetPop(u, target, growth) / near.length;
    for (const f of near) {
      out[f.id].demand += share;
      out[f.id].dongs.push(u.id);
    }
  }
  for (const v of Object.values(out)) v.rate = capacity ? v.demand / capacity : null;
  return out;
}

/** 사용률 3단계(README 가정): 70% 미만 low, 70~130% mid, 130% 초과 high. */
export const USAGE_LEVELS = { low: 0.7, high: 1.3 };
export const usageLevel = (rate) => (rate == null ? null : rate < USAGE_LEVELS.low ? "low" : rate > USAGE_LEVELS.high ? "high" : "mid");

/**
 * 동작 전후 사용률 비교: 화면 값(% 정수)이 바뀌었거나 새로 생기거나 빠진 시설만 → [{ id, before, after }], 바뀐 폭 큰 순.
 * 배정 읍면동만 바뀌고 사용률이 같은 시설은 넣지 않는다.
 */
export function usageChanges(before, after) {
  const ids = [...new Set([...Object.keys(before), ...Object.keys(after)])];
  const pct = (v) => (v?.rate == null ? null : Math.round(v.rate * 100));
  const same = (a, b) => a && b && pct(a) === pct(b);
  return ids
    .filter((id) => !same(before[id], after[id]))
    .map((id) => ({ id, before: before[id]?.rate ?? null, after: after[id]?.rate ?? null }))
    .sort((x, y) => Math.abs((y.after ?? 0) - (y.before ?? 0)) - Math.abs((x.after ?? 0) - (x.before ?? 0)));
}

/** 한 유형의 시나리오 결과: 접근 요약 + 사용률. env = { radiusKm, growthPct, minPop }. 분석 기준이 없으면(type 없음) null — 판정 없음. */
export function typeResult(state, type, env, capacity) {
  if (!type) return null;
  const ids = new Set(basisTypeIds(type));
  const facs = state.facilities.filter((f) => ids.has(f.typeId));
  return { summary: accessSummary(state.units, facs, type, env), usage: usage(state.units, facs, type.target, env.growthPct, capacity) };
}

/*
 * 분석 기준(basis): 유형 하나({ id, label, target, color }) 또는 그룹 전체(groupBasis, members = 공급으로 쓰는 유형 id들).
 * 분석 켜고 끄기는 "분석" 탭에서만 하고, 시설 분류 탭의 지도 표시 체크는 분석을 바꾸지 않는다.
 */

/**
 * 그룹 전체 분석 기준: 그룹 안 유형의 대상이 모두 같으면 그 대상, 섞여 있으면 전체 인구("all")이고
 * 대상이 다른 유형(예: 의료·보건의 산부인과 20~49세 여성)은 공급(members)에서 뺀다. 수용 능력은 members 시설 전체로 잡는다(그룹 평균).
 */
export function groupBasis(group) {
  const targets = new Set(group.types.map((t) => t.target));
  const target = targets.size === 1 ? [...targets][0] : "all";
  return { id: `g:${group.id}`, label: group.label, color: group.color, target, members: group.types.filter((t) => t.target === target).map((t) => t.id) };
}

/** 분석 기준이 공급으로 쓰는 유형 id들(유형 하나면 그 유형, 기준이 없으면 []). */
export const basisTypeIds = (basis) => (basis ? basis.members ?? [basis.id] : []);

/** 분석 탭에서 줄을 눌렀을 때: 같은 줄을 다시 누르면 분석 끔(null), 아니면 그 줄. */
export const toggleBasis = (current, id) => (current === id ? null : id);

/** 지도에 그릴 시설 점: 표시 체크가 켜진 유형 + 분석 기준의 공급 시설(체크와 관계없이 늘). */
export function mapFacilities(facilities, visible, basis) {
  const ids = new Set(basisTypeIds(basis));
  return facilities.filter((f) => visible.has(f.typeId) || ids.has(f.typeId));
}

/** 추가·전환 목록의 기본 유형: 유형 기준이면 그 유형, 그룹 기준이면 공급 유형 가운데 시설이 가장 많은 유형(같으면 앞 유형). */
export function basisDefaultType(basis, facilities) {
  if (!basis?.members) return basis?.id ?? null;
  const count = (id) => facilities.filter((f) => f.typeId === id).length;
  return basis.members.reduce((best, id) => (best == null || count(id) > count(best) ? id : best), null);
}

/** 가장 높은 사용률(시설이 없으면 null). */
export const maxRate = (u, ids = Object.keys(u)) => ids.reduce((m, id) => (u[id]?.rate != null && (m == null || u[id].rate > m) ? u[id].rate : m), null);
