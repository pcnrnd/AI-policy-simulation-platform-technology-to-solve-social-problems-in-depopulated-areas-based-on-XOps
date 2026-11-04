// 시나리오 저장값(localStorage) 형식과 데모 기본 시나리오 규칙 — 화면(ScenarioCompare)과 검사가 함께 쓴다.

export const MAX_PER_REGION = 8;

// 데모 표시 ON 기본 시나리오 — 지역(남원·신안)별 최초 1회만 넣는다. 넣은 지역은 같은 저장값에 표식을 남겨
// 사용자가 지운 뒤(다른 탭 포함)에도 다시 넣지 않는다. 실데이터 저장 키는 건드리지 않는다.
export const PRESET_REGIONS = ["namwon", "shinan"];
export const PRESET_SCENARIOS = [
  { name: "복지·돌봄 강화안", budgetTotal: 600, welfareWeight: 70, industryWeight: 40, housingWeight: 50 },
  { name: "일자리·산업 육성안", budgetTotal: 800, welfareWeight: 40, industryWeight: 75, housingWeight: 45 },
  { name: "주거·정주 여건 개선안", budgetTotal: 500, welfareWeight: 45, industryWeight: 40, housingWeight: 75 }
];

/** 저장값 → { scenarios, presets }. 예전 형식(시나리오 배열)도 읽는다. */
export function parseScenarioStore(raw) {
  try {
    const parsed = raw ? JSON.parse(raw) : [];
    const rows = Array.isArray(parsed) ? parsed : parsed?.scenarios;
    const presets = !Array.isArray(parsed) && parsed?.presets && typeof parsed.presets === "object" ? parsed.presets : {};
    return { scenarios: Array.isArray(rows) ? rows.filter((s) => s && s.id && s.regionId) : [], presets };
  } catch {
    return { scenarios: [], presets: {} };
  }
}

/** 저장할 값 — 표식이 없으면 예전 형식(배열) 그대로라 실데이터 키의 형식은 바뀌지 않는다. */
export function serializeScenarioStore(store) {
  return JSON.stringify(Object.keys(store.presets).length ? store : store.scenarios);
}

/** 기본 시나리오 주입 결과 — 이미 표식이 있거나 대상 지역이 아니면 null. 기존 행은 보존하고 남은 자리만 채운다. */
export function withPresetScenarios(store, region, now = new Date().toISOString()) {
  if (!PRESET_REGIONS.includes(region.id) || store.presets[region.id]) return null;
  const room = Math.max(0, MAX_PER_REGION - store.scenarios.filter((s) => s.regionId === region.id).length);
  const controls = region.case?.simulation?.controls ?? [];
  const controlValues = Object.fromEntries(controls.map((c) => [c.key, c.default]));
  const rows = PRESET_SCENARIOS.slice(0, room).map((preset, index) => ({
    id: `${region.id}-preset-${index + 1}`, regionId: region.id, createdAt: now, ...preset, controlValues
  }));
  return { scenarios: [...store.scenarios, ...rows], presets: { ...store.presets, [region.id]: true } };
}
