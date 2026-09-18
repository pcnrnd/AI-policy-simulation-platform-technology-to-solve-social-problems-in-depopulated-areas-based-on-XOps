// 데모 표시 OFF 시드 차단 회귀 검사 (vitest 미도입 → node 직접 실행).
//
// 화면 로직은 JSX 안에 흩어져 있어 순수 함수로 잘라내기 어렵다. 대신 깨지면 곧바로 시드가
// 노출되는 두 가지 불변식을 본다.
//   1. 시드 표식 — mock_data.json 의 is_seed 가 실제 시드/실데이터 구분과 일치하는가.
//   2. 호출 계약 — 시드를 폴백으로 내려줄 수 있는 GET 을 include_seed 없이 부르는 곳이 없는가.
//      (백엔드 기본값은 false 이므로 데모 ON 화면이 true 를 명시해야 값이 보인다.)
import assert from "node:assert/strict";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const srcRoot = join(here, "..");

let passed = 0;
const check = (name, fn) => {
  fn();
  passed += 1;
  console.log(`  ok  ${name}`);
};

// ── 1. 시드 표식 ─────────────────────────────────────────────
const seedFile = JSON.parse(readFileSync(join(srcRoot, "assets", "mock_data.json"), "utf-8"));
const schemas = seedFile.metadata_schemas ?? [];

check("모든 카탈로그 소스가 is_seed 를 명시한다", () => {
  for (const s of schemas) {
    assert.equal(typeof s.is_seed, "boolean", `${s.id} 에 is_seed 가 없다`);
  }
});

check("ds_01~07 은 시드, ds_08~12 는 실데이터다", () => {
  const seedIds = schemas.filter((s) => s.is_seed).map((s) => s.id);
  const realIds = schemas.filter((s) => !s.is_seed).map((s) => s.id);
  assert.deepEqual(seedIds, [
    "ds_01_resident_registry",
    "ds_02_local_welfare",
    "ds_03_industrial_factories",
    "ds_04_spatial_geojson",
    "ds_05_smartfarm",
    "ds_06_settlement_facility",
    "ds_07_civil_complaints"
  ]);
  assert.deepEqual(realIds, [
    "ds_08_admin_boundary",
    "ds_09_welfare_facility",
    "ds_10_bccard_dong_industry_sales",
    "ds_11_kt_namwon_monthly_dong_visitors",
    "ds_12_kt_namwon_visitors_by_sex_age"
  ]);
});

check("실데이터 소스는 '실데이터' 태그를 단다", () => {
  for (const s of schemas.filter((x) => !x.is_seed)) {
    assert.ok(s.tags?.includes("실데이터"), `${s.id} 태그에 '실데이터' 가 없다`);
  }
});

// ── 2. 호출 계약 ─────────────────────────────────────────────
// 실측이 없으면 시드를 내려줄 수 있는 GET 경로. 호출 표현식에 include_seed 가 있어야 한다.
const SEED_BEARING = [
  "/api/v3/dataops/catalog",
  "/api/v3/overview/summary",
  "/api/v3/orchestration/models",
  "/api/v3/orchestration/pipelines",
  "/api/v3/orchestration/runs",
  "/api/v3/monitoring/metrics",
  "/api/v3/monitoring/drift",
  "/api/v3/monitoring/explain"
];

// XopsFlowRibbon 은 화면에서 제거된 컴포넌트다(다시 연결하지 않는다) — 검사 대상에서 뺀다.
const SKIP_FILES = ["XopsFlowRibbon.jsx"];

function* sourceFiles(dir) {
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) {
      yield* sourceFiles(full);
    } else if (/\.(jsx?|mjs)$/.test(entry) && !entry.endsWith(".test.mjs") && !SKIP_FILES.includes(entry)) {
      yield full;
    }
  }
}

/** `apiGet(` 호출 하나의 인자 텍스트를 괄호 균형으로 잘라 낸다(여러 줄 호출 대응). */
function apiGetCalls(text) {
  const calls = [];
  const needle = "apiGet(";
  let at = text.indexOf(needle);
  while (at !== -1) {
    let depth = 0;
    let end = at + needle.length - 1;
    for (; end < text.length; end += 1) {
      if (text[end] === "(") depth += 1;
      else if (text[end] === ")") {
        depth -= 1;
        if (depth === 0) break;
      }
    }
    calls.push({ index: at, text: text.slice(at, end + 1) });
    at = text.indexOf(needle, end + 1);
  }
  return calls;
}

check("시드 폴백 GET 은 include_seed 를 함께 보낸다", () => {
  const offenders = [];
  for (const file of sourceFiles(srcRoot)) {
    const text = readFileSync(file, "utf-8");
    for (const call of apiGetCalls(text)) {
      // 단건 조회(`.../{id}` · `.../logs`)는 목록 필터와 무관하다 — 경로 뒤가 바로 닫히는 호출만 본다.
      const hit = SEED_BEARING.find((p) => new RegExp(`${p}(\\?|\\$|["'\`])`).test(call.text));
      if (!hit || call.text.includes("include_seed")) continue;
      const lineNo = text.slice(0, call.index).split("\n").length;
      offenders.push(`${file.slice(srcRoot.length + 1)}:${lineNo} → ${hit}`);
    }
  }
  assert.deepEqual(offenders, [], `include_seed 없이 부르는 곳:\n${offenders.join("\n")}`);
});

console.log(`\n${passed} checks passed`);
