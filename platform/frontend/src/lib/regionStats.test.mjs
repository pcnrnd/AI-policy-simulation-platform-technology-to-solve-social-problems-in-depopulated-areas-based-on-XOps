// 남원·신안 지역 통계 출처 검사 (vitest 미도입 → node 직접 실행).
// 공식 통계로 바꾼 값은 원자료 수치와 같고, 값 필드마다 basis(source | demo_example)·기준 시점·출처가 있다.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const seed = JSON.parse(readFileSync(join(here, "..", "assets", "mock_data.json"), "utf-8"));
const regionOf = (id) => seed.regions.find((r) => r.id === id);

// 원자료 수치(2026-11-06 확인):
// - population: 행정안전부 주민등록 인구통계, 2025년 12월 총인구(거주자·거주불명자·재외국민, 외국인 제외)
// - birthRate: KOSIS 인구동향조사 시군구 합계출산율 2025년(확정)
const OFFICIAL = {
  namwon: { population: 74272, birthRate: 0.853 },
  shinan: { population: 41858, birthRate: 0.896 }
};
const PERIODS = { population: "2025-12", birthRate: "2025" };
// 바꾸지 않고 demo_example로 둔 값.
// - riskIndex·agingIndex: 화면 라벨과 정의가 맞는 공식 통계를 찾지 못했다.
// - history: 공식 값(주민등록 2015~2024년 6월)은 확인했지만, 백엔드 데모 학습·요인 패널이 이 시드 이력에 맞춰 고정돼 있어
//   바꾸면 재학습 거부·요인 계수 부호 반전 등 백엔드 시험 12개가 실패한다(PM 결정 대기).
const NOT_OFFICIAL = {
  namwon: { riskIndex: 0.16, agingIndex: 33.4, history: [85200, 83800, 82400, 81100, 80300, 79400, 78600, 78000, 77600, 77200] },
  shinan: { riskIndex: 0.13, agingIndex: 43.1, history: [45600, 44700, 43800, 42600, 41700, 40800, 40000, 39300, 38800, 38400] }
};

for (const [id, values] of Object.entries(OFFICIAL)) {
  const region = regionOf(id);
  assert.ok(region, id);
  for (const [field, value] of Object.entries(values)) {
    assert.deepEqual(region[field], value, `${id}.${field}`);
    assert.equal(region.basis[field], "source", `${id}.${field} basis`);
    assert.equal(region.periods[field], PERIODS[field], `${id}.${field} 기준 시점`);
    const cited = region.sources.filter((s) => s.fields.includes(field));
    assert.equal(cited.length, 1, `${id}.${field} 출처`);
    assert.match(cited[0].url, /^https:\/\/(jumin\.mois\.go\.kr|kosis\.kr)\//, `${id}.${field} 공공기관 원자료`);
    assert.match(cited[0].checked, /^\d{4}-\d{2}-\d{2}$/);
    assert.ok(cited[0].title, `${id}.${field} 출처 제목`);
  }
  for (const [field, value] of Object.entries(NOT_OFFICIAL[id])) {
    assert.deepEqual(region[field], value, `${id}.${field} 원래 값 유지`);
    assert.equal(region.basis[field], "demo_example", `${id}.${field} basis`);
    assert.equal(region.periods[field], undefined, `${id}.${field} 기준 시점 없음`);
    assert.ok(!region.sources.some((s) => s.fields.includes(field)), `${id}.${field} 출처 없음`);
  }
  // basis는 다섯 통계 필드에만, 모든 출처 항목은 실제 필드를 가리킨다.
  assert.deepEqual(Object.keys(region.basis).sort(), ["agingIndex", "birthRate", "history", "population", "riskIndex"]);
  for (const s of region.sources) for (const field of s.fields) assert.equal(region.basis[field], "source", `${id} ${s.title}`);
  // 백엔드 학습 데이터셋 조건: history는 10점, 모두 양수.
  assert.equal(region.history.length, 10);
  assert.ok(region.history.every((v) => Number.isFinite(v) && v > 0));
}
// 범위 밖 값(정책 효과·사례·비교 군)은 바꾸지 않았다.
assert.deepEqual(regionOf("namwon").policyImpacts, { welfare: 0.2, industry: 0.4, housing: 0.18 });
assert.equal(seed.benchmark_regions.length, 8);

console.log("regionStats: 남원·신안 공식 통계 값(인구·합계출산율)·기준 시점·출처, 나머지 demo_example 유지 PASS");
