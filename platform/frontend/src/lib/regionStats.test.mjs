// 남원·신안 지역 통계 출처 검사 (vitest 미도입 → node 직접 실행).
// 공식 통계로 바꾼 값은 원자료 수치와 같고, 값 필드마다 basis(source | demo_example)·기준 시점·출처가 있다.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { RISK_GRADES, RISK_GRADE_SOURCE, riskGrade } from "./riskGrade.js";

const here = dirname(fileURLToPath(import.meta.url));
const seed = JSON.parse(readFileSync(join(here, "..", "assets", "mock_data.json"), "utf-8"));
const regionOf = (id) => seed.regions.find((r) => r.id === id);

// 원자료 수치(2026-11-06 확인):
// - population: 행정안전부 주민등록 인구통계, 2025년 12월 총인구(거주자·거주불명자·재외국민, 외국인 제외)
// - birthRate: KOSIS 인구동향조사 시군구 합계출산율 2025년(확정)
// - agingIndex(화면 라벨 "고령인구비율"): 주민등록 65세 이상 ÷ 전체(2025년 12월) — 남원 25,624/74,272, 신안 16,146/41,858
// - riskIndex: 한국고용정보원 지방소멸위험지수 원시자료(2025년 6월, 지수 ÷ 100) — 남원 20.40, 신안 12.19
const OFFICIAL = {
  namwon: { population: 74272, birthRate: 0.853, agingIndex: 34.5, riskIndex: 0.204 },
  shinan: { population: 41858, birthRate: 0.896, agingIndex: 38.6, riskIndex: 0.122 }
};
const OLD65 = { namwon: 25624, shinan: 16146 };
const PERIODS = { population: "2025-12", birthRate: "2025", agingIndex: "2025-12", riskIndex: "2025-06" };
const ORIGIN = /^https:\/\/(jumin\.mois\.go\.kr|kosis\.kr|www\.keis\.or\.kr)\//;
// 바꾸지 않고 demo_example로 둔 값 — history: 공식 값(주민등록 2015~2024년 6월)은 확인했지만 백엔드 데모 학습·요인 패널이
// 이 시드 이력에 맞춰 고정돼 있어 데모 학습과 함께 바꾸는 별도 작업으로 남긴다(11-06 사용자 권장안 승인).
const NOT_OFFICIAL = {
  namwon: { history: [85200, 83800, 82400, 81100, 80300, 79400, 78600, 78000, 77600, 77200] },
  shinan: { history: [45600, 44700, 43800, 42600, 41700, 40800, 40000, 39300, 38800, 38400] }
};
// 지역 이름은 원자료 현행 행정구역 이름(주민등록 인구통계 2026-11 조회 기준).
const NAMES = { namwon: "전북특별자치도 남원시", shinan: "전남광주통합특별시 신안군" };

for (const [id, values] of Object.entries(OFFICIAL)) {
  const region = regionOf(id);
  assert.ok(region, id);
  assert.equal(region.name, NAMES[id]);
  for (const [field, value] of Object.entries(values)) {
    assert.deepEqual(region[field], value, `${id}.${field}`);
    assert.equal(region.basis[field], "source", `${id}.${field} basis`);
    assert.equal(region.periods[field], PERIODS[field], `${id}.${field} 기준 시점`);
    const cited = region.sources.filter((s) => s.fields.includes(field));
    assert.equal(cited.length, 1, `${id}.${field} 출처`);
    assert.match(cited[0].url, ORIGIN, `${id}.${field} 공공기관 원자료`);
    assert.match(cited[0].checked, /^\d{4}-\d{2}-\d{2}$/);
    assert.ok(cited[0].title, `${id}.${field} 출처 제목`);
  }
  // 고령인구비율은 계산 값 — 계산 방법을 출처에 적고, 적힌 수치로 다시 계산하면 같은 값이다.
  const aging = region.sources.find((s) => s.fields.includes("agingIndex"));
  assert.match(aging.method, /65세 이상 인구 ÷ 전체 인구/);
  assert.equal(Math.round((OLD65[id] / region.population) * 1000) / 10, region.agingIndex);
  assert.ok(aging.method.includes(OLD65[id].toLocaleString("en-US")) && aging.method.includes(region.population.toLocaleString("en-US")));
  assert.match(region.sources.find((s) => s.fields.includes("riskIndex")).method, /20~39세 여성 인구 ÷ 65세 이상 인구/);
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

// 비교 8개 군: 현행 이름, 위험지수(한국고용정보원 2025-06)·인구(주민등록 2025-12) 원자료 값, 출처 기록.
assert.deepEqual(seed.benchmark_regions, [
  { name: "경상북도 의성군", riskIndex: 0.086, population: 47902 },
  { name: "전남광주통합특별시 고흥군", riskIndex: 0.099, population: 59455 },
  { name: "경상남도 합천군", riskIndex: 0.09, population: 39190 },
  { name: "전북특별자치도 임실군", riskIndex: 0.125, population: 25259 },
  { name: "충청남도 부여군", riskIndex: 0.128, population: 58377 },
  { name: "강원특별자치도 정선군", riskIndex: 0.15, population: 34879 },
  { name: "대구광역시 군위군", riskIndex: 0.078, population: 22452 },
  { name: "충청북도 괴산군", riskIndex: 0.105, population: 38293 }
]);
const bench = seed.benchmark_basis;
assert.deepEqual(bench.basis, { riskIndex: "source", population: "source" });
assert.deepEqual(bench.periods, { riskIndex: "2025-06", population: "2025-12" });
for (const field of ["riskIndex", "population"]) {
  const cited = bench.sources.filter((s) => s.fields.includes(field));
  assert.equal(cited.length, 1, field);
  assert.match(cited[0].url, ORIGIN);
  assert.match(cited[0].checked, /^\d{4}-\d{2}-\d{2}$/);
}
// 범위 밖 값(정책 효과)은 바꾸지 않았다.
assert.deepEqual(regionOf("namwon").policyImpacts, { welfare: 0.2, industry: 0.4, housing: 0.18 });

// 화면 라벨: 고령인구비율(%)로 통일 — 지역 현황 카드·리포트·모델 특징 이름.
const read = (...path) => readFileSync(join(here, "..", ...path), "utf-8");
const labelFiles = [["components", "RegionStatusCard.jsx"], ["pages", "ReporterPage.jsx"], ["lib", "reportContent.js"], ["lib", "monitorModels.js"]];
for (const path of labelFiles) {
  const src = read(...path);
  assert.doesNotMatch(src, /고령화\s?지수/, path.join("/"));
  assert.match(src, /고령인구비율/, path.join("/"));
}

// 위험 등급(수정 라운드 1): 한국고용정보원 2025 원시자료 자료설명의 소멸위험분류 6단계(×100 척도 → ÷100)를 모든 화면이 같이 쓴다.
assert.deepEqual(RISK_GRADES.map((g) => [g.label, g.rangeLabel]), [
  ["심각", "0.1 미만"], ["위험", "0.1 이상 0.2 미만"], ["경계", "0.2 이상 0.4 미만"], ["관리", "0.4 이상 0.6 미만"], ["보통", "0.6 이상 1.0 미만"], ["양호", "1.0 이상"]
]);
assert.deepEqual(RISK_GRADES.map((g) => g.maxExclusive), [0.1, 0.2, 0.4, 0.6, 1.0, Number.POSITIVE_INFINITY]);
assert.match(RISK_GRADE_SOURCE.url, /^https:\/\/www\.keis\.or\.kr\//);
assert.deepEqual([RISK_GRADE_SOURCE.period, RISK_GRADE_SOURCE.checked], ["2025-06", "2026-11-06"]);
assert.deepEqual([riskGrade(0.099).label, riskGrade(0.1).label, riskGrade(0.2).label, riskGrade(0.4).label, riskGrade(0.6).label, riskGrade(1).label], ["심각", "위험", "경계", "관리", "보통", "양호"]);
// 남원 0.204 → 경계(원시자료 ncls 3), 신안 0.122 → 위험(4), 비교 8개 군은 원시자료 분류대로.
assert.deepEqual(seed.regions.map((r) => [r.id, riskGrade(r.riskIndex).label]), [["namwon", "경계"], ["shinan", "위험"]]);
assert.deepEqual(seed.benchmark_regions.map((r) => riskGrade(r.riskIndex).label), ["심각", "심각", "심각", "위험", "위험", "위험", "심각", "위험"]);
// 등급을 내는 곳은 모두 공용 기준을 쓰고 자체 구간(0.15·0.18)이나 고정 문구("극심한", "경보")가 없다.
const mapSrc = read("components", "SimulatorMap.jsx");
const cardSrc = read("components", "RegionStatusCard.jsx");
assert.match(mapSrc, /import \{ RISK_GRADES, riskGrade \} from "\.\.\/lib\/riskGrade\.js";/);
assert.match(cardSrc, /import \{ riskGrade \} from "\.\.\/lib\/riskGrade\.js";/);
for (const [name, src] of [["SimulatorMap", mapSrc], ["RegionStatusCard", cardSrc]]) assert.doesNotMatch(src, /0\.15|0\.18|function riskGrade|maxExclusive:/, name);
assert.match(read("pages", "SimulatorPage.jsx"), /import SimulatorMap, \{ SimulatorMapDetails, riskGrade \} from "\.\.\/components\/SimulatorMap\.jsx";/);
const reporterSrc = read("pages", "ReporterPage.jsx");
const reportSrc = read("lib", "reportContent.js");
assert.match(reporterSrc, /\(소멸위험 <strong>\{riskGrade\(region\.riskIndex\)\.label\}<\/strong> 단계\)/);
assert.match(reportSrc, /\(소멸위험 \$\{riskGrade\(region\.riskIndex\)\.label\} 단계\)/);
for (const src of [reporterSrc, reportSrc]) assert.doesNotMatch(src, /극심한|위험등급 경보/);

console.log("regionStats: 남원·신안 공식 통계(인구·합계출산율·고령인구비율·위험지수)·현행 이름·비교 8개 군·출처, 이력 demo_example 유지·라벨·위험 등급(한국고용정보원 2025) PASS");
