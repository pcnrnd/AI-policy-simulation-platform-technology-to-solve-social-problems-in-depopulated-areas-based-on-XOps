import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { monitorInstances, monitorActiveVersion, legacyMonitorId, defaultMonitorInstance, regressionMetricText, baselineRatio,
  runDateLabels, monitorFeatureLabel, DEMO_MONITOR_ID } from "./monitorModels.js";
import { compactNumber, compactRangeLabel } from "./realSimulation.js";

const model = { id: "common-instance", mode: "real", region_id: "namwon", target: "observed_sales_krw", active_version: "active" };
const active = { id: "active", instance_id: model.id, mode: "real", metrics: { mae: 0, rmse: 12, wape: 0.1 } };
const candidate = { id: "newer-candidate", instance_id: model.id, mode: "real", metrics: { mae: 999 } };
const catalog = { mode: "real", source: "real", instances: [model], versions: [candidate, active] };
const history = [{ model_id: "namwon-observed-sales-next-month", metrics_source: "realdata" }];
assert.equal(monitorActiveVersion(catalog, model, "real"), active);
assert.equal(monitorActiveVersion({ ...catalog, versions: [candidate] }, model, "real"), null);
assert.equal(monitorActiveVersion({ ...catalog, versions: [{ ...active, instance_id: "different" }] }, model, "real"), null);
assert.equal(monitorActiveVersion({ ...catalog, instances: [{ ...model, active_version: null }] }, model, "real"), null);
assert.deepEqual(monitorInstances(catalog, "mock"), []);
assert.equal(monitorActiveVersion(catalog, model, "mock"), null);
assert.equal(legacyMonitorId(model, history, "real"), history[0].model_id);
assert.equal(legacyMonitorId({ ...model, region_id: "shinan" }, history, "real"), null);
assert.equal(legacyMonitorId({ ...model, target: "nonlocal_visitors" }, history, "real"), null);
assert.equal(legacyMonitorId(model, [], "real"), null);
// 모드 격리: 실데이터 인스턴스를 데모 모드 이력에 붙이지 않는다.
assert.equal(legacyMonitorId(model, history, "mock"), null);
// 데모 ON: 등록 모델이 없거나(첫 진입·빈 카탈로그) 데모 모델이면 예시 이력을 본다 — 화면이 비지 않는다.
assert.equal(legacyMonitorId(null, [], "mock"), DEMO_MONITOR_ID);
assert.equal(legacyMonitorId({ ...model, mode: "mock", target: "population_change_pct" }, [], "mock"), DEMO_MONITOR_ID);
assert.equal(regressionMetricText(active.metrics.mae, "원"), "0원");
assert.equal(regressionMetricText(active.metrics.wape, "%", true), "10%");
for (const invalid of [null, undefined, NaN, Infinity, -1]) assert.equal(regressionMetricText(invalid, "명"), "–");
// 실제 mock catalog의 인구 변화율 활성 값: 원 단위의 %를 유지하며 WAPE에만 100을 곱한다.
const percentInstances = ["namwon", "shinan"].map((region) => ({ id: region, mode: "mock",
  region_id: region, target: "population_change_pct", active_version: region + "-active" }));
const percentVersions = [
  { id: "namwon-active", instance_id: "namwon", mode: "mock", metrics: { mae: 0.227835, rmse: 0.271744 } },
  { id: "shinan-active", instance_id: "shinan", mode: "mock", metrics: { mae: 0.431562, rmse: 0.538476 } }
];
const percentCatalog = { mode: "mock", source: "mock", instances: percentInstances, versions: percentVersions };
const displayedErrors = percentInstances.map((instance) => {
  const version = monitorActiveVersion(percentCatalog, instance, "mock");
  return [regressionMetricText(version.metrics.mae, "%"), regressionMetricText(version.metrics.rmse, "%")];
});
assert.deepEqual(displayedErrors, [["0.23%", "0.27%"], ["0.43%", "0.54%"]]);
assert.equal(regressionMetricText(0.097818, "%", true), "9.78%");
for (const unit of ["원", "명", "%"]) assert.equal(regressionMetricText(0, unit), `0${unit}`);
assert.equal(regressionMetricText(0, "%", true), "0%");
for (const invalid of [null, undefined, NaN, Infinity, -0.1]) {
  assert.equal(regressionMetricText(invalid, "%"), "–");
  assert.equal(regressionMetricText(invalid, "%", true), "–");
}
// U7: 데모 이력은 같은 학습 대상(인구 변화율) 데모 모델에만 — 다른 대상을 고르면 이력 없음(카드 전체가 같은 자리 안내)
const mockFactor = { id: "f", mode: "mock", target: "living_population", region_id: "shinan", active_version: "fv" };
const mockPct = { id: "p", mode: "mock", target: "population_change_pct", region_id: "namwon", active_version: "pv" };
assert.equal(legacyMonitorId(mockFactor, [], "mock"), null);
const mockCatalog = { mode: "mock", source: "mock", instances: [mockFactor, mockPct],
  versions: [{ id: "fv", instance_id: "f", mode: "mock" }, { id: "pv", instance_id: "p", mode: "mock" }] };
assert.equal(defaultMonitorInstance(mockCatalog, [], "mock"), mockPct); // 이력이 연결된 활성 모델 먼저
assert.equal(defaultMonitorInstance({ ...mockCatalog, instances: [mockFactor] }, [], "mock"), mockFactor);
assert.equal(defaultMonitorInstance(null, [], "mock"), null);
// U8: 지표 카드·게이지 공통 표기 — 1만 이상 축약, 단위 유지
assert.equal(regressionMetricText(102928071.8, "원"), "1.03억 원");
assert.equal(regressionMetricText(173182427, "원"), "1.73억 원");
assert.equal(regressionMetricText(27390, "명"), "2.7만 명");
assert.equal(regressionMetricText(2277, "명"), "2,277명");
assert.equal(compactNumber(-30000000), "-3,000만");assert.equal(compactNumber(0.05), "0.05");
// U10: 분포 구간 경계 축약(%·작은 수 구간 그대로), 평가지표 x축은 날짜 표시명(ID 없음)
assert.equal(compactRangeLabel("712,425–613,218,650"), "71.2만–6.13억");
assert.equal(compactRangeLabel("-5% ~ -3%"), "-5% ~ -3%");
assert.deepEqual(runDateLabels(["v20260911-78d661", "v20261014-78d661", "v20261014-78d661-r2", "RUN-20260921-0009", "x"]),
  ["09-11", "10-14", "10-14 (2)", "09-21", "5회차"]);
// U9: SHAP 특징 한글 표시명 — 내부 변수명 노출 없음
const dongs = [{ dong_code: "45190250", dong_name: "운봉읍" }];
assert.deepEqual(["rate_t-1", "rate_t-2", "riskIndex", "birthRate", "agingIndex", "dong_45190250", "dong_45190999", "month_cos", "y_lag2", "y_yoy", "has_yoy"]
  .map((feature) => monitorFeatureLabel(feature, "observed_sales_krw", dongs)),
["직전 기간 인구 변화율", "2기간 전 인구 변화율", "위험지수", "출산율", "고령화지수", "행정동: 운봉읍", "행정동 45190999", "월별 계절성 2", "2개월 전 관측 소비", "전년 같은 달 관측 소비", "전년 관측값 유무"]);
assert.equal(monitorFeatureLabel("스마트팜 단지 규모 (Smartfarm Scale)", "x"), "스마트팜 단지 규모");
// U11: 게이지 호 = 기준선 대비 비율(0~1), 기준선 없으면 null
assert.equal(baselineRatio(50, 100), 0.5);assert.equal(baselineRatio(150, 100), 1);
assert.equal(baselineRatio(1, undefined), null);assert.equal(baselineRatio(null, 2), null);assert.equal(baselineRatio(1, 0), null);
const monitorSrc = readFileSync(join(dirname(fileURLToPath(import.meta.url)), "..", "pages", "MonitorPage.jsx"), "utf-8");
assert.doesNotMatch(monitorSrc, /정규화 판정 없음/);
assert.match(monitorSrc, /neutral value=\{ratio\}/);
assert.match(monitorSrc, /SHAP 기반 \{selectedInstance \? targetLabel : "인구 유출"\} 기여 특징 중요도 분석/);
assert.match(monitorSrc, /autoSkip: false/);
console.log("PASS: exact active metrics, zero/missing, explicit legacy mapping and mode isolation, U7~U11 선택 모델 연결·축약·날짜 축·특징명·기준선 비율");
