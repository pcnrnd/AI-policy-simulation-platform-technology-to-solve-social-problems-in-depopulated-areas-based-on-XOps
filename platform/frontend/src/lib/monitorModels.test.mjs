import assert from "node:assert/strict";
import { monitorInstances, monitorActiveVersion, legacyMonitorId, regressionMetricText } from "./monitorModels.js";

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
assert.equal(legacyMonitorId(model, history, "mock"), null);
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
console.log("PASS: exact active metrics, zero/missing, explicit legacy mapping and mode isolation");
