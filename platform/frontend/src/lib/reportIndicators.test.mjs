// 리포트 지표 실데이터 매핑 회귀 검사 (vitest 미도입 → node 직접 실행).
//
// 합성 상수(accuracy 0.892 / psi 0.045 / outliers 0·3)로 되돌아가는 회귀를 막는다.
//   1. 지표 매핑 — evaluation 의 WAPE·MAE·기준선 MAE, drift 의 실측 PSI 가 그대로 실린다.
//   2. 빈 상태 — 활성 모델 없음 / model_required / 오류면 null 이고, 본문·표에 모델 검증 항목이 없다.
import assert from "node:assert/strict";
import { pickActiveModel, toReportIndicators } from "./dataopsApi.js";
import { buildReportBlocks, buildReportRows } from "./reportContent.js";

let passed = 0;
const check = (name, fn) => {
  fn();
  passed += 1;
  console.log(`  ok  ${name}`);
};

const REGION = { id: "namwon", name: "전북 남원시", population: 76000, birthRate: 0.82, agingIndex: 33.1, riskIndex: 0.21 };
const TEMPLATE = { id: "template_analysis", title: "인구감소 대응 R&D 분석 리포트" };
const MODEL = { model_id: "namwon-nonlocal-visitors-next-month", active_version: "v3" };

const EVALUATION_OK = {
  status: "ok",
  data: {
    validation: { kind: "validation", metrics: { mae: 812.5, rmse: 990.1, wape: 0.1234 }, baseline: { mae: 1010.0 } },
    operational: { kind: "pending" }
  }
};
const DRIFT_OK = {
  status: "ok",
  data: {
    status: "ok",
    kind: "validation",
    features: [{ feature: "y_lag1", psi: 0.0731, n_reference: 60, n_current: 23 }],
    target: { psi: 0.1902, n_reference: 60, n_current: 23 }
  }
};

// ── 1. 지표 매핑 ─────────────────────────────────────────────
check("활성 버전이 있는 모델만 고른다", () => {
  assert.equal(pickActiveModel([{ model_id: "a", active_version: null }, MODEL]), MODEL);
  assert.equal(pickActiveModel([{ model_id: "a", active_version: null }]), null);
  assert.equal(pickActiveModel([]), null);
});

check("evaluation·drift 실측값이 그대로 지표로 실린다", () => {
  const bound = toReportIndicators(REGION, MODEL, EVALUATION_OK, DRIFT_OK);
  assert.equal(bound.source, "/api/v3/realdata/models/namwon-nonlocal-visitors-next-month/evaluation");
  assert.equal(bound.version, "v3");
  assert.deepEqual(bound.indicators.wape, 0.1234);
  assert.deepEqual(bound.indicators.mae, 812.5);
  assert.deepEqual(bound.indicators.baselineMae, 1010.0);
  assert.deepEqual(bound.indicators.psi, [
    { label: "y_lag1", psi: 0.0731 },
    { label: "타깃(y)", psi: 0.1902 }
  ]);
  // 대응 엔드포인트가 없는 지표는 만들지 않는다.
  assert.equal("accuracy" in bound.indicators, false);
  assert.equal("outliers" in bound.indicators, false);
});

check("드리프트 비교 구간이 없으면 PSI 는 빈 목록이다", () => {
  const noDrift = { status: "ok", data: { status: "none", kind: null, features: [], target: null } };
  assert.deepEqual(toReportIndicators(REGION, MODEL, EVALUATION_OK, noDrift).indicators.psi, []);
  assert.deepEqual(toReportIndicators(REGION, MODEL, EVALUATION_OK, { status: "model_required" }).indicators.psi, []);
});

check("본문·표에 실측 WAPE·MAE·PSI 가 들어가고 합성 상수는 없다", () => {
  const live = toReportIndicators(REGION, MODEL, EVALUATION_OK, DRIFT_OK).indicators;
  const text = JSON.stringify(buildReportBlocks(REGION, TEMPLATE, { live }));
  assert.match(text, /WAPE\): 0\.123/);
  assert.match(text, /MAE\): 812\.500 \(기준선 MAE 1010\.000\)/);
  assert.match(text, /PSI\): y_lag1 0\.0731/);
  assert.doesNotMatch(text, /0\.892|Accuracy|Outliers|이상치/);

  const rows = buildReportRows(REGION, TEMPLATE, { live });
  assert.deepEqual(rows.find((r) => r[1] === "WAPE"), ["모델 검증", "WAPE", 0.1234]);
  assert.deepEqual(rows.find((r) => r[1] === "기준선 MAE"), ["모델 검증", "기준선 MAE", 1010.0]);
  assert.deepEqual(rows.find((r) => r[1] === "PSI (타깃(y))"), ["모델 검증", "PSI (타깃(y))", 0.1902]);
});

// ── 2. 빈 상태 ───────────────────────────────────────────────
check("활성 모델·검증지표가 없으면 바인딩이 null 이다", () => {
  assert.equal(toReportIndicators(REGION, null, EVALUATION_OK, DRIFT_OK), null);
  assert.equal(toReportIndicators(REGION, MODEL, { status: "model_required", data: null }, DRIFT_OK), null);
  assert.equal(toReportIndicators(REGION, MODEL, { status: "error", message: "boom" }, DRIFT_OK), null);
  assert.equal(toReportIndicators(REGION, MODEL, { status: "ok", data: {} }, DRIFT_OK), null);
});

check("바인딩이 없으면 본문·표에 모델 검증 항목을 만들지 않는다", () => {
  const text = JSON.stringify(buildReportBlocks(REGION, TEMPLATE, {}));
  assert.doesNotMatch(text, /검증지표|WAPE|PSI/);
  assert.equal(buildReportRows(REGION, TEMPLATE, {}).some((r) => r[0] === "모델 검증"), false);
});

console.log(`\n${passed} passed`);
