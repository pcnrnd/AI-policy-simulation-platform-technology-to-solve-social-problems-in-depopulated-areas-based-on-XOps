// 리포트 지표 실데이터 매핑 회귀 검사 (vitest 미도입 → node 직접 실행).
//
// 합성 상수(accuracy 0.892 / psi 0.045 / outliers 0·3)로 되돌아가는 회귀를 막는다.
//   1. 지표 매핑 — evaluation 의 WAPE·MAE·기준선 MAE, drift 의 실측 PSI 가 그대로 실린다.
//   2. 빈 상태 — 활성 모델 없음 / model_required / 오류면 null 이고, 본문·표에 모델 검증 항목이 없다.
import assert from "node:assert/strict";
import {
  pickActiveModel,
  pickExplainTarget,
  pickModelDataset,
  toExplainBinding,
  toReportIndicators
} from "./dataopsApi.js";
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

// ── 3. SHAP 기여도 실호출 바인딩 ──────────────────────────────
// 합성 상수(+0.354 / +0.281 / -0.152)로 되돌아가는 회귀와, 최신월·최대 행정동 선택 규칙이
// 무너지는 회귀를 막는다.
const DATASETS = [
  { dataset_id: "ds_other", spec: { model_id: "other-model" }, observed_to: 202312 },
  { dataset_id: "ds_nw", spec: { model_id: MODEL.model_id }, observed_to: 202310 }
];
const ROWS_OK = {
  status: "ok",
  message: null,
  data: {
    rows: [
      { base_ym: 202309, dong_code: "45190250", y: 10, local_visitors: 99999 },
      { base_ym: 202310, dong_code: "45190250", y: 11, local_visitors: 120 },
      { base_ym: 202310, dong_code: "45190310", y: 12, local_visitors: 480 },
      { base_ym: 202310, dong_code: "45190320", y: 13, local_visitors: 300 }
    ]
  }
};
const TARGET = { baseYm: 202310, dongCode: "45190310" };
const EXPLAIN_OK = {
  status: "ok",
  data: {
    base_value: 100.0,
    contributions: [
      { feature: "y_lag1", value: 480, phi: 0.12 },
      { feature: "month_sin", value: 0.5, phi: -0.87 },
      { feature: "y_lag12", value: 410, phi: 0.35 }
    ],
    prediction: 99.6,
    note: "내부 설명 문구 — 리포트에 싣지 않는다."
  }
};
const DONG_MAP_OK = {
  status: "ok",
  data: { entries: [{ dong_name: "운봉읍", dong_code: "45190250" }, { dong_name: "주천면", dong_code: "45190310" }] }
};

check("활성 모델의 스냅샷만 고른다", () => {
  assert.equal(pickModelDataset(DATASETS, MODEL).dataset_id, "ds_nw");
  assert.equal(pickModelDataset(DATASETS, { model_id: "none" }), null);
  assert.equal(pickModelDataset([], MODEL), null);
  assert.equal(pickModelDataset(DATASETS, null), null);
});

check("최신 base_ym 에서 생활인구 최대 행정동을 고른다", () => {
  // 이전 달(202309)의 더 큰 local_visitors 에 끌려가지 않는다.
  assert.deepEqual(pickExplainTarget(ROWS_OK, 202310), TARGET);
});

check("rows 가 절단되면 빈 상태로 둔다", () => {
  const truncated = { ...ROWS_OK, message: "rows가 9000행이라 5000행으로 절단했습니다." };
  assert.equal(pickExplainTarget(truncated, 202310), null);
  assert.equal(pickExplainTarget({ status: "error", message: null, data: null }, 202310), null);
  assert.equal(pickExplainTarget(ROWS_OK, undefined), null);
});

check("local_visitors 가 없으면 다른 컬럼으로 대체하지 않고 빈 상태로 둔다", () => {
  // 소비 모델(_build_sales_rows) 스냅샷 모양 — sales_est_krw 만 있다.
  const sales = {
    status: "ok",
    message: null,
    data: { rows: [{ base_ym: 202310, dong_code: "45190250", y: 500, sales_est_krw: 7000 }] }
  };
  assert.equal(pickExplainTarget(sales, 202310), null);
  // 최신월에 해당 행이 아예 없어도 빈 상태다.
  assert.equal(pickExplainTarget(ROWS_OK, 202401), null);
});

check("기여도는 |phi| 내림차순이고 행정동명으로 표기한다", () => {
  const bound = toExplainBinding(EXPLAIN_OK, TARGET, DONG_MAP_OK);
  assert.equal(bound.baseYm, 202310);
  assert.equal(bound.dongCode, "45190310");
  assert.equal(bound.dongName, "주천면");
  assert.deepEqual(bound.contributions.map((c) => c.feature), ["month_sin", "y_lag12", "y_lag1"]);
  // note 는 내부 문구라 옮기지 않는다.
  assert.equal("note" in bound, false);
});

check("explain 이 ok 가 아니면 빈 상태로 둔다", () => {
  assert.equal(toExplainBinding({ status: "model_required", data: null }, TARGET, DONG_MAP_OK), null);
  assert.equal(toExplainBinding({ status: "empty", message: "관측행이 없습니다.", data: null }, TARGET, DONG_MAP_OK), null);
  assert.equal(toExplainBinding({ status: "error", message: "boom", data: null }, TARGET, DONG_MAP_OK), null);
  assert.equal(toExplainBinding(EXPLAIN_OK, null, DONG_MAP_OK), null);
  // 대응표를 못 읽으면 이름을 지어내지 않고 코드만 남긴다.
  assert.equal(toExplainBinding(EXPLAIN_OK, TARGET, { status: "error" }).dongName, null);
});

check("본문·표에 실측 기여도가 실리고 SHAP 합성 상수는 없다", () => {
  const explain = toExplainBinding(EXPLAIN_OK, TARGET, DONG_MAP_OK);
  const text = JSON.stringify(buildReportBlocks(REGION, TEMPLATE, { explain }));
  assert.match(text, /month_sin \(기여도 -0\.8700\)/);
  assert.match(text, /202310 · 주천면/);
  assert.doesNotMatch(text, /0\.354|0\.281|0\.152|청년 복지 예산 가중치|제조업 공장 일자리/);
  assert.doesNotMatch(text, /내부 설명 문구/);

  const rows = buildReportRows(REGION, TEMPLATE, { explain });
  assert.deepEqual(rows.find((r) => r[1] === "month_sin"), ["SHAP 기여 (202310 · 주천면)", "month_sin", -0.87]);
});

check("explain 바인딩이 없으면 본문·표에 SHAP 항목을 만들지 않는다", () => {
  const text = JSON.stringify(buildReportBlocks(REGION, TEMPLATE, {}));
  assert.doesNotMatch(text, /SHAP/);
  assert.equal(buildReportRows(REGION, TEMPLATE, {}).some((r) => String(r[0]).startsWith("SHAP")), false);
});

check("리포트 바인딩에 explain 이 그대로 실린다", () => {
  const explain = toExplainBinding(EXPLAIN_OK, TARGET, DONG_MAP_OK);
  assert.deepEqual(toReportIndicators(REGION, MODEL, EVALUATION_OK, DRIFT_OK, explain).explain, explain);
  // 넘기지 않으면 null — 기본값이 상수로 채워지지 않는다.
  assert.equal(toReportIndicators(REGION, MODEL, EVALUATION_OK, DRIFT_OK).explain, null);
});

console.log(`\n${passed} passed`);
