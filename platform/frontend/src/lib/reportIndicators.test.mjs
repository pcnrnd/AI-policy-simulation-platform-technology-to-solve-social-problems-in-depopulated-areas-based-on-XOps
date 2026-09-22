// 리포트 지표 실데이터 매핑 회귀 검사 (vitest 미도입 → node 직접 실행).
//
// 합성 상수(accuracy 0.892 / psi 0.045 / outliers 0·3)로 되돌아가는 회귀를 막는다.
//   1. 지표 매핑 — evaluation 의 WAPE·MAE·기준선 MAE, drift 의 실측 PSI 가 그대로 실린다.
//   2. 빈 상태 — 활성 모델 없음 / model_required / 오류면 null 이고, 본문·표에 모델 검증 항목이 없다.
//   3. 피처 표기명 — 내부 피처명(y_lag1·month_sin·dong_*)이 리포트에 그대로 나가지 않는다.
//   4. 표시 판정 — 데모 OFF 라도 실데이터 바인딩이 있으면 표시한다(토글이 아니라 바인딩이 기준).
import assert from "node:assert/strict";
import {
  pickActiveModel,
  pickExplainTarget,
  pickModelDataset,
  featureLabel,
  toExplainBinding,
  toReportIndicators
} from "./dataopsApi.js";
import { buildReportBlocks, buildReportRows, reportGateMode } from "./reportContent.js";

let passed = 0;
const check = (name, fn) => {
  fn();
  passed += 1;
  console.log(`  ok  ${name}`);
};

const REGION = { id: "namwon", name: "전북 남원시", population: 76000, birthRate: 0.82, agingIndex: 33.1, riskIndex: 0.21 };
const TEMPLATE = { id: "template_analysis", title: "인구감소 대응 R&D 분석 리포트" };
const MODEL = { model_id: "namwon-nonlocal-visitors-next-month", active_version: "v3", target: "nonlocal_visitors" };

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
    features: [
      { feature: "y_lag1", psi: 0.0731, n_reference: 60, n_current: 23 },
      { feature: "dong_45190250", psi: 0.0102, n_reference: 60, n_current: 23 }
    ],
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
  // 표기명은 SHAP 기여도와 같은 매핑을 타고, 원본 피처명도 함께 남는다.
  // dong-map 을 받지 않는 경로라 `dong_*` 는 원본 그대로다(이름을 지어내지 않는다).
  assert.deepEqual(bound.indicators.psi, [
    { feature: "y_lag1", label: "직전월 외지인 방문객", psi: 0.0731 },
    { feature: "dong_45190250", label: "dong_45190250", psi: 0.0102 },
    { feature: "y", label: "외지인 방문객", psi: 0.1902 }
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
  assert.match(text, /PSI\): 직전월 외지인 방문객 0\.0731/);
  // 내부 기호 y 를 노출하던 "타깃(y)" 도 타깃 표기명으로 바뀐다.
  assert.match(text, /PSI\): 외지인 방문객 0\.1902/);
  assert.doesNotMatch(text, /y_lag1|타깃\(y\)/);
  assert.doesNotMatch(text, /0\.892|Accuracy|Outliers|이상치/);

  const rows = buildReportRows(REGION, TEMPLATE, { live });
  assert.deepEqual(rows.find((r) => r[1] === "WAPE"), ["모델 검증", "WAPE", 0.1234]);
  assert.deepEqual(rows.find((r) => r[1] === "기준선 MAE"), ["모델 검증", "기준선 MAE", 1010.0]);
  assert.deepEqual(rows.find((r) => r[1] === "PSI (외지인 방문객)"), ["모델 검증", "PSI (외지인 방문객)", 0.1902]);
  assert.equal(rows.some((r) => String(r[1]).includes("y_lag1")), false);
  // dong-map 없는 경로의 폴백 — 코드가 그대로 남는다(본문·엑셀 공통).
  assert.deepEqual(rows.find((r) => r[1] === "PSI (dong_45190250)"), ["모델 검증", "PSI (dong_45190250)", 0.0102]);
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
  const bound = toExplainBinding(EXPLAIN_OK, TARGET, DONG_MAP_OK, MODEL.target);
  assert.equal(bound.baseYm, 202310);
  assert.equal(bound.dongCode, "45190310");
  assert.equal(bound.dongName, "주천면");
  assert.deepEqual(bound.contributions.map((c) => c.feature), ["month_sin", "y_lag12", "y_lag1"]);
  // note 는 내부 문구라 옮기지 않는다.
  assert.equal("note" in bound, false);
});

check("explain 이 ok 가 아니면 빈 상태로 둔다", () => {
  assert.equal(toExplainBinding({ status: "model_required", data: null }, TARGET, DONG_MAP_OK, MODEL.target), null);
  assert.equal(toExplainBinding({ status: "empty", message: "관측행이 없습니다.", data: null }, TARGET, DONG_MAP_OK, MODEL.target), null);
  assert.equal(toExplainBinding({ status: "error", message: "boom", data: null }, TARGET, DONG_MAP_OK, MODEL.target), null);
  assert.equal(toExplainBinding(EXPLAIN_OK, null, DONG_MAP_OK, MODEL.target), null);
  // 대응표를 못 읽으면 이름을 지어내지 않고 코드만 남긴다.
  assert.equal(toExplainBinding(EXPLAIN_OK, TARGET, { status: "error" }, MODEL.target).dongName, null);
});

check("본문·표에 실측 기여도가 실리고 SHAP 합성 상수는 없다", () => {
  const explain = toExplainBinding(EXPLAIN_OK, TARGET, DONG_MAP_OK, MODEL.target);
  const text = JSON.stringify(buildReportBlocks(REGION, TEMPLATE, { explain }));
  assert.match(text, /계절성\(월, 사인\) \(기여도 -0\.8700\)/);
  assert.match(text, /202310 · 주천면/);
  assert.doesNotMatch(text, /0\.354|0\.281|0\.152|청년 복지 예산 가중치|제조업 공장 일자리/);
  assert.doesNotMatch(text, /내부 설명 문구/);

  const rows = buildReportRows(REGION, TEMPLATE, { explain });
  assert.deepEqual(rows.find((r) => r[1] === "계절성(월, 사인)"), ["SHAP 기여 (202310 · 주천면)", "계절성(월, 사인)", -0.87]);
  // 원본 피처명은 바인딩에 남지만 본문·엑셀로는 나가지 않는다(폴백 y_lag12 만 예외).
  assert.doesNotMatch(text, /month_sin|y_lag1/);
  assert.equal(rows.some((r) => r[1] === "y_lag1"), false);
});

check("explain 바인딩이 없으면 본문·표에 SHAP 항목을 만들지 않는다", () => {
  const text = JSON.stringify(buildReportBlocks(REGION, TEMPLATE, {}));
  assert.doesNotMatch(text, /SHAP/);
  assert.equal(buildReportRows(REGION, TEMPLATE, {}).some((r) => String(r[0]).startsWith("SHAP")), false);
});

check("리포트 바인딩에 explain 이 그대로 실린다", () => {
  const explain = toExplainBinding(EXPLAIN_OK, TARGET, DONG_MAP_OK, MODEL.target);
  assert.deepEqual(toReportIndicators(REGION, MODEL, EVALUATION_OK, DRIFT_OK, explain).explain, explain);
  // 넘기지 않으면 null — 기본값이 상수로 채워지지 않는다.
  assert.equal(toReportIndicators(REGION, MODEL, EVALUATION_OK, DRIFT_OK).explain, null);
});

check("피처 표기명은 모델 타깃에 따라 조립된다", () => {
  const DONGS = DONG_MAP_OK.data.entries;
  // y_* 는 타깃의 랙이다 — 생활인구가 아니라 모델 타깃 이름이 들어가야 한다.
  assert.equal(featureLabel("y_lag1", "nonlocal_visitors", DONGS), "직전월 외지인 방문객");
  assert.equal(featureLabel("y_lag1", "observed_sales_krw", DONGS), "직전월 소비 매출 추정액");
  assert.equal(featureLabel("y_lag2", "nonlocal_visitors", DONGS), "2개월 전 외지인 방문객");
  assert.equal(featureLabel("y_lag3", "nonlocal_visitors", DONGS), "3개월 전 외지인 방문객");
  assert.equal(featureLabel("y_yoy", "observed_sales_krw", DONGS), "전년 동월 소비 매출 추정액");
  assert.equal(featureLabel("has_yoy", "nonlocal_visitors", DONGS), "전년 동월 값 유무");
  // 두 계절성 피처가 같이 상위에 와도 서로 구분된다.
  assert.equal(featureLabel("month_sin", "nonlocal_visitors", DONGS), "계절성(월, 사인)");
  assert.equal(featureLabel("month_cos", "nonlocal_visitors", DONGS), "계절성(월, 코사인)");
  // 행정동 원핫은 대응표의 이름으로 바꾼다.
  assert.equal(featureLabel("dong_45190250", "nonlocal_visitors", DONGS), "행정동: 운봉읍");
});

check("뜻을 모르는 피처·타깃·행정동 코드는 원본을 그대로 남긴다", () => {
  const DONGS = DONG_MAP_OK.data.entries;
  assert.equal(featureLabel("unknown_feature", "nonlocal_visitors", DONGS), "unknown_feature");
  // 대응표에 없는 코드는 이름을 지어내지 않는다.
  assert.equal(featureLabel("dong_99999999", "nonlocal_visitors", DONGS), "dong_99999999");
  assert.equal(featureLabel("dong_45190250", "nonlocal_visitors", []), "dong_45190250");
  // 매핑에 없는 타깃 문자열은 그대로 쓰고, 타깃을 모르면 y_* 도 원본으로 둔다.
  assert.equal(featureLabel("y_lag1", "some_new_target", DONGS), "직전월 some_new_target");
  assert.equal(featureLabel("y_lag1", null, DONGS), "y_lag1");
  // 바인딩은 표기명과 원본 피처명을 함께 남긴다(추적 가능성).
  const bound = toExplainBinding(EXPLAIN_OK, TARGET, DONG_MAP_OK, MODEL.target);
  assert.deepEqual(bound.contributions.map((c) => c.feature), ["month_sin", "y_lag12", "y_lag1"]);
  assert.deepEqual(bound.contributions.map((c) => c.label), ["계절성(월, 사인)", "y_lag12", "직전월 외지인 방문객"]);
});

check("PSI 타깃 행은 타깃을 모르면 내부 기호 대신 폴백 문구를 쓴다", () => {
  const unknown = { ...MODEL, target: "some_new_target" };
  const psi = toReportIndicators(REGION, unknown, EVALUATION_OK, DRIFT_OK).indicators.psi;
  // 매핑에 없는 타깃 문자열은 그대로 쓴다(이름을 지어내지 않는다).
  assert.equal(psi.at(-1).label, "some_new_target");
  assert.equal(psi[0].label, "직전월 some_new_target");

  const noTarget = { model_id: MODEL.model_id, active_version: "v3" };
  const bare = toReportIndicators(REGION, noTarget, EVALUATION_OK, DRIFT_OK).indicators.psi;
  assert.equal(bare.at(-1).label, "타깃");
  // 타깃을 모르면 y_* 도 원본을 유지한다 — 내부 기호 y 는 어디에도 남지 않는다.
  assert.equal(bare[0].label, "y_lag1");
});

// ── 4. 표시 판정 ─────────────────────────────
const BINDING = toReportIndicators(REGION, MODEL, EVALUATION_OK, DRIFT_OK);

check("데모 OFF 라도 실데이터 바인딩이 있으면 표시한다", () => {
  assert.ok(BINDING, "전제: 바인딩이 만들어져야 한다");
  assert.equal(reportGateMode({ binding: BINDING, allowSeed: false }), "live");
  // 데모 ON 이어도 판정은 같다 — 토글은 바인딩이 없을 때만 의미가 있다.
  assert.equal(reportGateMode({ binding: BINDING, allowSeed: true }), "live");
});

check("데모 OFF + 바인딩 없음은 빈 상태다", () => {
  assert.equal(reportGateMode({ binding: null, allowSeed: false }), "empty");
});

check("데모 ON + 바인딩 없음은 시드 폴백이다", () => {
  assert.equal(reportGateMode({ binding: null, allowSeed: true }), "seed");
});

console.log(`\n${passed} passed`);
