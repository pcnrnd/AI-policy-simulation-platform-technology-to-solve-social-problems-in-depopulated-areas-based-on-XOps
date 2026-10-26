import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { datasetMatchesModel, mergeDatasets } from "./trainingDatasets.js";

const visitor = { model_id: "namwon-nonlocal-visitors-next-month", target: "nonlocal_visitors" };
const sales = { model_id: "namwon-observed-sales-next-month", target: "observed_sales_krw" };
assert.equal(datasetMatchesModel({ spec: sales }, visitor), false);
assert.equal(datasetMatchesModel({ spec: visitor }, visitor), true);
assert.equal(datasetMatchesModel({ spec: { ...visitor, target: sales.target } }, visitor), false);
assert.equal(datasetMatchesModel({ spec: {} }, visitor), false);
assert.equal(datasetMatchesModel(null, visitor), false);
console.log("trainingDatasets: 모델·target 일치 및 불일치 차단 PASS");

const oldSnapshot = { dataset_id: "ds-sales", spec: sales, row_count: 10 };
const newSnapshot = { ...oldSnapshot, row_count: 20 };
const visitorSnapshot = { dataset_id: "ds-visitors", spec: visitor };
assert.deepEqual(mergeDatasets([newSnapshot], [oldSnapshot, visitorSnapshot]), [newSnapshot, visitorSnapshot]);
assert.deepEqual(mergeDatasets([newSnapshot], mergeDatasets([newSnapshot], [oldSnapshot, visitorSnapshot])), [newSnapshot, visitorSnapshot]);
assert.deepEqual(mergeDatasets([oldSnapshot, oldSnapshot, visitorSnapshot, null, {}]), [oldSnapshot, visitorSnapshot]);
// DataOps 화면 가드(U12~U15): 토큰은 앞뒤 일부 + [복사], 응답 카드는 왼쪽 카드 높이에 맞춤, 태그 칩 줄바꿈 금지·긴 칸 말줄임,
// 라우팅 단계 상자 축소, 학습데이터 표 데이터셋 열은 원본 이름(식별자는 툴팁)
const src = (...p) => readFileSync(join(dirname(fileURLToPath(import.meta.url)), "..", ...p), "utf-8");
const dataopsPage = src("pages", "DataOpsPage.jsx");
assert.doesNotMatch(dataopsPage, /Authorization: Bearer \{token\}/);
assert.match(dataopsPage, /Authorization: Bearer \{maskToken\(token\)\}/);
assert.match(dataopsPage, /navigator\.clipboard\.writeText\(token\)/);
const css = src("styles", "dataops.css");
assert.match(css, /\.dataops-builder-grid \{[^}]*align-items: stretch;/);
assert.match(css, /\.dataops-builder-grid \.api-response \{[^}]*flex: 1 1 auto;[^}]*height: 0;/);
assert.match(css, /\.catalog-tag-chip \{[^}]*white-space: nowrap;/);
assert.match(css, /\.routing-step \{[^}]*min-width: 0;/);
assert.match(css, /\.catalog-table th,\s*\.catalog-table td \{\s*padding-left: 10px;\s*padding-right: 10px;/); // 1280에서 9열 표 944→899px
const registration = src("components", "TrainingDatasetRegistration.jsx");
assert.match(registration, /title=\{`\$\{row\.id\} · 원본 \$\{row\.source_dataset_id \?\? "—"\}`\}>\{sourceName\(row\)\}/);
assert.doesNotMatch(registration, /<span title=\{row\.id\}>\{row\.id\}<\/span>/);
console.log("trainingDatasets: 스냅샷 ID 갱신·재생성·중복 목록 제거·DataOps 토큰 가림/카드 높이/1280 잘림/데이터셋 이름 PASS");
