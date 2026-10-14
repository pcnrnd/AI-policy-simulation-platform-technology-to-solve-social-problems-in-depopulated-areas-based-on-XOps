import assert from "node:assert/strict";
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
console.log("trainingDatasets: 스냅샷 ID 갱신·재생성·중복 목록 제거 PASS");
