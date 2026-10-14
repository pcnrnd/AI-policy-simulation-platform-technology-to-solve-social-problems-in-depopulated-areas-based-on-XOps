import assert from "node:assert/strict";
import { datasetMatchesModel } from "./trainingDatasets.js";

const visitor = { model_id: "namwon-nonlocal-visitors-next-month", target: "nonlocal_visitors" };
const sales = { model_id: "namwon-observed-sales-next-month", target: "observed_sales_krw" };
assert.equal(datasetMatchesModel({ spec: sales }, visitor), false);
assert.equal(datasetMatchesModel({ spec: visitor }, visitor), true);
assert.equal(datasetMatchesModel({ spec: { ...visitor, target: sales.target } }, visitor), false);
assert.equal(datasetMatchesModel({ spec: {} }, visitor), false);
assert.equal(datasetMatchesModel(null, visitor), false);
console.log("trainingDatasets: 모델·target 일치 및 불일치 차단 PASS");
