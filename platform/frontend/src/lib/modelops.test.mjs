import assert from "node:assert/strict";
import { MODEL_TYPES, pipelineDefinition, datasetRegionName } from "./modelops.js";
const dataset = { id: "registered_real", content_hash: "a".repeat(64) };
const form = { id: "PL_new", name: "등록 학습", model_type: "factor_analysis", dataset_id: dataset.id };
assert.deepEqual(MODEL_TYPES.map((row) => row.id), ["factor_analysis", "simulation"]);
const body = pipelineDefinition(form, [dataset]);
assert.equal(body.dataset_id, dataset.id);
assert.equal(body.dataset_hash, dataset.content_hash);
assert.throws(() => pipelineDefinition({ ...form, dataset_id: "other_mode" }, [dataset]));
assert.throws(() => pipelineDefinition({ ...form, model_type: "population" }, [dataset]));
assert.throws(() => pipelineDefinition({ ...form, id: "../../x" }, [dataset]));
const regions = [{ id: "namwon", name: "남원시" }, { id: "shinan", name: "신안군" }];
// 목록을 다시 받아도 각 등록 행의 지역을 사용한다. 선택된 원본/다른 행의 지역에 의존하지 않는다.
const registered = [{ id: "real_namwon", region_id: "namwon" }, { id: "mock_shinan", region_id: "shinan" }];
assert.deepEqual(JSON.parse(JSON.stringify(registered)).map((row) => datasetRegionName(row, regions)), ["남원시", "신안군"]);
assert.equal(datasetRegionName({ region_id: "new_region" }, regions), "new_region");
assert.equal(datasetRegionName({}, regions), "—");
console.log("modelops: 두 종류 제한·등록 데이터셋 ID/hash 전달·입력 차단 PASS");
