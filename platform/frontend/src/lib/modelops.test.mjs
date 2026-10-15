import assert from "node:assert/strict";
import { MODEL_TYPES, pipelineDefinition, datasetRegionName, versionsWithStatus, jobNodeStatuses, connectorStatusFrom, jobStatusLabel,
  mergeRunHistory, jobLogLines, storeRows, lastJobOf, cardPipelineId, versionStatus, isLocked, jobLocks } from "./modelops.js";
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

// 실험 ID: 앞뒤 공백 제거, 없으면 빈 값, 64자 초과 거부
assert.equal(body.experiment, "");
assert.equal(pipelineDefinition({ ...form, experiment: "  EXP-1 " }, [dataset]).experiment, "EXP-1");
assert.throws(() => pipelineDefinition({ ...form, experiment: "x".repeat(65) }, [dataset]));

// 버전 상태 파생은 백엔드 service.versions 규칙과 같다
const instance = { id: "inst", active_version: "v2", previous_version: "v1" };
const versions = versionsWithStatus([{ id: "v3", instance_id: "inst" }, { id: "v2", instance_id: "inst" }, { id: "v1", instance_id: "inst" }], [instance]);
assert.deepEqual(versions.map((row) => row.status), ["candidate", "active", "previous"]);
// 버전 상태 용어는 Model Store·학습 카드와 같은 "운영/이전/후보"
assert.deepEqual(versions.map((row) => versionStatus(row.status)), ["후보", "운영", "이전"]);
assert.equal(versionStatus(undefined), "후보");

// 잠금: 진행 중 작업은 잠그지만 상태 폴링이 실패하면 풀린다(요청 중 busy는 그대로 잠금)
const runningJob = { id: "j", state: "running" };
assert.equal(isLocked(false, runningJob, null), true);
assert.equal(isLocked(false, { id: "j", state: "queued" }, null), true);
assert.equal(isLocked(false, runningJob, { message: "502" }), false);
assert.equal(jobLocks(runningJob, { message: "502" }), false);
assert.equal(isLocked(true, runningJob, { message: "502" }), true);
assert.equal(isLocked(false, { id: "j", state: "saved" }, null), false);
assert.equal(isLocked(false, null, null), false);

// 6단계 노드: 작업 상태·버전 결과별 표시
const classes = (job, version) => jobNodeStatuses(job, version).map((n) => n.className);
const good = { metrics: { mae: 1 }, baseline: { mae: 2 } };
assert.deepEqual(classes(null), Array(6).fill("idle"));
assert.deepEqual(classes({ state: "queued" }), ["running", "idle", "idle", "idle", "idle", "idle"]);
assert.deepEqual(classes({ state: "running" }), ["completed", "running", "running", "running", "idle", "idle"]);
assert.deepEqual(classes({ state: "saved" }, { ...good, status: "candidate" }), ["completed", "completed", "completed", "completed", "completed", "idle"]);
assert.equal(jobNodeStatuses({ state: "saved" }, { ...good, status: "candidate" })[5].label, "활성화 대기");
const below = jobNodeStatuses({ state: "saved" }, { metrics: { mae: 2 }, baseline: { mae: 2 }, status: "candidate" });
assert.deepEqual([below[3].className, below[3].label, below[5].label], ["failed", "기준선 미달", "활성화 불가"]);
assert.equal(jobNodeStatuses({ state: "saved" }, { ...good, status: "active" })[5].label, "활성");
assert.equal(jobNodeStatuses({ state: "saved" }, { ...good, status: "previous" })[5].label, "이전 버전");
assert.deepEqual(classes({ state: "failed", error: "restart" }), ["completed", "failed", "idle", "idle", "idle", "idle"]);
assert.deepEqual(classes({ state: "failed", error: "ValueError" }), ["completed", "failed", "failed", "failed", "idle", "idle"]);
assert.deepEqual(classes({ state: "failed", failed_stage: "training" }), ["completed", "completed", "failed", "idle", "idle", "idle"]);
assert.deepEqual(classes({ state: "failed", failed_stage: "registering" }), ["completed", "completed", "completed", "completed", "failed", "idle"]);
const runningNodes = jobNodeStatuses({ state: "running" });
assert.deepEqual([0, 1, 4].map((i) => connectorStatusFrom(runningNodes, i)), ["active", "", ""]);
assert.deepEqual([jobStatusLabel(null).label, jobStatusLabel({ state: "running" }).label, jobStatusLabel({ state: "saved" }, { metrics: {}, baseline: {}, status: "candidate" }).label,
  jobStatusLabel({ state: "saved" }, { ...good, status: "active" }).label], ["대기", "진행 중", "승급 없음", "활성"]);

// 실행 이력: 데모 OFF면 옛 기록 제외, ON이면 병합·최신 시작 순·키 충돌 없음
const jobs = [{ id: "j2", pipeline_id: "p1", created_at: "2026-10-15T02:00:00Z", started_at: "2026-10-15T02:00:01Z", state: "saved" },
  { id: "j1", pipeline_id: "gone", created_at: "2026-10-14T00:00:00Z", state: "failed" }];
const legacy = [{ run_id: "j2", model_id: "population-forecast", started_at: "2026-10-15T01:00:00Z", state: "rejected" }];
assert.deepEqual(mergeRunHistory(jobs, legacy, false, [{ id: "p1", trigger_policy: "수동" }]).map((row) => row.key), ["modelops:j2", "modelops:j1"]);
const merged = mergeRunHistory(jobs, legacy, true, [{ id: "p1", trigger_policy: "수동" }]);
assert.deepEqual(merged.map((row) => row.key), ["modelops:j2", "legacy:j2", "modelops:j1"]);
assert.deepEqual(merged.map((row) => row.readOnly), [false, true, false]);
assert.equal(merged[2].pipelineMissing, true);
assert.equal(lastJobOf({ id: "p1" }, jobs).id, "j2");
assert.equal(lastJobOf({ id: "none" }, jobs), null);

// 로그: 접수→시작→등록→활성화 순, 학습 완료 감사 중복 없음
const savedJob = { id: "j2", pipeline_id: "p1", dataset_id: "d1", dataset_hash: "b".repeat(64), version_id: "v9", state: "saved",
  created_at: "2026-10-15T02:00:00Z", started_at: "2026-10-15T02:00:01Z", finished_at: "2026-10-15T02:00:05Z" };
const audit = [{ action: "version_activated", version_id: "v9", actor: "api", created_at: "2026-10-15T03:00:00Z" },
  { action: "training_saved", job_id: "j2", version_id: "v9", actor: "api", created_at: "2026-10-15T02:00:05Z" },
  { action: "version_activated", version_id: "other", actor: "api", created_at: "2026-10-15T04:00:00Z" }];
const lines = jobLogLines(savedJob, { metrics: { mae: 1.5 }, baseline: { mae: 2, name: "leave_one_out_mean" } }, audit);
assert.deepEqual(lines.map((row) => row.message.split(" — ")[0]), ["학습 접수", "학습 시작", "학습 결과 등록", "버전 활성화"]);
assert.ok(lines[2].message.includes("MAE 1.5 / 기준선 2"));
assert.equal(jobLogLines({ ...savedJob, state: "failed", error: "restart", started_at: undefined }, null, [])[1].message, "학습 중단 — 서버 재시작");

// Model Store: 이전 고정 모델은 데모 ON에서만, 새 버전 뒤에 읽기 전용으로
const legacyStore = [{ modelId: "population-forecast", version: "v3.0-R3", status: "운영", accuracy: 0.892 }];
const fresh = [{ id: "v9", instance_id: "inst", dataset_id: "d1", features: ["a", "b"], metrics: { mae: 1 }, baseline: { mae: 2 }, status: "active" }];
assert.deepEqual(storeRows(fresh, [{ id: "d1", name: "학습 입력" }], legacyStore, false).map((row) => row.origin), ["modelops"]);
const store = storeRows(fresh, [{ id: "d1", name: "학습 입력" }], legacyStore, true);
assert.deepEqual(store.map((row) => [row.origin, row.datasetName ?? row.originalStatus]), [["modelops", "학습 입력"], ["legacy", "운영"]]);
assert.equal(store[0].featureCount, 2);

// 카드 자동 파이프라인 ID: 결정적·백엔드 ID 패턴 통과
const cardId = cardPipelineId("simulation", "modelops_0123456789abcdef0123456789abcdef");
assert.equal(cardId, cardPipelineId("simulation", "modelops_0123456789abcdef0123456789abcdef"));
assert.notEqual(cardId, cardPipelineId("factor_analysis", "modelops_0123456789abcdef0123456789abcdef"));
assert.match(cardId, /^[A-Za-z0-9_-]{3,128}$/);
console.log("modelops: 두 종류 제한·등록 데이터셋 ID/hash 전달·입력 차단·노드/이력/로그/저장소 매핑 PASS");
