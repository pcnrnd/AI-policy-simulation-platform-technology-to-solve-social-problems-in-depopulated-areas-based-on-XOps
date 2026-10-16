import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { MODEL_TYPES, pipelineDefinition, datasetRegionName, versionsWithStatus, jobNodeStatuses, connectorStatusFrom, jobStatusLabel,
  mergeRunHistory, jobLogLines, storeRows, lastJobOf, cardPipelineId, versionStatus, isLocked, jobLocks,
  instanceLabel, auditLogLines, auditTime, displayId } from "./modelops.js";
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
assert.ok(lines[2].message.endsWith("MAE 2 / 기준선 2 명 (검증 시점을 뺀 관측 평균)"), lines[2].message);
// 실행 로그 MAE·기준선은 학습 대상 단위 규칙(명·원 정수 천 단위, % 소수 1~2), 기준선 방법은 한글 이름 — 내부 코드명 금지
const BASELINE_CODES = /same_month_last_year|yoy_or_prev_month|leave_one_out_mean/;
const scoreLine = (target, mae, base, name) => jobLogLines(savedJob, { target, metrics: { mae }, baseline: { mae: base, name } }, [])[2].message;
assert.ok(scoreLine("living_population", 1224.215, 6020, "same_month_last_year").endsWith("MAE 1,224 / 기준선 6,020 명 (전년 같은 달 관측값)"));
assert.ok(scoreLine("population_change_pct", 0.0512, 0.3, "yoy_or_prev_month").endsWith("MAE 0.05 / 기준선 0.3 % (전년 같은 달·전월 관측값)"));
assert.ok(scoreLine("observed_sales_krw", 1890312.5, 2604410.3, "unknown_method").endsWith("MAE 1,890,313 / 기준선 2,604,410 원"));
for (const [target, name] of [["living_population", "same_month_last_year"], ["living_population", "yoy_or_prev_month"], ["living_population", "leave_one_out_mean"]]) {
  assert.doesNotMatch(scoreLine(target, 1, 2, name), BASELINE_CODES);
}
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

// ⑥ 등록 모델 이름 — 종류·지역·학습 대상(학습 카드와 같은 문구), 데모 요인 패널 대상 포함
const regionName = (id) => regions.find((row) => row.id === id)?.name ?? id;
assert.equal(instanceLabel({ model_type: "factor_analysis", region_id: "namwon", target: "living_population" }, regionName), "요인분석 · 남원시 · 생활인구");
assert.equal(instanceLabel({ model_type: "simulation", region_id: "shinan", target: "observed_sales_krw" }, regionName), "시뮬레이션 · 신안군 · 관측 소비");
// ⑥ 최근 운영 기록 → 콘솔 줄: 서버 최신 순을 뒤집어 최신이 아래, 20건 제한, 수준·메시지 형식
const auditRows = Array.from({ length: 25 }, (_, i) => ({ id: `a${i}`, created_at: new Date(2026, 9, 16, 14, 59 - i, 5).toISOString(),
  action: i === 0 ? "version_activated" : i === 1 ? "training_failed" : "pipeline_registered", asset_id: `inst${i}`, actor: "rnd", ...(i === 0 ? { version_id: "v9", job_id: "j9" } : {}) }));
const auditLines = auditLogLines(auditRows);
assert.equal(auditLines.length, 20);
assert.equal(auditLines.at(-1).message, "버전 활성화 — inst0 · 버전 v9 · 작업 j9 · rnd");
assert.equal(auditLines.at(-1).level, "SUCCESS");assert.equal(auditLines.at(-2).level, "ERROR");assert.equal(auditLines[0].level, "INFO");
assert.equal(auditLines.at(-1).time, "10-16 14:59:05");assert.equal(auditLines[0].time, "10-16 14:40:05");
assert.equal(auditLogLines([{ action: "unknown", asset_id: "x", actor: "a", created_at: "bad" }])[0].message, "운영 기록 — x · a");
assert.equal(auditTime("bad"), "bad");
assert.deepEqual(auditLogLines(), []);
// ⑥ 구조: 종류 카드 삭제, 운영 기록은 콘솔 패널, 카드 간격은 다른 카드와 같은 page-section
const section = readFileSync(join(dirname(fileURLToPath(import.meta.url)), "..", "components", "ModelVersionSection.jsx"), "utf-8");
assert.doesNotMatch(section, /MODEL_TYPES\.map|aria-label="모델 종류"|버전 보기|grid-cols-2|selectedType/);
assert.equal(section.match(/<ConsoleLog /g)?.length, 1);
assert.equal(section.match(/className="page-section/g)?.length, 2);
assert.match(section, /instanceLabel\(row, regionName\)/);
// 화면용 짧은 ID(공용) — 내부 접두사를 떼고 끝 8자(끝 조각 6~7자면 그 전체), 접두사가 없거나 짧으면 그대로. ⑥·Model Store·실행 이력이 같은 함수를 쓰고 title 에 전체 ID.
assert.equal(displayId("modelops_881abc0123456789abcdef0123456789"), "…23456789");
assert.equal(displayId("demo_factor_5f2e9a"), "…5f2e9a");assert.equal(displayId("demo_factor_5f2e9a1"), "…5f2e9a1");
// 끝 6자가 같던 두 ID(…1b9bfe)가 8자로는 갈린다
assert.notEqual(displayId("modelops_00000000000000000000000ad71b9bfe"), displayId("modelops_00000000000000000000000c041b9bfe"));
assert.equal(displayId("ver-sim-1"), "ver-sim-1");assert.equal(displayId("x_12"), "x_12");assert.equal(displayId(null), "");
const orchestratorPage = readFileSync(join(dirname(fileURLToPath(import.meta.url)), "..", "pages", "OrchestratorPage.jsx"), "utf-8");
const trainingPanel = readFileSync(join(dirname(fileURLToPath(import.meta.url)), "..", "components", "realdata", "RealdataTrainingPanel.jsx"), "utf-8");
assert.match(section, /<code title=\{id\}>\{displayId\(id\)\}<\/code>/);
assert.match(orchestratorPage, /title=\{m\.version\}>\{displayId\(m\.version\)\}/);
assert.doesNotMatch(orchestratorPage, /versionId\.slice|active_version\?\.slice|m\.version\.slice/);
// 내부 접두사만 뗀다 — 카드 파이프라인은 card-<종류>-, 사용자 ID·과거 기록 ID는 그대로
assert.equal(displayId(cardPipelineId("factor_analysis", "modelops_0123456789abcdef0123456789abcdef")), "…89abcdef");
assert.equal(displayId(cardPipelineId("simulation", "modelops_0123456789abcdef0123456789abcdef")), "…89abcdef");
assert.equal(displayId("PL_namwon_retrain"), "PL_namwon_retrain");assert.equal(displayId("run-20261015-001"), "run-20261015-001");
// 실행 로그·⑥ 운영 기록 줄도 짧은 표기
const longJob = { ...savedJob, id: "modelops_aaaaaaaaaaaaaaaaaaaaaaaaaa111111", pipeline_id: "card-factor_analysis-bbbbbb222222",
  dataset_id: "modelops_cccccccccccccccccccccccccc333333", version_id: "modelops_dddddddddddddddddddddddddd444444" };
const longLines = jobLogLines(longJob, null, [{ action: "version_activated", version_id: longJob.version_id, actor: "api", created_at: "2026-10-15T03:00:00Z" }]);
assert.deepEqual([0, 2, 3].map((i) => longLines[i].message), [`학습 접수 — 파이프라인 …bb222222 · 학습데이터 …cc333333(${"b".repeat(12)})`,
  "학습 결과 등록 — 버전 …dd444444", "버전 활성화 — …dd444444 · api"]);
assert.equal(auditLogLines([{ action: "version_activated", asset_id: "modelops_eeeeeeeeeeeeeeeeeeeeeeeeee555555", version_id: longJob.version_id, job_id: longJob.id, actor: "rnd", created_at: "bad" }])[0].message,
  "버전 활성화 — …ee555555 · 버전 …dd444444 · 작업 …aa111111 · rnd");
// 소스 가드: 오케스트레이터·시뮬레이터 화면이 ID 값을 본문({x.id}·${x.version_id}·orDash(x.id)·x.id.slice(…)·?? x.id)에 그대로 내지 않는다.
// title 속성(전체 ID)·키(modelops:${id})·과거 고정 모델 이름(model_id·modelId)·지역 코드(region.id)는 예외. ③ 계산 근거(ModelEvidence.jsx)는 근거 추적용으로 전체 ID를 두므로 대상에서 뺀다.
const RAW_ID = /(?<![=:$\w-])(?:\$\{|\{)(?:[^{}`]*?\?\?\s*)?(?:orDash\()?[\w?.]+?(?:\.id|_id|Id|\.version|_version)(?:\.slice\([^)]*\))?\)?\}/g;
// 선택 키에서 잘라 낸 ID({selectedKey.slice(…indexOf(":")…)})는 slice 인자에 괄호가 들어 있어 RAW_ID 가 놓쳤다 — 키 slice 형태를 따로 잡는다.
const KEY_SLICE = /(?<![=:$\w-])\{[\w?.]*Key\??\.slice\(/g;
const rawIds = (text) => {
  const body = text.replace(/title=\{`[^`]*`\}/g, "");
  return [...(body.match(RAW_ID) ?? []).filter((hit) => !/(?:model_?[iI]d|[rR]egion\.id)\)?\}$/.test(hit)), ...(body.match(KEY_SLICE) ?? [])];
};
const src = (...parts) => readFileSync(join(dirname(fileURLToPath(import.meta.url)), "..", ...parts), "utf-8");
const modelopsLib = src("lib", "modelops.js");
const guarded = {
  "pages/OrchestratorPage.jsx": orchestratorPage, "components/realdata/RealdataTrainingPanel.jsx": trainingPanel, "components/ModelVersionSection.jsx": section,
  "components/PipelineStatusCard.jsx": src("components", "PipelineStatusCard.jsx"), "components/FactorAnalysisStage.jsx": src("components", "FactorAnalysisStage.jsx"),
  "pages/SimulatorPage.jsx": src("pages", "SimulatorPage.jsx"), "hooks/useFactorAnalysis.js": src("hooks", "useFactorAnalysis.js"),
  "lib/modelops.js(로그 줄)": modelopsLib.slice(modelopsLib.indexOf("export function jobLogLines"), modelopsLib.indexOf("export function storeRows"))
};
for (const [file, text] of Object.entries(guarded)) assert.deepEqual(rawIds(text), [], `${file}: ID는 displayId(…)로 보이고 전체 ID는 title 에 둔다`);
// 가드가 실제로 잡는지 — 옛 직접 출력 형태 5종
assert.equal(rawIds("<code>{job.id}</code>{orDash(last.id)}{c.id.slice(0, 20)}`후보 ${latest.version_id}`{name ?? proof.dataset_id}").length, 5);
// 실행 로그 카드 제목의 옛 형태(D-1c)도 잡는다 — 지금은 displayId + title 전체 ID
assert.equal(rawIds('<code style={{ marginLeft: 8 }}>{selectedKey.slice(selectedKey.indexOf(":") + 1)}</code>').length, 1);
assert.match(orchestratorPage, /<code title=\{selectedRunId\}[^>]*>\{displayId\(selectedRunId\)\}<\/code>/);
// 활동·실행 로그를 만드는 소스에 기준선 내부 코드명 금지(한글 이름은 realSimulation.baselineName 한 곳)
for (const text of [orchestratorPage, modelopsLib]) assert.doesNotMatch(text, BASELINE_CODES);
assert.deepEqual(rawIds("<code title={c.id}>{displayId(c.id)}</code> key={`modelops:${job.id}`} title={`${last.id} — 로그`}"), []);
// ③ 계산 근거는 전체 ID 유지(예외)
assert.match(src("components", "ModelEvidence.jsx"), /\{p\.version_id\}/);
// MAE·기준선(Model Store·⑥ 버전 표·실행 이력)은 학습 대상 단위 규칙 — 명·원 정수 천 단위, % 소수 1~2자리
for (const text of [orchestratorPage, section]) { assert.match(text, /formatUnitValue\(value, targetUnit\(target\)\)/); assert.doesNotMatch(text, /maximumFractionDigits: 3/); }
assert.equal(orchestratorPage.match(/metric\([^)]*mae[^)]*, (?:m|version)\.target\)/gi)?.length, 4);
assert.match(section, /metric\(row\.metrics\.mae, row\.target\)\} \/ \{metric\(row\.baseline\.mae, row\.target\)/);
// 학습 카드 후보 버전 표도 같은 규칙 — MAE·RMSE·기준선은 대상 단위, WAPE는 비율(소수 2~3자리), 머리글에 단위
assert.match(trainingPanel, /const fmtNum = \(v\) => formatUnitValue\(v, unit\)/);
assert.doesNotMatch(trainingPanel, /toLocaleString\(undefined/);
assert.match(trainingPanel, /MAE \/ RMSE \(\{unit\}\) \/ WAPE \(비율\)[\s\S]*기준선 MAE \(\{unit\}\)/);
// 같은 내용 재등록은 서버가 기존 등록본을 돌려준다 — DataOps 안내는 "이미 등록" 문구로 구분한다.
const registration = readFileSync(join(dirname(fileURLToPath(import.meta.url)), "..", "components", "TrainingDatasetRegistration.jsx"), "utf-8");
assert.match(registration, /const reused = datasets\.some\(\(row\) => row\.id === registered\.id\);[\s\S]*이미 등록되어 있습니다/);
// 내부 알고리즘명 대신 "선형 회귀"
for (const src of [orchestratorPage, trainingPanel]) { assert.doesNotMatch(src, /Ridge|릿지/); assert.match(src, /선형 회귀/); }
console.log("modelops: 두 종류 제한·등록 데이터셋 ID/hash 전달·입력 차단·노드/이력/로그/저장소 매핑·⑥ 운영 기록 콘솔·등록 모델 이름·짧은 ID·선형 회귀 표기 PASS");
