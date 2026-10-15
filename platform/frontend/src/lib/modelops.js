export const MODEL_TYPES = [{ id: "factor_analysis", name: "요인분석" }, { id: "simulation", name: "시뮬레이션" }];
export const TARGET_LABELS = { nonlocal_visitors: "타지역 방문객", observed_sales_krw: "관측 소비", population_change_pct: "인구 변화율" };
export const JOB_STATE_NAMES = { queued: "대기", running: "학습 중", saved: "등록 완료", failed: "실패" };
export const EXPERIMENT_MAX = 64;
export const typeName = (id) => MODEL_TYPES.find((type) => type.id === id)?.name ?? id ?? "—";
export function datasetRegionName(dataset, regions = []) {
  return regions.find((region) => region.id === dataset.region_id)?.name ?? dataset.region_id ?? "—";
}
export function pipelineDefinition(form, datasets) {
  if (!MODEL_TYPES.some((type) => type.id === form.model_type)) throw new Error("대상 모델을 선택하세요.");
  const dataset = datasets.find((row) => row.id === form.dataset_id);
  if (!dataset) throw new Error("등록된 학습데이터를 선택하세요.");
  if (!/^[A-Za-z0-9_-]{3,128}$/.test(form.id.trim())) throw new Error("파이프라인 ID는 영문·숫자·밑줄·하이픈 3~128자로 입력하세요.");
  if (!form.name.trim()) throw new Error("파이프라인 이름을 입력하세요.");
  const experiment = (form.experiment ?? "").trim();
  if (experiment.length > EXPERIMENT_MAX) throw new Error(`실험 ID는 ${EXPERIMENT_MAX}자 이하로 입력하세요.`);
  return { id: form.id.trim(), name: form.name.trim(), model_type: form.model_type,
    dataset_id: dataset.id, dataset_hash: dataset.content_hash, trigger_policy: form.trigger_policy || "수동", experiment };
}
export function versionStatus(status) { return ({ active: "활성", previous: "이전", candidate: "등록" })[status] ?? "등록"; }

/** 버전 상태는 인스턴스의 활성/이전 버전에서 파생한다(백엔드 service.versions 와 같은 규칙). */
export function versionsWithStatus(versions = [], instances = []) {
  const byId = new Map(instances.map((row) => [row.id, row]));
  return versions.map((version) => {
    const instance = byId.get(version.instance_id);
    const status = version.id === instance?.active_version ? "active" : version.id === instance?.previous_version ? "previous" : "candidate";
    return { ...version, status };
  });
}

/** 활성화 조건(MAE < 기준선 MAE)을 만족하지 못하는 버전. */
export function belowBaseline(version) {
  const mae = version?.metrics?.mae;
  const base = version?.baseline?.mae;
  return !(Number.isFinite(mae) && Number.isFinite(base) && mae < base);
}

export const datasetMatchesInstance = (dataset, instance) =>
  Boolean(instance && dataset && dataset.region_id === instance.region_id && dataset.target === instance.target);

/** 실데이터 학습 카드가 자동 등록하는 파이프라인 ID — 같은 (종류, 등록본)이면 같은 ID라 다시 쓰인다. */
export const cardPipelineId = (modelType, datasetId) => `card-${modelType}-${String(datasetId).slice(-12)}`;

/** 파이프라인별 마지막 작업(서버가 created_at 내림차순으로 준다). */
export const lastJobOf = (pipeline, jobs = []) => jobs.find((job) => job.pipeline_id === pipeline.id) ?? null;

const node = (className, label, icon) => ({ className, label, icon });
const IDLE = node("idle", "대기", "fa-clock");
const DONE = node("completed", "완료", "fa-check");
const FAILED = node("failed", "실패", "fa-triangle-exclamation");
const FAILED_STAGE_INDEX = { preparing: 1, training: 2, registering: 4 };

/**
 * 새 경로 작업·버전 → 6단계 노드(접수·데이터 검증·학습·평가·버전 등록·활성화).
 * 새 경로는 단계 시각을 저장하지 않으므로 작업 상태와 버전 결과로 결정적으로 그린다.
 */
export function jobNodeStatuses(job, version) {
  if (!job) return Array(6).fill(IDLE);
  if (job.state === "queued") return [node("running", "진행 중", "fa-spinner fa-spin"), ...Array(5).fill(IDLE)];
  if (job.state === "running") {
    const running = node("running", "진행 중", "fa-spinner fa-spin");
    return [DONE, running, running, running, IDLE, IDLE];
  }
  if (job.state === "failed") {
    if (job.error === "restart") return [DONE, node("failed", "중단", "fa-triangle-exclamation"), IDLE, IDLE, IDLE, IDLE];
    const at = FAILED_STAGE_INDEX[job.failed_stage];
    if (at === undefined) return [DONE, FAILED, FAILED, FAILED, IDLE, IDLE];
    return Array.from({ length: 6 }, (_, index) => (index < at ? DONE : index === at ? FAILED : IDLE));
  }
  const below = belowBaseline(version);
  const evaluation = version && below ? node("failed", "기준선 미달", "fa-ban") : DONE;
  const activation = version?.status === "active" ? node("completed", "활성", "fa-check")
    : version?.status === "previous" ? node("idle", "이전 버전", "fa-rotate-left")
      : version && below ? node("idle", "활성화 불가", "fa-ban") : node("idle", "활성화 대기", "fa-clock");
  return [DONE, DONE, DONE, evaluation, node("completed", "등록", "fa-check"), activation];
}

/** 노드 사이 연결선: 왼쪽이 완료면 success, 오른쪽이 진행 중이면 active. */
export function connectorStatusFrom(nodes, index) {
  if (nodes[index]?.className !== "completed") return "";
  return nodes[index + 1]?.className === "running" ? "active" : "success";
}

/** 상태 칩 라벨과 색 계열(idle·busy·failed·warn·ok). */
export function jobStatusLabel(job, version) {
  if (!job) return { label: "대기", tone: "idle" };
  if (job.state === "queued" || job.state === "running") return { label: "진행 중", tone: "busy" };
  if (job.state === "failed") return { label: "실패", tone: "failed" };
  if (version?.status === "active") return { label: "활성", tone: "ok" };
  if (version?.status === "previous") return { label: "이전 버전", tone: "idle" };
  if (version && belowBaseline(version)) return { label: "승급 없음", tone: "warn" };
  return { label: "등록 완료", tone: "ok" };
}

const time = (value) => (value ? Date.parse(value) || 0 : 0);

/** 실행 이력: 새 경로 작업 + (데모 ON일 때만) 옛 경로 과거 기록. 최신 시작 순. */
export function mergeRunHistory(jobs = [], legacyRuns = [], demoVisible = false, pipelines = []) {
  const pipelineById = new Map(pipelines.map((row) => [row.id, row]));
  const fresh = jobs.map((job) => ({
    key: `modelops:${job.id}`, origin: "modelops", id: job.id, pipelineId: job.pipeline_id,
    pipelineMissing: !pipelineById.has(job.pipeline_id), modelType: job.model_type, instanceId: job.instance_id,
    datasetId: job.dataset_id, trigger: pipelineById.get(job.pipeline_id)?.trigger_policy ?? "수동",
    state: job.state, startedAt: job.started_at ?? job.created_at, finishedAt: job.finished_at ?? null,
    versionId: job.version_id, error: job.error ?? null, readOnly: false
  }));
  const legacy = demoVisible ? legacyRuns.map((run) => ({
    key: `legacy:${run.run_id}`, origin: "legacy", id: run.run_id, pipelineId: run.pipeline_id ?? null,
    model: run.model_id, datasetId: run.dataset_id ?? null, trigger: run.trigger, state: run.state,
    startedAt: run.started_at, finishedAt: run.finished_at ?? null, versionId: run.active_version ?? null,
    error: run.error ?? null, readOnly: true, raw: run
  })) : [];
  return [...fresh, ...legacy].sort((a, b) => time(b.startedAt) - time(a.startedAt));
}

const metric = (value) => (Number.isFinite(value) ? String(Math.round(value * 1000) / 1000) : "–");

/** 새 경로 작업의 로그 — 작업 레코드·버전·감사 기록에서 만든다(저장된 사실만). */
export function jobLogLines(job, version, audit = []) {
  if (!job) return [];
  const lines = [{ ts: job.created_at, level: "INFO",
    message: `학습 접수 — 파이프라인 ${job.pipeline_id} · 학습데이터 ${job.dataset_id}(${String(job.dataset_hash ?? "").slice(0, 12)})` }];
  if (job.started_at) lines.push({ ts: job.started_at, level: "INFO", message: "학습 시작" });
  if (job.state === "saved") {
    const scores = version ? ` · MAE ${metric(version.metrics?.mae)} / 기준선 ${metric(version.baseline?.mae)}(${version.baseline?.name ?? "기준선"})` : "";
    lines.push({ ts: job.finished_at, level: "INFO", message: `학습 결과 등록 — 버전 ${job.version_id}${scores}` });
  }
  if (job.state === "failed") {
    lines.push({ ts: job.finished_at, level: "ERROR",
      message: job.error === "restart" ? "학습 중단 — 서버 재시작" : `학습 실패 — ${job.error ?? "원인 미기록"}${job.failed_stage ? ` (${job.failed_stage})` : ""}` });
  }
  const related = audit.filter((row) => (row.action === "version_activated" || row.action === "version_rolled_back")
    && (row.job_id === job.id || row.version_id === job.version_id));
  [...related].sort((a, b) => time(a.created_at) - time(b.created_at)).forEach((row) => {
    lines.push({ ts: row.created_at, level: "INFO",
      message: `${row.action === "version_activated" ? "버전 활성화" : "이전 버전 복원"} — ${row.version_id} · ${row.actor}` });
  });
  return lines;
}

/** Model Store 행: 새 경로 버전(최신 순) + 데모 ON에서만 이전 고정 모델 과거 버전(읽기 전용). */
export function storeRows(versions = [], datasets = [], legacyStore = [], demoVisible = false) {
  const datasetById = new Map(datasets.map((row) => [row.id, row]));
  const fresh = versions.map((version) => ({
    key: `modelops:${version.id}`, origin: "modelops", modelType: version.model_type, regionId: version.region_id,
    target: version.target, instanceId: version.instance_id, version: version.id,
    datasetName: datasetById.get(version.dataset_id)?.name ?? version.dataset_id, datasetHash: version.dataset_hash,
    featureCount: Array.isArray(version.features) ? version.features.length : 0,
    mae: version.metrics?.mae, baselineMae: version.baseline?.mae, status: version.status, createdAt: version.created_at
  }));
  const legacy = demoVisible ? legacyStore.map((row) => ({
    key: `legacy:${row.modelId}-${row.version}`, origin: "legacy", modelId: row.modelId, version: row.version,
    dataVersion: row.dataVersion, params: row.params, accuracy: row.accuracy, originalStatus: row.status,
    registeredAt: row.registeredAt, source: row.source ?? "mock"
  })) : [];
  return [...fresh, ...legacy];
}
