import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Card from "../components/Card.jsx";
import ConsoleLog from "../components/ConsoleLog.jsx";
import InfoTip from "../components/InfoTip.jsx";
import ModelPipelineRegisterForm from "../components/ModelPipelineRegisterForm.jsx";
import NextStepBanner from "../components/NextStepBanner.jsx";
import PipelineStatusCard, { LEGACY_RUN_STATE_LABEL, logTime, orDash } from "../components/PipelineStatusCard.jsx";
import TablePager, { paginate, usePager } from "../components/TablePager.jsx";
import RealdataTrainingPanel from "../components/realdata/RealdataTrainingPanel.jsx";
import { getRealdataToken } from "../components/realdata/realdataClient.jsx";
import { useAppState } from "../context/AppStateContext.jsx";
import { MODEL_REGISTRY } from "../constants/models.js";
import { apiGet, apiSend } from "../lib/api.js";
import { alertText } from "../lib/alertText.js";
import {
  JOB_STATE_NAMES, TARGET_LABELS, auditLogLines, auditTime, displayId, instanceLabel, isLocked, jobLocks, jobLogLines, lastJobOf, mergeRunHistory, storeRows, typeName, versionsWithStatus
} from "../lib/modelops.js";
import { formatUnitValue, targetUnit } from "../lib/realSimulation.js";

const PAGE_SIZE = 5;
const JOB_POLL_MS = 1500;
const POLL_MAX_FAILURES = 3;
const EMPTY = { instances: [], datasets: [], versions: [], pipelines: [], jobs: [], audit: [] };

// 승급(활성화) 기준 — 새 경로 service.activate 의 실제 검사와 같은 문구. 자동 승급·자동 롤백은 없다.
const PROMOTION_RULE =
  "학습이 끝난 버전은 '후보'로 등록됩니다. 운영(활성) 버전이 되려면 ① 학습 작업이 정상 완료되고 ② 등록 학습데이터와 모델 파일의 내용 해시가 저장 당시와 같으며 ③ 검증 평균 절대 오차(MAE)가 기준선보다 작아야 합니다. 기준선은 데모 데이터에서 나머지 기간 평균, 실데이터에서 전년 같은 달(없으면 전월) 값으로 예측한 오차입니다. 활성화는 [활성화]를 눌러 실행하며, [이전 버전 복원]은 직전 활성 버전으로만 되돌립니다. 자동 승급·자동 롤백은 하지 않습니다.";

const STORE_STATUS = {
  active: { label: "운영", color: "var(--accent-teal)", bg: "rgba(var(--accent-teal-rgb), 0.02)" },
  previous: { label: "이전", color: "var(--text-muted)", bg: "var(--surface-hover)" },
  candidate: { label: "후보", color: "var(--accent-orange)", bg: "rgba(var(--accent-orange-rgb), 0.02)" },
  legacy: { label: "과거 버전", color: "var(--text-muted)", bg: "var(--surface-hover)" }
};

// MAE·기준선은 학습 대상 단위 규칙(시뮬레이터 ①과 같은 formatUnitValue)으로 보인다.
const metric = (value, target) => formatUnitValue(value, targetUnit(target));
const day = (value) => (value ? String(value).slice(0, 10) : null);

export default function OrchestratorPage() {
  const { appData, modelStore, consoleLogs, addConsoleLog, navigateToTab, mockDataVisible } = useAppState();
  // 데모 표시는 화면 구조가 아니라 데이터만 바꾼다. 새 경로는 X-Data-Mode(mock|real)로 모드별 자산을 받고,
  // 이전 경로(옛 시드 파이프라인·고정 모델 3개)의 과거 기록은 데모 ON에서만 읽기 전용으로 붙인다.
  const demo = mockDataVisible;
  const [catalog, setCatalog] = useState(EMPTY);
  const [catalogLoaded, setCatalogLoaded] = useState(false);
  const [catalogError, setCatalogError] = useState(null);
  const [legacyRuns, setLegacyRuns] = useState([]);
  const [registerOpen, setRegisterOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [feedback, setFeedback] = useState(null);
  const [versionResult, setVersionResult] = useState(null); // { origin: "store", ok, message }
  const clearVersionResult = useCallback(() => setVersionResult(null), []);
  const [liveJob, setLiveJob] = useState(null);
  const [pollError, setPollError] = useState(null); // { message, stopped } — 상태 폴링 실패(잠금 해제·다시 확인)
  const [selectedKey, setSelectedKey] = useState(null); // "modelops:<job>" | "legacy:<run>" — ②·⑤가 함께 본다
  const [legacyLogs, setLegacyLogs] = useState(null);
  const [versionInstanceId, setVersionInstanceId] = useState("");
  const [plPage, setPlPage] = useState(1);
  const [runPage, setRunPage] = useState(1);
  const [statusFocusRequest, setStatusFocusRequest] = useState(0);
  const mounted = useRef(true);
  const catalogRequest = useRef(0);
  const jobTimer = useRef(null);
  // 폴링 세대 — 새 실행·[다시 확인]·[초기화]마다 올려 이전 루프의 늦은 응답과 타이머를 버린다.
  const pollGen = useRef(0);
  const statusAnchorRef = useRef(null);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; clearTimeout(jobTimer.current); }; }, []);

  const reloadCatalog = useCallback(async () => {
    const request = ++catalogRequest.current;
    try {
      const token = await getRealdataToken();
      const response = await apiGet("/api/v3/modelops/catalog", { token });
      if (!mounted.current || request !== catalogRequest.current) return;
      setCatalog({ ...EMPTY, ...response });
      setCatalogError(null);
    } catch (err) {
      if (!mounted.current || request !== catalogRequest.current) return;
      setCatalogError(alertText(err, "잠시 뒤 다시 시도하세요."));
      addConsoleLog(`ERROR: 파이프라인 카탈로그 로드 실패 — ${err?.message ?? "알 수 없는 오류"}`);
    } finally {
      if (mounted.current && request === catalogRequest.current) setCatalogLoaded(true);
    }
  }, [addConsoleLog]);

  useEffect(() => { reloadCatalog(); }, [reloadCatalog]);

  // 이전 경로 과거 기록 — 데모 ON에서만 읽는다(새 실행·등록·삭제는 이 경로로 보내지 않는다).
  useEffect(() => {
    if (!demo) { setLegacyRuns([]); return undefined; }
    let alive = true;
    apiGet(`/api/v3/orchestration/runs?include_seed=${demo}`)
      .then((rows) => alive && setLegacyRuns(Array.isArray(rows) ? rows : []))
      .catch((err) => alive && addConsoleLog(`WARN: 이전 실행 기록 조회 실패 — ${err?.message ?? "알 수 없는 오류"}`));
    return () => { alive = false; };
  }, [demo, addConsoleLog]);

  const versions = useMemo(() => versionsWithStatus(catalog.versions, catalog.instances), [catalog.versions, catalog.instances]);
  const versionById = useMemo(() => new Map(versions.map((row) => [row.id, row])), [versions]);
  const instanceById = useMemo(() => new Map(catalog.instances.map((row) => [row.id, row])), [catalog.instances]);
  const pipelineById = useMemo(() => new Map(catalog.pipelines.map((row) => [row.id, row])), [catalog.pipelines]);
  const history = useMemo(() => mergeRunHistory(catalog.jobs, legacyRuns, demo, catalog.pipelines), [catalog.jobs, catalog.pipelines, legacyRuns, demo]);
  const regionName = (id) => appData.regions.find((row) => row.id === id)?.name ?? id;
  const datasetName = (id) => catalog.datasets.find((row) => row.id === id)?.name ?? id;

  const selectedRow = history.find((row) => row.key === selectedKey) ?? null;
  const selectedRunId = selectedKey ? selectedKey.slice(selectedKey.indexOf(":") + 1) : null; // 작업 ID 또는 과거 기록 실행 ID
  const selectedJob = selectedKey?.startsWith("modelops:")
    ? (liveJob && `modelops:${liveJob.id}` === selectedKey ? liveJob : catalog.jobs.find((row) => `modelops:${row.id}` === selectedKey) ?? null)
    : null;
  const statusTarget = selectedJob
    ? { kind: "modelops", job: selectedJob, version: versionById.get(selectedJob.version_id), instance: instanceById.get(selectedJob.instance_id), pipeline: pipelineById.get(selectedJob.pipeline_id) }
    : selectedRow?.origin === "legacy" ? { kind: "legacy", run: selectedRow.raw } : null;
  const jobActive = jobLocks(liveJob, pollError);
  const locked = isLocked(busy, liveJob, pollError);

  // 선택한 이전 경로 기록의 저장 로그(단건 조회라 include_seed 불요).
  useEffect(() => {
    if (!selectedKey?.startsWith("legacy:")) { setLegacyLogs(null); return undefined; }
    const runId = selectedKey.slice("legacy:".length);
    let alive = true;
    apiGet(`/api/v3/orchestration/runs/${encodeURIComponent(runId)}/logs`)
      .then((data) => alive && setLegacyLogs(data))
      .catch((err) => {
        if (!alive) return;
        setLegacyLogs(null);
        addConsoleLog(`WARN: 실행 로그 조회 실패 (${displayId(runId)}) — ${err?.message ?? "알 수 없는 오류"}`);
      });
    return () => { alive = false; };
  }, [selectedKey, addConsoleLog]);

  const logLines = selectedJob
    ? jobLogLines(selectedJob, versionById.get(selectedJob.version_id), catalog.audit)
    : legacyLogs?.logs ?? [];

  useEffect(() => {
    if (statusFocusRequest === 0) return;
    const anchor = statusAnchorRef.current;
    if (!anchor) return;
    anchor.focus({ preventScroll: true });
    anchor.scrollIntoView({ block: "start" });
  }, [statusFocusRequest]);

  // 작업 상태 폴링 — 종결(saved/failed)까지. 실패하면 화면을 잠그지 않고 ②에 알리며 다시 시도하고,
  // 연속 POLL_MAX_FAILURES회면 멈춘다([다시 확인]으로 재개). 호출마다 세대를 올려 이전 루프를 무효화한다.
  const pollJob = useCallback((jobId, delay) => {
    const gen = ++pollGen.current;
    clearTimeout(jobTimer.current);
    const current = () => mounted.current && gen === pollGen.current;
    let failures = 0;
    const tick = async () => {
      try {
        const token = await getRealdataToken();
        const latest = await apiGet(`/api/v3/modelops/jobs/${encodeURIComponent(jobId)}`, { token });
        if (!current()) return;
        failures = 0;
        setPollError(null);
        setLiveJob(latest);
        if (latest.state !== "saved" && latest.state !== "failed") { jobTimer.current = setTimeout(tick, JOB_POLL_MS); return; }
        setFeedback(latest.state === "saved" ? "학습 결과를 등록했습니다. 버전을 확인한 뒤 활성화하세요." : "학습데이터와 관측 기간을 확인한 뒤 다시 실행하세요.");
        addConsoleLog(latest.state === "saved"
          ? `INFO: 학습 완료 — ${displayId(latest.id)} · 후보 ${displayId(latest.version_id)}`
          : `WARN: 학습 실패 — ${displayId(latest.id)} (${latest.error ?? "원인 미기록"})`, false, latest.state === "failed");
        await reloadCatalog();
        if (!current()) return;
        setVersionInstanceId(latest.instance_id);
      } catch (error) {
        if (!current()) return;
        failures += 1;
        const stopped = failures >= POLL_MAX_FAILURES;
        setPollError({ message: alertText(error, null), stopped });
        addConsoleLog(`WARN: 학습 상태 확인 실패 (${failures}/${POLL_MAX_FAILURES}) — ${displayId(jobId)}: ${error?.message ?? "알 수 없는 오류"}`, false, true);
        if (!stopped) jobTimer.current = setTimeout(tick, JOB_POLL_MS);
        // 멈춘 뒤에는 행 [실행] 잠금의 원천(catalog.jobs)을 서버 기준으로 한 번 맞춘다.
        else reloadCatalog();
      }
    };
    jobTimer.current = setTimeout(tick, delay);
  }, [addConsoleLog, reloadCatalog]);

  // 파이프라인 실행 — 202 접수 후 폴링. 실데이터 학습 카드도 이 함수를 쓴다. 진행 중 잠금은 liveJob 이 맡는다.
  const runPipeline = useCallback(async (pipeline) => {
    if (locked) return;
    setBusy(true);
    setFeedback(null);
    try {
      const token = await getRealdataToken();
      const accepted = await apiSend("POST", `/api/v3/modelops/pipelines/${encodeURIComponent(pipeline.id)}/run`, { token });
      if (!mounted.current) return;
      setPollError(null);
      setLiveJob(accepted);
      setSelectedKey(`modelops:${accepted.id}`);
      setStatusFocusRequest((request) => request + 1);
      addConsoleLog(`INFO: 재학습 파이프라인 ${displayId(pipeline.id)}(${pipeline.name}) 실행 접수 — ${displayId(accepted.id)}`);
      pollJob(accepted.id, 500);
      await reloadCatalog();
    } catch (error) {
      if (!mounted.current) return;
      setFeedback(error.status === 409 ? "선택한 모델의 학습이 진행 중입니다." : alertText(error, "파이프라인을 실행하지 못했습니다. 학습데이터를 확인한 뒤 다시 시도하세요."));
      addConsoleLog(`WARN: 파이프라인 실행 실패 — ${error.message ?? "알 수 없는 오류"}`, false, true);
    } finally { if (mounted.current) setBusy(false); }
  }, [locked, addConsoleLog, reloadCatalog, pollJob]);

  // 활성화·복원 — 결과는 페이지 상태 하나(versionResult)에 두고 Model Store(origin "store")가 그린다.
  // 학습 카드·모델 선택 변경은 onClearResult 로 낡은 문구를 지운다.
  const activate = useCallback(async (instanceId, version, rollback, origin) => {
    if (locked) {
      setVersionResult({ origin, ok: false, message: "진행 중인 작업이 끝난 뒤 다시 시도하세요." });
      return;
    }
    setBusy(true);
    setVersionResult(null);
    try {
      const token = await getRealdataToken();
      await apiSend("POST", `/api/v3/modelops/instances/${encodeURIComponent(instanceId)}/versions/${encodeURIComponent(version.id)}/${rollback ? "rollback" : "activate"}`, { token });
      if (!mounted.current) return;
      await reloadCatalog();
      const message = rollback ? "이전 버전을 복원했습니다." : "선택한 버전을 활성화했습니다.";
      setFeedback(message);
      setVersionResult({ origin, ok: true, message });
      addConsoleLog(`INFO: ${rollback ? "이전 버전 복원" : "버전 활성화"} — ${displayId(version.id)}`);
    } catch (error) {
      const message = alertText(error, "학습 결과를 확인한 뒤 다시 시도하세요.");
      addConsoleLog(`WARN: ${rollback ? "복원" : "활성화"} 실패 — ${error.message ?? "알 수 없는 오류"}`, false, true);
      if (!mounted.current) return;
      setFeedback(message);
      setVersionResult({ origin, ok: false, message: `${rollback ? "복원" : "활성화"} 실패 — ${message}` });
      // 409·400 뒤에도 외부 변경이 있을 수 있어 활성/이전 표시를 서버 기준으로 다시 맞춘다.
      await reloadCatalog();
    } finally { if (mounted.current) setBusy(false); }
  }, [locked, addConsoleLog, reloadCatalog]);

  // aria-disabled 는 클릭을 막지 않으므로(§9 A11Y-01 규약) 실행 차단은 여기서 한다.
  const handleDeletePipeline = async (pipeline) => {
    if (locked || !window.confirm(`${pipeline.name} 파이프라인 등록을 해제할까요? 실행 이력은 남습니다.`)) return;
    setBusy(true);
    try {
      const token = await getRealdataToken();
      await apiSend("DELETE", `/api/v3/modelops/pipelines/${encodeURIComponent(pipeline.id)}`, { token });
      addConsoleLog(`INFO: 재학습 파이프라인 등록 해제 — ${displayId(pipeline.id)} (실행 이력은 보존됩니다)`);
      await reloadCatalog();
    } catch (err) {
      addConsoleLog(`WARN: 파이프라인 삭제 실패 — ${err?.message ?? "알 수 없는 오류"}`, false, true);
    } finally { if (mounted.current) setBusy(false); }
  };

  const handleReset = () => {
    pollGen.current += 1; // 응답 대기 중인 폴링도 결과를 버린다
    clearTimeout(jobTimer.current);
    setLiveJob(null);
    setPollError(null);
    setSelectedKey(null);
    addConsoleLog("INFO: MLOps 재학습 파이프라인 실행 상태 표시를 초기화했습니다.");
    // 화면 표시만 끊었으므로 행 [실행] 잠금은 서버의 실제 작업 상태로 다시 맞춘다.
    reloadCatalog();
  };

  const pl = paginate(catalog.pipelines, plPage, PAGE_SIZE);
  const storeInstance = catalog.instances.find((row) => row.id === versionInstanceId) ?? null;
  const storeAll = storeRows(versions, catalog.datasets, modelStore, demo)
    .filter((row) => !storeInstance || row.instanceId === storeInstance.id);
  const store = usePager(storeAll, PAGE_SIZE, storeInstance?.id ?? "");
  const storeResult = versionResult?.origin === "store" ? versionResult : null;
  const auditLines = auditLogLines(catalog.audit);
  const runList = paginate(history, runPage, PAGE_SIZE);
  const activatedVersion = statusTarget?.kind === "modelops" && statusTarget.version?.status === "active" ? statusTarget.version : null;

  return (
    <>
      {feedback && <p className="async-feedback" role="status">{feedback}</p>}
      {/* ① 진입점: 등록된 재학습 파이프라인 — 등록·실행·등록 해제 */}
      <Card
        title="등록된 재학습 파이프라인"
        icon="fa-list-check"
        className="page-section orchestrator-live"
        dataSource="api"
        headerRight={
          <span style={{ display: "inline-flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
            <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
              {catalogLoaded ? `${catalog.pipelines.length}건 등록` : "불러오는 중"} · 수동 실행
            </span>
            <button className="btn btn-primary" style={{ padding: "5px 14px", fontSize: 12 }} onClick={() => setRegisterOpen((open) => !open)} aria-expanded={registerOpen}>
              <i className="fa-solid fa-plus" aria-hidden="true"></i> 파이프라인 등록
            </button>
            <button className="btn btn-secondary" style={{ padding: "5px 12px", fontSize: 12 }} onClick={() => reloadCatalog()} disabled={busy}>목록 새로고침</button>
            <button className="btn btn-secondary" style={{ padding: "5px 12px", fontSize: 12 }} onClick={() => navigateToTab("tab-dataops")}>DataOps에서 학습데이터 등록</button>
          </span>
        }
      >
        {registerOpen && (
          <ModelPipelineRegisterForm
            datasets={catalog.datasets}
            onCancel={() => setRegisterOpen(false)}
            onRegistered={async (created) => {
              setRegisterOpen(false);
              addConsoleLog(`INFO: 재학습 파이프라인 등록 완료 — ${displayId(created.id)} (${created.name}) · 대상 모델 ${typeName(created.model_type)} · 학습데이터 ${displayId(created.dataset_id)}`);
              await reloadCatalog();
              setFeedback("파이프라인을 등록했습니다.");
            }}
          />
        )}
        {/* 빠른 학습 — 스냅샷 생성 후 이 목록의 파이프라인으로 실행한다(옛 "데이터 학습" 카드를 합침). */}
        <RealdataTrainingPanel
          catalog={catalog}
          demoVisible={demo}
          regionName={regionName}
          busy={locked}
          job={liveJob}
          onReload={reloadCatalog}
          onRunPipeline={runPipeline}
          onClearResult={clearVersionResult}
          onLog={addConsoleLog}
        />
        {catalogError && (
          <p className="pipeline-empty" role="alert">
            <i className="fa-solid fa-circle-exclamation" aria-hidden="true"></i> 파이프라인 카탈로그를 불러오지 못했습니다 — {catalogError}
          </p>
        )}
        {catalogLoaded && catalog.pipelines.length === 0 && !catalogError && (
          <p className="pipeline-empty">
            <i className="fa-solid fa-circle-info" aria-hidden="true"></i> 등록된 재학습 파이프라인이 없습니다.
            [파이프라인 등록]으로 대상 모델과 학습데이터를 지정해 추가하세요.
          </p>
        )}
        <div className="table-container">
          <table>
            <caption className="sr-only">등록된 재학습 파이프라인과 실행 상태</caption>
            <thead>
              <tr>
                <th scope="col">파이프라인</th>
                <th scope="col">대상 모델</th>
                <th scope="col">트리거 조건</th>
                <th scope="col">마지막 실행</th>
                <th scope="col">상태</th>
                <th scope="col" className="cell-actions">동작</th>
              </tr>
            </thead>
            <tbody>
              {pl.pageRows.map((p) => {
                const instance = instanceById.get(p.instance_id);
                const isRunning = catalog.jobs.some((job) => job.instance_id === p.instance_id && (job.state === "queued" || job.state === "running"))
                  || (jobActive && liveJob?.instance_id === p.instance_id);
                const last = lastJobOf(p, catalog.jobs);
                const runLocked = locked || isRunning;
                const runTitle = isRunning
                  ? "이 모델의 학습이 실행 중입니다."
                  : locked
                    ? "다른 재학습 파이프라인이 실행 중입니다. 완료 후 실행할 수 있습니다."
                    : `${p.name} 파이프라인을 즉시 실행합니다`;
                return (
                  <tr key={p.id}>
                    <td>
                      <strong style={{ fontSize: 13 }}>{orDash(p.name)}</strong>
                      <div><code style={{ fontSize: 11, color: "var(--accent-purple-text)" }} title={p.id}>{orDash(displayId(p.id))}</code></div>
                    </td>
                    <td style={{ fontSize: 12 }}>
                      {typeName(p.model_type)} · {regionName(p.region_id)} · {TARGET_LABELS[p.target] ?? p.target}
                      <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
                        활성 <span title={instance?.active_version ?? undefined}>{instance?.active_version ? displayId(instance.active_version) : "없음"}</span> · {orDash(p.experiment)} · 학습데이터 {datasetName(p.dataset_id)}
                      </div>
                    </td>
                    <td style={{ fontSize: 12, color: "var(--text-secondary)" }}>{orDash(p.trigger_policy)}</td>
                    <td style={{ fontSize: 11, color: "var(--text-secondary)" }}>
                      {last ? (
                        <>
                          <button className="pipeline-run-link" onClick={() => setSelectedKey(`modelops:${last.id}`)} title={`${last.id} — 이 실행의 저장된 로그를 아래에서 봅니다`}>
                            {displayId(last.id)}
                          </button>
                          <div style={{ color: "var(--text-muted)" }}>
                            {orDash(last.finished_at ? auditTime(last.finished_at) : null)} · {orDash(JOB_STATE_NAMES[last.state] ?? last.state)}
                          </div>
                        </>
                      ) : "실행 이력 없음"}
                    </td>
                    <td>
                      <span className="system-status" style={{
                        padding: "1px 8px", fontSize: 10, borderColor: "currentColor",
                        color: isRunning ? "var(--accent-orange)" : "var(--accent-teal)",
                        backgroundColor: isRunning ? "rgba(var(--accent-orange-rgb), 0.02)" : "rgba(var(--accent-teal-rgb), 0.02)"
                      }}>
                        {isRunning ? "진행 중" : "대기"}
                      </span>
                    </td>
                    <td className="cell-actions">
                      <button
                        className="btn btn-primary"
                        style={{ padding: "5px 14px", fontSize: 12 }}
                        onClick={(event) => {
                          event.preventDefault();
                          if (runLocked) { addConsoleLog(`WARN: 파이프라인 실행 불가 — ${runTitle}`, false, true); return; }
                          runPipeline(p);
                        }}
                        aria-disabled={runLocked}
                        title={runTitle}
                        aria-label={`${orDash(p.name)} 파이프라인 실행`}
                      >
                        <i className="fa-solid fa-play"></i> 실행
                      </button>
                      <button
                        className="btn btn-secondary"
                        style={{ padding: "5px 12px", fontSize: 12, marginLeft: 6 }}
                        onClick={() => handleDeletePipeline(p)}
                        aria-disabled={locked}
                        title="등록을 해제합니다. 실행 이력은 남습니다."
                        aria-label={`${orDash(p.name)} 파이프라인 등록 해제`}
                      >
                        <i className="fa-solid fa-trash" aria-hidden="true"></i>
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <TablePager page={pl.safePage} totalPages={pl.totalPages} totalCount={catalog.pipelines.length} pageSize={PAGE_SIZE} onChange={setPlPage} />
      </Card>

      {/* ② 실행 상태 — 상시 표시. 새 경로 작업 또는 선택한 이전 경로 기록 */}
      <PipelineStatusCard
        target={statusTarget}
        regionName={regionName}
        onReset={handleReset}
        anchorRef={statusAnchorRef}
        pollError={liveJob && selectedJob?.id === liveJob.id ? pollError : null}
        onRecheck={() => liveJob && pollJob(liveJob.id, 0)}
      />

      {/* 활성화된 버전의 다음 단계 — 시뮬레이터는 활성 버전으로 추론한다. */}
      {activatedVersion && (
        <NextStepBanner
          tone="success"
          // 데모 표시의 시뮬레이터는 예시 값으로 계산한다 — 이 버전으로 추론한다고 말하지 않는다.
          message={demo
            ? `${displayId(activatedVersion.id)} 활성 — 데모 데이터 표시 중에는 정책 시뮬레이터가 예시 값으로 계산합니다`
            : `${displayId(activatedVersion.id)} 활성 — 정책 시뮬레이터가 이 버전으로 추론합니다`}
          actionLabel="정책 시뮬레이터에서 활용"
          targetTab="tab-simulator"
        />
      )}

      {/* ③ Model Store: 새 경로 버전 + (데모 ON) 이전 고정 모델 과거 버전(읽기 전용). 모델을 고르면 그 모델의
          버전만 보이고, 행 끝에서 활성화·이전 버전 복원을 한다(과거 기록 행은 동작 없음). */}
      <Card
        title={<>Model Store<span className="title-tag">모델·실험 버전 이력</span><InfoTip label="모델 승급 기준" text={PROMOTION_RULE} /></>}
        icon="fa-boxes-stacked"
        className="page-section realdata-panel"
        dataSource="api"
      >
        <div className="realdata-controls">
          <label className="realdata-field"><span>모델 선택</span>
            <select value={storeInstance?.id ?? ""} onChange={(event) => { clearVersionResult(); setVersionInstanceId(event.target.value); }}>
              <option value="">전체 모델</option>
              {catalog.instances.map((row) => <option key={row.id} value={row.id}>{instanceLabel(row, regionName)}</option>)}
            </select>
          </label>
        </div>
        {storeInstance && (
          <p className="realdata-hint">
            활성 버전: {storeInstance.active_version ? <code title={storeInstance.active_version}>{displayId(storeInstance.active_version)}</code> : "—"}
            {" "}· 이전 버전: {storeInstance.previous_version ? <code title={storeInstance.previous_version}>{displayId(storeInstance.previous_version)}</code> : "—"}
          </p>
        )}
        {storeResult && <p className={`async-feedback ${storeResult.ok ? "is-success" : "is-error"}`} role={storeResult.ok ? "status" : "alert"}>{storeResult.message}</p>}
        <div className="table-container">
          <table>
            <caption className="sr-only">모델과 실험의 버전 이력 및 운영 상태</caption>
            <thead>
              <tr>
                <th scope="col">모델</th>
                <th scope="col">버전</th>
                <th scope="col">학습데이터</th>
                <th scope="col" className="cell-num">MAE / 기준선 · Accuracy</th>
                <th scope="col" style={{ minWidth: 64 }}>상태</th>
                <th scope="col">등록일</th>
                <th scope="col" className="cell-actions">동작</th>
              </tr>
            </thead>
            <tbody>
              {store.pageRows.length === 0 && (
                <tr>
                  <td colSpan={7} style={{ fontSize: 12, color: "var(--text-muted)", textAlign: "center", padding: "18px 8px" }}>재학습 후 표시됩니다.</td>
                </tr>
              )}
              {store.pageRows.map((m) => {
                if (m.origin === "legacy") {
                  return (
                    <tr key={m.key} data-values-source={m.source}>
                      <td style={{ fontSize: 13 }}>
                        <strong>{MODEL_REGISTRY.find((row) => row.id === m.modelId)?.name ?? m.modelId}</strong>
                        <div><code style={{ fontSize: 11, color: "var(--accent-purple-text)" }}>{orDash(m.modelId)}</code></div>
                        <div style={{ fontSize: 11, color: "var(--text-secondary)" }} data-values-source="mock">{orDash(m.params)}</div>
                      </td>
                      <td><code style={{ fontSize: 12, fontWeight: 600 }} title={m.version}>{orDash(displayId(m.version))}</code></td>
                      <td style={{ fontSize: 12, color: "var(--text-secondary)" }} data-values-source="mock">{orDash(m.dataVersion)}</td>
                      <td className="cell-num" style={{ fontWeight: 600 }}>{typeof m.accuracy === "number" ? `Acc ${m.accuracy.toFixed(3)}` : "–"}</td>
                      <td>
                        <span className="system-status" title={`과거 고정 모델입니다. 읽기 전용이며 새 파이프라인 대상이 아닙니다. 원래 상태: ${orDash(m.originalStatus)}`}
                          style={{ padding: "1px 8px", fontSize: 10, whiteSpace: "nowrap", color: STORE_STATUS.legacy.color, backgroundColor: STORE_STATUS.legacy.bg }}>
                          {STORE_STATUS.legacy.label}
                        </span>
                      </td>
                      <td style={{ fontSize: 11, color: "var(--text-muted)" }} data-values-source="mock">{orDash(m.registeredAt)}</td>
                      <td className="cell-actions"></td>
                    </tr>
                  );
                }
                const st = STORE_STATUS[m.status] ?? STORE_STATUS.candidate;
                const version = versionById.get(m.version);
                return (
                  <tr key={m.key} data-values-source="api" style={m.status === "active" ? { backgroundColor: "rgba(var(--accent-teal-rgb), 0.04)" } : undefined}>
                    <td style={{ fontSize: 13 }}>
                      <strong>{typeName(m.modelType)}</strong>
                      <div style={{ fontSize: 11, color: "var(--text-muted)" }}>{regionName(m.regionId)} · {TARGET_LABELS[m.target] ?? m.target}</div>
                      <div style={{ fontSize: 11, color: "var(--text-secondary)" }}>선형 회귀 · 입력 피처 {m.featureCount}개</div>
                    </td>
                    <td><code style={{ fontSize: 12, fontWeight: 600 }} title={m.version}>{displayId(m.version)}</code></td>
                    <td style={{ fontSize: 12, color: "var(--text-secondary)" }}>
                      {orDash(m.datasetName)}<div style={{ fontSize: 11 }} title={m.datasetHash}>{String(m.datasetHash ?? "").slice(0, 12)}</div>
                    </td>
                    <td className="cell-num" style={{ fontWeight: 600 }}>MAE {metric(m.mae, m.target)} / {metric(m.baselineMae, m.target)}</td>
                    <td><span className="system-status" style={{ padding: "1px 8px", fontSize: 10, whiteSpace: "nowrap", color: st.color, backgroundColor: st.bg }}>{st.label}</span></td>
                    <td style={{ fontSize: 11, color: "var(--text-muted)", whiteSpace: "nowrap" }}>{orDash(day(m.createdAt))}</td>
                    <td className="cell-actions">
                      <span className="store-actions">
                        <button className="btn btn-secondary" disabled={locked || m.status === "active" || !version}
                          onClick={() => activate(m.instanceId, version, false, "store")} aria-label={`${displayId(m.version)} 버전 활성화`}>활성화</button>
                        <button className="btn btn-secondary" disabled={locked || m.status !== "previous" || !version}
                          onClick={() => activate(m.instanceId, version, true, "store")} aria-label={`${displayId(m.version)} 이전 버전 복원`}>이전 버전 복원</button>
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <TablePager {...store.pager} />
      </Card>

      {/* ④ 실행 이력 — 새 경로 학습 작업 + (데모 ON) 이전 경로 과거 기록. 행을 고르면 ②·⑤가 그 실행을 보여준다. */}
      <Card
        title="파이프라인 실행 이력"
        icon="fa-clock-rotate-left"
        className="page-section orchestrator-live"
        dataSource="api"
        headerRight={<span style={{ fontSize: 11, color: "var(--text-muted)" }}>{history.length}건 · 학습 작업 저장소{demo ? " + 과거 기록" : ""}</span>}
      >
        {history.length === 0 ? (
          <p className="pipeline-empty">
            <i className="fa-solid fa-circle-info" aria-hidden="true"></i> 저장된 실행 기록이 없습니다. 위 목록에서 [실행]을 누르면
            실행 레코드·단계 상태·로그가 여기에 남습니다.
          </p>
        ) : (
          <>
            <div className="table-container">
              <table>
                <caption className="sr-only">저장된 재학습 실행 이력</caption>
                <thead>
                  <tr>
                    <th scope="col">실행 ID</th>
                    <th scope="col">파이프라인 / 모델</th>
                    <th scope="col">트리거</th>
                    <th scope="col">결과</th>
                    <th scope="col">시작 → 종료</th>
                    <th scope="col" className="cell-actions">동작</th>
                  </tr>
                </thead>
                <tbody>
                  {runList.pageRows.map((r) => {
                    const version = r.origin === "modelops" ? versionById.get(r.versionId) : null;
                    const instance = r.origin === "modelops" ? instanceById.get(r.instanceId) : null;
                    return (
                      <tr key={r.key} className={r.key === selectedKey ? "is-selected-run" : undefined}>
                        <td>
                          <code style={{ fontSize: 11, color: "var(--accent-purple-text)" }} title={r.id}>{orDash(displayId(r.id))}</code>
                          {r.readOnly && <div><span className="system-status" style={{ padding: "0 6px", fontSize: 10 }}>과거 기록</span></div>}
                        </td>
                        <td style={{ fontSize: 12 }}>
                          <span title={r.pipelineId ?? undefined}>{orDash(displayId(r.pipelineId))}</span>{r.origin === "modelops" && r.pipelineMissing ? " (등록 해제됨)" : ""}
                          <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
                            {r.origin === "modelops"
                              ? `${typeName(r.modelType)}${instance ? ` · ${regionName(instance.region_id)} · ${TARGET_LABELS[instance.target] ?? instance.target}` : ""}`
                              : orDash(r.model)}
                            {r.datasetId ? <span title={r.datasetId}> · {displayId(r.datasetId)}</span> : ""}
                          </div>
                        </td>
                        <td style={{ fontSize: 12, color: "var(--text-secondary)" }}>{orDash(r.trigger)}</td>
                        <td style={{ fontSize: 12 }}>
                          {r.origin === "modelops" ? orDash(JOB_STATE_NAMES[r.state] ?? r.state) : orDash(LEGACY_RUN_STATE_LABEL[r.state] ?? r.state)}
                          <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
                            {r.versionId ? <span title={r.versionId}>버전 {displayId(r.versionId)}</span> : ""}
                            {version ? ` · MAE ${metric(version.metrics?.mae, version.target)} / 기준선 ${metric(version.baseline?.mae, version.target)}` : ""}
                            {r.error ? ` · ${r.error}` : ""}
                          </div>
                        </td>
                        <td style={{ fontSize: 11, color: "var(--text-secondary)" }}>
                          {orDash(r.startedAt ? auditTime(r.startedAt) : null)} → {orDash(r.finishedAt ? auditTime(r.finishedAt) : null)}
                        </td>
                        <td className="cell-actions">
                          <button className="btn btn-secondary" style={{ padding: "4px 12px", fontSize: 12 }} onClick={() => setSelectedKey(r.key)} aria-label={`${displayId(r.id)} 실행 로그 보기`}>
                            로그
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <TablePager page={runList.safePage} totalPages={runList.totalPages} totalCount={history.length} pageSize={PAGE_SIZE} onChange={setRunPage} />
          </>
        )}
      </Card>

      {/* ⑤ 선택한 실행의 로그 — 새 경로는 작업·버전·감사 기록에서, 이전 경로는 저장된 로그 라인에서 */}
      <Card
        title={<>실행 로그{selectedRunId ? <code title={selectedRunId} style={{ marginLeft: 8, fontSize: 12, color: "var(--accent-purple-text)", fontWeight: 500 }}>{displayId(selectedRunId)}</code> : null}</>}
        icon="fa-terminal"
        className="page-section orchestrator-live"
      >
        {logLines.length ? (
          <ConsoleLog logs={logLines.map((entry) => ({ time: logTime(entry.ts), level: entry.level, message: entry.message }))} height={220} />
        ) : (
          <p className="pipeline-empty">
            <i className="fa-solid fa-circle-info" aria-hidden="true"></i>{" "}
            {selectedKey ? "이 실행에 저장된 로그 라인이 없습니다." : "실행 이력에서 [로그]를 누르거나 파이프라인을 실행하면 저장된 로그가 여기에 표시됩니다."}
          </p>
        )}
      </Card>

      <Card title="UI 활동 로그" icon="fa-list" className="page-section">
        <ConsoleLog logs={consoleLogs} />
      </Card>

      {/* ⑥ 최근 운영 기록 — 등록·학습·활성화·복원 감사 기록(버전 활성화·복원은 ③ Model Store에서) */}
      <Card title="최근 운영 기록" icon="fa-clock-rotate-left" className="page-section" dataSource="api">
        {auditLines.length ? <ConsoleLog logs={auditLines} height={220} /> : (
          <p className="pipeline-empty">
            <i className="fa-solid fa-circle-info" aria-hidden="true"></i>{" "}아직 운영 기록이 없습니다.
          </p>
        )}
      </Card>
    </>
  );
}
