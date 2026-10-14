import { useCallback, useEffect, useRef, useState } from "react";
import Card from "../components/Card.jsx";
import ModelPipelineRegisterForm from "../components/ModelPipelineRegisterForm.jsx";
import { getRealdataToken } from "../components/realdata/realdataClient.jsx";
import { useAppState } from "../context/AppStateContext.jsx";
import { apiGet, apiSend } from "../lib/api.js";
import { MODEL_TYPES, TARGET_LABELS, versionStatus } from "../lib/modelops.js";

const EMPTY = { instances: [], datasets: [], pipelines: [], jobs: [], audit: [] };
const ACTION_NAMES = { dataset_registered: "학습데이터 등록", instance_registered: "모델 등록", pipeline_registered: "파이프라인 등록", training_saved: "학습 완료", training_failed: "학습 실패", version_activated: "버전 활성화", version_rolled_back: "이전 버전 복원", pipeline_deleted: "파이프라인 삭제" };
const STATE_NAMES = { queued: "대기", running: "학습 중", saved: "등록 완료", failed: "실패" };
const metric = (value) => Number.isFinite(value) ? value.toLocaleString("ko-KR", { maximumFractionDigits: 3 }) : "—";

export default function ModelOpsPage() {
  const { appData, navigateToTab } = useAppState();
  const [catalog, setCatalog] = useState(EMPTY);
  const [selectedType, setSelectedType] = useState("factor_analysis");
  const [instanceId, setInstanceId] = useState("");
  const [versions, setVersions] = useState([]);
  const [registerOpen, setRegisterOpen] = useState(false);
  const [feedback, setFeedback] = useState(null);
  const [busy, setBusy] = useState(false);
  const [job, setJob] = useState(null);
  const mounted = useRef(true);
  const catalogRequest = useRef(0);
  const versionRequest = useRef(0);
  const jobTimer = useRef(null);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; clearTimeout(jobTimer.current); }; }, []);
  const load = useCallback(async () => {
    const request = ++catalogRequest.current;
    const token = await getRealdataToken();
    const response = await apiGet("/api/v3/modelops/catalog", { token });
    if (mounted.current && request === catalogRequest.current) setCatalog(response);
  }, []);
  const loadVersions = useCallback(async (id) => {
    const request = ++versionRequest.current;
    setVersions([]);
    if (!id) { setVersions([]); return; }
    const token = await getRealdataToken();
    const response = await apiGet(`/api/v3/modelops/instances/${encodeURIComponent(id)}/versions`, { token });
    if (mounted.current && request === versionRequest.current) setVersions(response.data ?? []);
  }, []);
  useEffect(() => { load().catch(() => { if (mounted.current) setFeedback("모델 목록을 다시 불러오세요."); }); }, [load]);
  const modelInstances = catalog.instances.filter((row) => row.model_type === selectedType);
  const selectedInstance = modelInstances.find((row) => row.id === instanceId) ?? modelInstances[0];
  const selectedId = selectedInstance?.id ?? "";
  useEffect(() => { loadVersions(selectedId).catch(() => { if (mounted.current) setFeedback("버전 목록을 다시 불러오세요."); }); }, [selectedId, loadVersions]);
  const regionName = (id) => appData.regions.find((row) => row.id === id)?.name ?? id;
  const run = async (pipeline) => {
    if (busy) return;
    setBusy(true);
    setFeedback(null);
    try {
      const token = await getRealdataToken();
      const accepted = await apiSend("POST", `/api/v3/modelops/pipelines/${encodeURIComponent(pipeline.id)}/run`, { token });
      if (!mounted.current) return;
      setJob(accepted);
      const poll = async () => {
        try {
          const latest = await apiGet(`/api/v3/modelops/jobs/${encodeURIComponent(accepted.id)}`, { token });
          if (!mounted.current) return;
          setJob(latest);
          if (latest.state === "saved" || latest.state === "failed") {
            setBusy(false);
            setFeedback(latest.state === "saved" ? "학습 결과를 등록했습니다. 버전을 확인한 뒤 활성화하세요." : "학습데이터와 관측 기간을 확인한 뒤 다시 실행하세요.");
            await load();
            setSelectedType(pipeline.model_type);
            setInstanceId(pipeline.instance_id);
            await loadVersions(pipeline.instance_id);
          } else jobTimer.current = setTimeout(poll, 1500);
        } catch (error) { if (mounted.current) { setBusy(false); setFeedback(error.message || "학습 상태를 다시 확인하세요."); } }
      };
      jobTimer.current = setTimeout(poll, 500);
    } catch (error) { if (mounted.current) { setBusy(false); setFeedback(error.message || "학습데이터를 확인하세요."); } }
  };
  const activate = async (version, rollback) => {
    if (busy) return;
    setBusy(true);
    try {
      const token = await getRealdataToken();
      await apiSend("POST", `/api/v3/modelops/instances/${encodeURIComponent(selectedId)}/versions/${encodeURIComponent(version.id)}/${rollback ? "rollback" : "activate"}`, { token });
      if (!mounted.current) return;
      await load();
      await loadVersions(selectedId);
      setFeedback(rollback ? "이전 버전을 복원했습니다." : "선택한 버전을 활성화했습니다.");
    } catch (error) { if (mounted.current) setFeedback(error.message || "학습 결과를 확인하세요."); }
    finally { if (mounted.current) setBusy(false); }
  };
  const remove = async (pipeline) => {
    if (busy || !window.confirm(`${pipeline.name} 파이프라인을 삭제할까요?`)) return;
    setBusy(true);
    try {
      const token = await getRealdataToken();
      await apiSend("DELETE", `/api/v3/modelops/pipelines/${encodeURIComponent(pipeline.id)}`, { token });
      await load();
      if (mounted.current) setFeedback("파이프라인을 삭제했습니다. 학습 이력은 유지됩니다.");
    } catch (error) { if (mounted.current) setFeedback(error.message || "목록을 다시 확인하세요."); }
    finally { if (mounted.current) setBusy(false); }
  };
  return <>
    <div className="grid-cols-2" role="group" aria-label="모델 종류">
      {MODEL_TYPES.map((type) => <Card key={type.id} title={type.name} icon={type.id === "factor_analysis" ? "fa-magnifying-glass-chart" : "fa-chart-line"} dataSource="api">
        <p>등록 모델 {catalog.instances.filter((row) => row.model_type === type.id).length}개</p>
        <button className="btn btn-secondary" aria-pressed={selectedType === type.id} onClick={() => { setSelectedType(type.id); setInstanceId(""); }}>{type.name} 버전 보기</button>
      </Card>)}
    </div>
    {feedback && <p className="async-feedback" role="status">{feedback}</p>}
    <Card title="모델 버전 관리" icon="fa-code-branch" dataSource="api">
      <label className="realdata-field"><span>등록 모델 선택</span><select value={selectedId} onChange={(event) => setInstanceId(event.target.value)}>
        {!modelInstances.length && <option value="">등록된 모델 없음</option>}
        {modelInstances.map((row) => <option key={row.id} value={row.id}>{regionName(row.region_id)} · {TARGET_LABELS[row.target] ?? row.target}</option>)}
      </select></label>
      {!selectedId && <p>학습데이터를 등록하고 파이프라인을 연결해 시작하세요.</p>}
      {selectedInstance && <p>활성 버전: {selectedInstance.active_version ?? "—"} · 이전 버전: {selectedInstance.previous_version ?? "—"}</p>}
      <div className="table-container"><table><caption className="sr-only">선택한 모델의 학습 버전과 상태</caption>
        <thead><tr><th>버전</th><th>학습데이터·해시</th><th>지역·대상·기간</th><th>MAE / 기준선</th><th>상태</th><th>동작</th></tr></thead>
        <tbody>{versions.map((row) => <tr key={row.id}><td>{row.id}<div>학습: {row.job_id}</div></td>
          <td>{row.dataset_id}<div title={row.dataset_hash}>{row.dataset_hash.slice(0, 12)}</div></td>
          <td>{regionName(row.region_id)} · {TARGET_LABELS[row.target] ?? row.target}<div>{row.observed_from}~{row.observed_to}</div></td>
          <td>{metric(row.metrics.mae)} / {metric(row.baseline.mae)}</td><td>{versionStatus(row.status)}</td>
          <td><button className="btn btn-secondary" disabled={busy || row.status === "active"} onClick={() => activate(row, false)}>활성화</button>{" "}
            <button className="btn btn-secondary" disabled={busy || row.status !== "previous"} onClick={() => activate(row, true)}>이전 버전 복원</button></td>
        </tr>)}</tbody>
      </table></div>
    </Card>
    <Card title="학습 파이프라인" icon="fa-diagram-project" dataSource="api">
      <div className="pipeline-register-actions">
        <button className="btn btn-primary" onClick={() => setRegisterOpen(!registerOpen)} disabled={busy}>파이프라인 등록</button>
        <button className="btn btn-secondary" onClick={() => load().catch(() => setFeedback("목록을 다시 불러오세요."))} disabled={busy}>목록 새로고침</button>
        <button className="btn btn-secondary" onClick={() => navigateToTab("tab-dataops")}>DataOps에서 학습데이터 등록</button>
      </div>
      {registerOpen && <ModelPipelineRegisterForm datasets={catalog.datasets} onCancel={() => setRegisterOpen(false)} onRegistered={async () => { setRegisterOpen(false); await load(); setFeedback("파이프라인을 등록했습니다."); }} />}
      <div className="table-container"><table><caption className="sr-only">등록된 학습 파이프라인</caption>
        <thead><tr><th>파이프라인</th><th>모델 종류</th><th>학습데이터</th><th>트리거</th><th>동작</th></tr></thead>
        <tbody>{catalog.pipelines.map((row) => <tr key={row.id}><td>{row.name}<div>{row.id}</div></td><td>{MODEL_TYPES.find((type) => type.id === row.model_type)?.name}</td>
          <td>{catalog.datasets.find((ds) => ds.id === row.dataset_id)?.name ?? row.dataset_id}<div title={row.dataset_hash}>{row.dataset_hash.slice(0, 12)}</div></td>
          <td>{row.trigger_policy}</td><td><button className="btn btn-primary" disabled={busy} aria-label={`${row.name} 실행`} onClick={() => run(row)}>실행</button>{" "}<button className="btn btn-secondary" disabled={busy} aria-label={`${row.name} 삭제`} onClick={() => remove(row)}>삭제</button></td></tr>)}</tbody>
      </table></div>
      {job && <p role="status">학습 {job.id} — {STATE_NAMES[job.state]}</p>}
    </Card>
    <Card title="학습·감사 이력" icon="fa-clock-rotate-left" dataSource="api">
      <div className="table-container"><table><caption className="sr-only">학습 작업 이력</caption><thead><tr><th>학습</th><th>파이프라인</th><th>버전</th><th>상태</th></tr></thead>
        <tbody>{catalog.jobs.map((row) => <tr key={row.id}><td>{row.id}</td><td>{row.pipeline_id}</td><td>{row.version_id}</td><td>{STATE_NAMES[row.state]}</td></tr>)}</tbody></table></div>
      <ul>{catalog.audit.slice(0, 20).map((row) => <li key={row.id}>{ACTION_NAMES[row.action] ?? "운영 기록"} · {new Date(row.created_at).toLocaleString("ko-KR")} · {row.asset_id}</li>)}</ul>
    </Card>
  </>;
}
