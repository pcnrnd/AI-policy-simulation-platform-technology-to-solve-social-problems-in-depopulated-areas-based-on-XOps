import { useCallback, useEffect, useRef, useState } from "react";
import Card from "./Card.jsx";
import ConfirmDialog from "./ConfirmDialog.jsx";
import { apiGet, apiSend } from "../lib/api.js";
import { getRealdataToken } from "./realdata/realdataClient.jsx";
import { useAppState } from "../context/AppStateContext.jsx";
import { alertText } from "../lib/alertText.js";
import { dataModeEpoch } from "../lib/dataMode.js";
import { TARGET_LABELS as TARGET_NAMES, datasetRegionName } from "../lib/modelops.js";


export default function TrainingDatasetRegistration() {
  const { appData, mockDataVisible, addConsoleLog } = useAppState();
  const [available, setAvailable] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [sourceId, setSourceId] = useState("");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [feedback, setFeedback] = useState(null);
  const [deleteTarget, setDeleteTarget] = useState(null);
  const request = useRef(0);
  const busyRef = useRef(false);
  const load = useCallback(async () => {
    const sequence = ++request.current;
    const epoch = dataModeEpoch();
    const token = await getRealdataToken();
    if (epoch !== dataModeEpoch()) return;
    const [sources, registered] = await Promise.all([
      apiGet("/api/v3/modelops/available-datasets", { token }),
      apiGet("/api/v3/modelops/datasets", { token })
    ]);
    if (sequence !== request.current || epoch !== dataModeEpoch()) return;
    setAvailable(sources.data ?? []);
    setDatasets(registered.data ?? []);
  }, [mockDataVisible]);
  useEffect(() => {
    let active = true;
    setAvailable([]); setDatasets([]); setSourceId(""); setFeedback(null); setDeleteTarget(null);
    setBusy(false); busyRef.current = false;
    load().catch(() => { if (active) setFeedback("학습데이터 목록을 다시 불러오세요."); });
    return () => { active = false; request.current += 1; };
  }, [load]);
  const source = available.find((row) => row.id === sourceId);
  const submit = async (event) => {
    event.preventDefault();
    if (busyRef.current || !source || !name.trim()) return;
    const epoch = dataModeEpoch();
    busyRef.current = true;
    setBusy(true);
    setFeedback(null);
    try {
      const token = await getRealdataToken();
      if (epoch !== dataModeEpoch()) return;
      const registered = await apiSend("POST", "/api/v3/modelops/datasets", { token, body: {
        name: name.trim(), source_dataset_id: source.id, source_hash: source.content_hash
      } });
      if (epoch !== dataModeEpoch()) return;
      // 같은 원본·같은 내용은 서버가 기존 등록본을 돌려준다 — 새로 만든 것처럼 안내하지 않는다.
      const reused = datasets.some((row) => row.id === registered.id);
      await load();
      if (epoch !== dataModeEpoch()) return;
      setName("");
      setFeedback(reused
        ? `같은 내용의 학습데이터가 이미 등록되어 있습니다(${registered.name}). 오케스트레이터에서 그 학습데이터로 파이프라인을 연결하세요.`
        : "학습데이터를 등록했습니다. 오케스트레이터에서 파이프라인을 연결하세요.");
    } catch (error) {
      if (epoch !== dataModeEpoch()) return;
      setFeedback(alertText(error, "학습데이터를 등록하지 못했습니다. 확인한 뒤 다시 등록하세요."));
      addConsoleLog(`WARN: 학습데이터 등록 실패 — ${error?.message ?? error}`, false, true);
    }
    finally { if (epoch === dataModeEpoch()) { setBusy(false); busyRef.current = false; } }
  };
  const remove = async () => {
    if (!deleteTarget || busyRef.current) return;
    const epoch = dataModeEpoch();
    busyRef.current = true; setBusy(true); setFeedback(null);
    try {
      const token = await getRealdataToken();
      if (epoch !== dataModeEpoch()) return;
      const result = await apiSend("DELETE", `/api/v3/modelops/datasets/${encodeURIComponent(deleteTarget.id)}`, { token });
      if (epoch !== dataModeEpoch()) return;
      await load();
      if (epoch !== dataModeEpoch()) return;
      setDeleteTarget(null);
      setFeedback(result.managed_copy_removed ? "학습데이터 등록본을 삭제했습니다. 원본 데이터는 유지됩니다."
        : "등록본을 삭제했습니다. 관리 복사본 정리는 운영 확인이 필요합니다. 원본 데이터는 유지됩니다.");
    } catch (error) {
      if (epoch !== dataModeEpoch()) return;
      setDeleteTarget(null);
      setFeedback(alertText(error, "학습데이터를 삭제하지 못했습니다. 사용 중인지 확인하세요."));
    } finally {
      if (epoch === dataModeEpoch()) { busyRef.current = false; setBusy(false); }
    }
  };
  // 학습 자산은 모드별로 분리돼 데모 표시의 목록은 전부 예시다("데모 포함"이 아니라 "데모").
  return <Card title="학습데이터 등록" icon="fa-database" dataSource="api" sourceLabel={mockDataVisible ? "데모" : undefined}>
    <form className="pipeline-register" onSubmit={submit} aria-label="학습데이터 등록">
      <div className="pipeline-register-grid">
        <label><span>원본 데이터 선택</span><select value={sourceId} onChange={(event) => setSourceId(event.target.value)} required>
          <option value="">선택하세요</option>
          {available.map((row) => <option key={row.id} value={row.id}>{row.name}</option>)}
        </select></label>
        <label><span>학습데이터 이름</span><input value={name} onChange={(event) => setName(event.target.value)} maxLength={120} required /></label>
      </div>
      {source && <p>학습 대상: {TARGET_NAMES[source.target] ?? source.target} · {source.observed_from}~{source.observed_to}</p>}
      {!available.length && <p role="status">사용할 데이터를 확인한 뒤 목록을 새로고침하세요.</p>}
      <div className="pipeline-register-actions">
        <button className="btn btn-primary" type="submit" disabled={busy || !source || !name.trim()}>{busy ? "등록 중…" : "학습데이터 등록"}</button>
        <button className="btn btn-secondary" type="button" onClick={() => {
          const epoch = dataModeEpoch();
          load().catch(() => { if (epoch === dataModeEpoch()) setFeedback("목록을 다시 불러오세요."); });
        }} disabled={busy}>목록 새로고침</button>
      </div>
    </form>
    {feedback && <p role="status">{feedback}</p>}
    <div className="table-container"><table className="training-dataset-table">
      <caption className="sr-only">등록된 학습데이터 목록</caption>
      <colgroup>{[18,18,13,12,13,8,10,8].map((width,index) => <col key={index} style={{width:`${width}%`}} />)}</colgroup>
      <thead><tr><th>이름</th><th>데이터셋</th><th>지역</th><th>학습 대상</th><th>기간</th><th>행 수</th><th>내용 해시</th><th>관리</th></tr></thead>
      <tbody>{datasets.map((row) => <tr key={row.id}><td><span title={row.name}>{row.name}</span></td><td><span title={row.id}>{row.id}</span></td>
        <td><span title={`${datasetRegionName(row, appData.regions)} (${row.region_id})`}>{datasetRegionName(row, appData.regions)}</span></td><td><span title={TARGET_NAMES[row.target] ?? row.target}>{TARGET_NAMES[row.target] ?? row.target}</span></td>
        <td><span title={`${row.observed_from}~${row.observed_to}`}>{row.observed_from}~{row.observed_to}</span></td><td><span title={row.row_count.toLocaleString()}>{row.row_count.toLocaleString()}</span></td><td><span title={row.content_hash}>{row.content_hash.slice(0, 12)}</span></td>
        <td><button type="button" className="btn btn-secondary" disabled={busy} aria-label={`${row.name} 등록본 삭제`} onClick={() => setDeleteTarget(row)}>삭제</button></td></tr>)}</tbody>
    </table></div>
    <ConfirmDialog open={Boolean(deleteTarget)} title="학습데이터 등록본을 삭제할까요?"
      description={`${deleteTarget?.name ?? ""} 등록본과 관리 복사본만 삭제합니다. 원본은 유지되며 파이프라인·버전·작업에서 참조 중이면 삭제할 수 없습니다.`}
      confirmLabel="등록본 삭제" busy={busy} onConfirm={remove} onCancel={() => setDeleteTarget(null)} />
  </Card>;
}
