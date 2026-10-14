import { useCallback, useEffect, useState } from "react";
import Card from "./Card.jsx";
import { apiGet, apiSend } from "../lib/api.js";
import { getRealdataToken } from "./realdata/realdataClient.jsx";
import { useAppState } from "../context/AppStateContext.jsx";
import { datasetRegionName } from "../lib/modelops.js";

const TARGET_NAMES = { nonlocal_visitors: "타지역 방문객", observed_sales_krw: "관측 소비", population_change_pct: "인구 변화율" };

export default function TrainingDatasetRegistration() {
  const { appData } = useAppState();
  const [available, setAvailable] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [sourceId, setSourceId] = useState("");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [feedback, setFeedback] = useState(null);
  const load = useCallback(async () => {
    const token = await getRealdataToken();
    const [sources, registered] = await Promise.all([
      apiGet("/api/v3/modelops/available-datasets", { token }),
      apiGet("/api/v3/modelops/datasets", { token })
    ]);
    setAvailable(sources.data ?? []);
    setDatasets(registered.data ?? []);
  }, []);
  useEffect(() => {
    let active = true;
    load().catch(() => { if (active) setFeedback("학습데이터 목록을 다시 불러오세요."); });
    return () => { active = false; };
  }, [load]);
  const source = available.find((row) => row.id === sourceId);
  const submit = async (event) => {
    event.preventDefault();
    if (busy || !source || !name.trim()) return;
    setBusy(true);
    setFeedback(null);
    try {
      const token = await getRealdataToken();
      await apiSend("POST", "/api/v3/modelops/datasets", { token, body: {
        name: name.trim(), source_dataset_id: source.id, source_hash: source.content_hash
      } });
      await load();
      setName("");
      setFeedback("학습데이터를 등록했습니다. 오케스트레이터에서 파이프라인을 연결하세요.");
    } catch (error) { setFeedback(error.message || "학습데이터를 확인한 뒤 다시 등록하세요."); }
    finally { setBusy(false); }
  };
  return <Card title="학습데이터 등록" icon="fa-database" dataSource="api">
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
        <button className="btn btn-secondary" type="button" onClick={() => load().catch(() => setFeedback("목록을 다시 불러오세요."))} disabled={busy}>목록 새로고침</button>
      </div>
    </form>
    {feedback && <p role="status">{feedback}</p>}
    <div className="table-container"><table className="training-dataset-table">
      <caption className="sr-only">등록된 학습데이터 목록</caption>
      <colgroup>{[20,20,16,13,14,8,9].map((width,index) => <col key={index} style={{width:`${width}%`}} />)}</colgroup>
      <thead><tr><th>이름</th><th>데이터셋</th><th>지역</th><th>학습 대상</th><th>기간</th><th>행 수</th><th>내용 해시</th></tr></thead>
      <tbody>{datasets.map((row) => <tr key={row.id}><td><span title={row.name}>{row.name}</span></td><td><span title={row.id}>{row.id}</span></td>
        <td><span title={`${datasetRegionName(row, appData.regions)} (${row.region_id})`}>{datasetRegionName(row, appData.regions)}</span></td><td><span title={TARGET_NAMES[row.target] ?? row.target}>{TARGET_NAMES[row.target] ?? row.target}</span></td>
        <td><span title={`${row.observed_from}~${row.observed_to}`}>{row.observed_from}~{row.observed_to}</span></td><td><span title={row.row_count.toLocaleString()}>{row.row_count.toLocaleString()}</span></td><td><span title={row.content_hash}>{row.content_hash.slice(0, 12)}</span></td></tr>)}</tbody>
    </table></div>
  </Card>;
}
