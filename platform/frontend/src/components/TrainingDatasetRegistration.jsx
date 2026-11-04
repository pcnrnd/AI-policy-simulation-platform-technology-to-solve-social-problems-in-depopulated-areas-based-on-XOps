import { useCallback, useEffect, useRef, useState } from "react";
import Card from "./Card.jsx";
import ConfirmDialog from "./ConfirmDialog.jsx";
import TablePager, { usePager } from "./TablePager.jsx";
import { apiGet, apiSend } from "../lib/api.js";
import { getRealdataToken } from "./realdata/realdataClient.jsx";
import { useAppState } from "../context/AppStateContext.jsx";
import { alertText } from "../lib/alertText.js";
import { dataModeEpoch } from "../lib/dataMode.js";
import { catalogRegistrationBody, catalogSelectionKey } from "../lib/trainingDatasets.js";
import { formatDateTime } from "../lib/reportContent.js";

const PURPOSES = { training: "학습", validation: "검증", test: "평가" };

export default function TrainingDatasetRegistration({ selectedSource = null, filter = "", sort = "" }) {
  const { mockDataVisible, addConsoleLog } = useAppState();
  const mode = mockDataVisible ? "mock" : "real";
  const key = catalogSelectionKey(selectedSource, filter, sort, mode);
  const currentKey = useRef(key);
  currentKey.current = key;
  const [available, setAvailable] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [name, setName] = useState("");
  const [purpose, setPurpose] = useState("training");
  const [prepared, setPrepared] = useState(null);
  const [busy, setBusy] = useState(false);
  const [feedback, setFeedback] = useState(null);
  const [deleteTarget, setDeleteTarget] = useState(null);
  const [detail, setDetail] = useState(null);
  const datasetList = usePager(datasets);
  const request = useRef(0);
  const operation = useRef(0);
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
    setAvailable([]); setDatasets([]); setFeedback(null); setDeleteTarget(null); setDetail(null);
    setBusy(false); busyRef.current = false; operation.current += 1;
    load().catch((error) => { if (active) setFeedback(alertText(error, "학습데이터 목록을 다시 불러오세요.")); });
    return () => { active = false; request.current += 1; operation.current += 1; };
  }, [load]);
  useEffect(() => { setPrepared(null); setFeedback(null); }, [key]);
  const ready = prepared?.key === key ? prepared.data : null;
  const sourceName = (row) => row.source_name ?? available.find((item) => item.id === row.source_dataset_id)?.name ?? "원본 목록에 없음";
  const begin = () => {
    if (busyRef.current) return null;
    busyRef.current = true; setBusy(true); setFeedback(null);
    const sequence = ++operation.current;
    const epoch = dataModeEpoch();
    return { selection: () => sequence === operation.current && epoch === dataModeEpoch() && key === currentKey.current,
      mode: () => sequence === operation.current && epoch === dataModeEpoch(),
      finish: () => { if (sequence === operation.current) { busyRef.current = false; setBusy(false); } } };
  };
  const prepare = async () => {
    if (!selectedSource) return;
    const action = begin();
    if (!action) return;
    setPrepared(null);
    try {
      const token = await getRealdataToken();
      if (!action.selection()) return;
      const data = await apiSend("POST", "/api/v3/modelops/datasets/prepare", { token,
        body: { source_id: selectedSource.id, filter: filter.trim(), sort } });
      if (!action.selection()) return;
      setPrepared({ key, data });
      setFeedback(`조건에 맞는 전체 ${data.row_count.toLocaleString()}행을 확인했습니다. 이름과 용도를 입력해 등록하세요.`);
    } catch (error) {
      if (action.selection()) setFeedback(alertText(error, "원본 자료를 준비하지 못했습니다. 원본과 조회 조건을 확인하세요."));
    } finally { action.finish(); }
  };
  const submit = async (event) => {
    event.preventDefault();
    const body = catalogRegistrationBody(selectedSource, filter, sort, prepared, name, purpose, mode);
    if (!body || !name.trim()) return;
    const action = begin();
    if (!action) return;
    try {
      const token = await getRealdataToken();
      if (!action.selection()) return;
      const registered = await apiSend("POST", "/api/v3/modelops/datasets/from-catalog", { token, body });
      if (!action.mode()) return;
      const reused = datasets.some((row) => row.id === registered.id);
      await load();
      if (!action.selection()) return;
      setPrepared(null); setName("");
      setFeedback(reused ? `같은 원본·조건·내용·용도의 학습데이터가 이미 등록되어 있습니다(${registered.name}).`
        : "선택한 원본의 전체 조건 결과를 학습 자료로 등록했습니다. 모델 학습에는 대상·지역·기간 매핑 확인이 필요합니다.");
    } catch (error) {
      if (!action.selection()) return;
      setPrepared(null);
      setFeedback(alertText(error, "학습데이터를 등록하지 못했습니다. 현재 선택으로 다시 준비하세요."));
      addConsoleLog("WARN: 학습데이터 등록 실패 — 현재 원본과 조회 조건을 확인하세요.", false, true);
    } finally { action.finish(); }
  };
  const remove = async () => {
    if (!deleteTarget) return;
    const action = begin();
    if (!action) return;
    try {
      const token = await getRealdataToken();
      if (!action.mode()) return;
      const result = await apiSend("DELETE", `/api/v3/modelops/datasets/${encodeURIComponent(deleteTarget.id)}`, { token });
      if (!action.mode()) return;
      await load();
      if (!action.mode()) return;
      setDeleteTarget(null); setDetail(null);
      setFeedback(result.managed_copy_removed ? "학습데이터 등록본을 삭제했습니다. 원본 데이터는 유지됩니다."
        : "등록본을 삭제했습니다. 관리 복사본 정리는 운영 확인이 필요합니다. 원본 데이터는 유지됩니다.");
    } catch (error) {
      if (action.mode()) { setDeleteTarget(null); setFeedback(alertText(error, "학습데이터를 삭제하지 못했습니다. 사용 중인지 확인하세요.")); }
    } finally { action.finish(); }
  };
  const showDetail = async (row) => {
    const action = begin();
    if (!action) return;
    try {
      const token = await getRealdataToken();
      if (!action.mode()) return;
      const data = await apiGet(`/api/v3/modelops/datasets/${encodeURIComponent(row.id)}`, { token });
      if (action.mode()) setDetail(data);
    } catch (error) {
      if (action.mode()) setFeedback(alertText(error, "등록 상세를 불러오지 못했습니다."));
    } finally { action.finish(); }
  };
  return <Card title="학습데이터 등록" icon="fa-database" dataSource="api">
    <form className="pipeline-register" onSubmit={submit} aria-label="학습데이터 등록">
      <div className="pipeline-register-grid">
        <label><span>STEP ①에서 선택한 원본</span><input value={selectedSource?.label ?? "선택한 원본이 없습니다"} readOnly /></label>
        <label><span>학습데이터 이름</span><input value={name} onChange={(event) => setName(event.target.value)} maxLength={120} required disabled={busy} /></label>
        <label><span>용도</span><select value={purpose} onChange={(event) => setPurpose(event.target.value)} disabled={busy}>
          {Object.entries(PURPOSES).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
        </select></label>
      </div>
      <p>STEP ③ 조회 조건: {filter.trim() || "추가 조건 없음"}{sort ? ` · ${sort} 내림차순` : ""}</p>
      {selectedSource?.range && <p>원본 조회 범위: {selectedSource.range.column} {selectedSource.range.from}~{selectedSource.range.to}</p>}
      <p>화면의 미리보기 페이지 크기와 관계없이 원본 조회 범위와 조건에 맞는 전체 행을 등록합니다. 조건을 바꾸면 다시 준비하세요.</p>
      {!selectedSource && <p role="status">STEP ① 카탈로그에서 등록할 원본을 먼저 선택하세요.</p>}
      {ready && <details className="training-registration-detail" open><summary>준비 완료 · 전체 {ready.row_count.toLocaleString()}행 · {ready.schema.length}개 컬럼</summary>
        <p>내용 해시: <code>{ready.content_hash}</code></p>
        <div className="table-container"><table><caption>준비한 자료 미리보기 {ready.preview_count}행 (전체 {ready.row_count}행)</caption>
          <thead><tr>{ready.schema.map((column) => <th key={column.name}>{column.name}</th>)}</tr></thead>
          <tbody>{ready.preview_rows.map((row, index) => <tr key={index}>{ready.schema.map((column) => <td key={column.name}>{row[column.name] === null ? "—" : String(row[column.name])}</td>)}</tr>)}</tbody>
        </table></div>
      </details>}
      <div className="pipeline-register-actions">
        <button className="btn btn-secondary" type="button" onClick={prepare} disabled={busy || !selectedSource}>선택 원본 준비</button>
        <button className="btn btn-primary" type="submit" disabled={busy || !ready || !name.trim()}>{busy ? "처리 중…" : "학습데이터 등록"}</button>
        <button className="btn btn-secondary" type="button" onClick={() => {
          const epoch = dataModeEpoch();
          load().catch(() => { if (epoch === dataModeEpoch()) setFeedback("목록을 다시 불러오세요."); });
        }} disabled={busy}>목록 새로고침</button>
      </div>
    </form>
    {feedback && <p role="status">{feedback}</p>}
    <div className="table-container"><table className="training-dataset-table">
      <caption className="sr-only">등록된 학습데이터 목록</caption>
      <colgroup>{[18,18,7,8,15,9,11,14].map((width,index) => <col key={index} style={{width:`${width}%`}} />)}</colgroup>
      <thead><tr><th>이름</th><th>원본 데이터</th><th>용도</th><th>행 수</th><th>등록일</th><th>상태</th><th>내용 해시</th><th>관리</th></tr></thead>
      <tbody>{datasetList.pageRows.map((row) => <tr key={row.id}><td><span title={row.name}>{row.name}</span></td><td><span title={`${row.id} · 원본 ${row.source_dataset_id ?? "—"}`}>{sourceName(row)}</span></td>
        <td>{PURPOSES[row.purpose] ?? "학습"}</td><td>{row.row_count.toLocaleString()}</td><td><span title={formatDateTime(row.created_at)}>{formatDateTime(row.created_at)}</span></td>
        <td>등록 완료</td><td><span title={row.content_hash}>{row.content_hash.slice(0, 12)}</span></td>
        <td className="training-dataset-actions"><button type="button" className="btn btn-secondary" disabled={busy} aria-label={`${row.name} 등록 상세`} onClick={() => showDetail(row)}>상세</button>{" "}
          <button type="button" className="btn btn-secondary" disabled={busy} aria-label={`${row.name} 등록본 삭제`} onClick={() => setDeleteTarget(row)}>삭제</button></td></tr>)}
        {!datasets.length && <tr><td colSpan={8}>등록된 학습데이터가 없습니다. 카탈로그 원본을 선택해 준비한 뒤 등록하세요.</td></tr>}</tbody>
    </table></div>
    <TablePager {...datasetList.pager} />
    {detail && <details className="training-registration-detail" open><summary>{detail.name} 등록 상세</summary>
      <p>원본: {sourceName(detail)} · 용도: {PURPOSES[detail.purpose] ?? "학습"} · 전체 {detail.row_count}행 · 등록일: {formatDateTime(detail.created_at)}</p>
      <p>내용 해시: <code>{detail.content_hash}</code></p>
      {detail.query && <p>조건: {detail.query.filter || "추가 조건 없음"} · 정렬: {detail.query.sort || "없음"} · 범위: {detail.query.range ? `${detail.query.range.column} ${detail.query.range.from}~${detail.query.range.to}` : "전체"}</p>}
      {detail.schema && <p>컬럼: {detail.schema.map((column) => `${column.name} (${column.type})`).join(", ")} · 자료 버전: {detail.schema_version}</p>}
      {detail.training_compatible === false && <p>표형 자료 등록 완료 · 모델 학습에는 대상·지역·기간 매핑 확인이 필요합니다.</p>}
    </details>}
    <ConfirmDialog open={Boolean(deleteTarget)} title="학습데이터 등록본을 삭제할까요?"
      description={`${deleteTarget?.name ?? ""} 등록본과 관리 복사본만 삭제합니다. 원본은 유지되며 파이프라인·버전·작업에서 참조 중이면 삭제할 수 없습니다.`}
      confirmLabel="등록본 삭제" busy={busy} onConfirm={remove} onCancel={() => setDeleteTarget(null)} />
  </Card>;
}
