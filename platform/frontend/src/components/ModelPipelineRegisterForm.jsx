import { useState } from "react";
import { apiSend } from "../lib/api.js";
import { MODEL_TYPES, TARGET_LABELS, pipelineDefinition } from "../lib/modelops.js";
import { getRealdataToken } from "./realdata/realdataClient.jsx";

const EMPTY = { id: "", name: "", model_type: "factor_analysis", dataset_id: "", trigger_policy: "수동" };
export default function ModelPipelineRegisterForm({ datasets = [], onRegistered, onCancel }) {
  const [form, setForm] = useState(EMPTY);
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const set = (key) => (event) => setForm((previous) => ({ ...previous, [key]: event.target.value }));
  const handleSubmit = async (event) => {
    event.preventDefault();
    if (submitting) return;
    setError(null);
    try {
      const body = pipelineDefinition(form, datasets);
      setSubmitting(true);
      const token = await getRealdataToken();
      const created = await apiSend("POST", "/api/v3/modelops/pipelines", { token, body });
      onRegistered?.(created);
    } catch (err) { setError(err.message || "입력한 내용을 확인한 뒤 다시 등록하세요."); }
    finally { setSubmitting(false); }
  };
  return <form className="pipeline-register" onSubmit={handleSubmit} aria-label="파이프라인 등록">
    <div className="pipeline-register-grid">
      <label><span>파이프라인 ID</span><input value={form.id} onChange={set("id")} required maxLength={128} /></label>
      <label><span>파이프라인 이름</span><input value={form.name} onChange={set("name")} required maxLength={120} /></label>
      <label><span>모델 종류</span><select value={form.model_type} onChange={set("model_type")}>{MODEL_TYPES.map((type) => <option key={type.id} value={type.id}>{type.name}</option>)}</select></label>
      <label><span>등록 학습데이터</span><select value={form.dataset_id} onChange={set("dataset_id")} required>
        <option value="">선택하세요</option>{datasets.map((row) => <option key={row.id} value={row.id}>{row.name} · {TARGET_LABELS[row.target] ?? row.target} · {row.content_hash.slice(0, 12)}</option>)}
      </select></label>
      <label><span>트리거 조건</span><input value={form.trigger_policy} onChange={set("trigger_policy")} maxLength={120} /></label>
    </div>
    {!datasets.length && <p role="status">DataOps에서 학습데이터를 등록한 뒤 선택하세요.</p>}
    {error && <p className="pipeline-register-error" role="alert">{error}</p>}
    <div className="pipeline-register-actions">
      <button type="submit" className="btn btn-primary" disabled={submitting || !datasets.length}>{submitting ? "등록 중…" : "파이프라인 등록"}</button>
      <button type="button" className="btn btn-secondary" onClick={onCancel} disabled={submitting}>취소</button>
    </div>
  </form>;
}
