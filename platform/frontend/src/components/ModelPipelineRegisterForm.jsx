import { useState } from "react";
import { apiSend } from "../lib/api.js";
import { alertText } from "../lib/alertText.js";
import { EXPERIMENT_MAX, MODEL_TYPES, TARGET_LABELS, pipelineDefinition } from "../lib/modelops.js";
import { getRealdataToken } from "./realdata/realdataClient.jsx";
import { useAppState } from "../context/AppStateContext.jsx";

// 대상 모델 = 모델 종류 2개(이전 고정 모델 3개는 과거 버전으로만 남고 선택지에 없다).
const EMPTY = { id: "", name: "", model_type: "factor_analysis", dataset_id: "", trigger_policy: "수동", experiment: "" };
export default function ModelPipelineRegisterForm({ datasets = [], onRegistered, onCancel }) {
  const { addConsoleLog } = useAppState();
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
      setForm(EMPTY);
      onRegistered?.(created);
    } catch (err) {
      setError(alertText(err, "등록하지 못했습니다. 입력한 내용을 확인한 뒤 다시 등록하세요."));
      addConsoleLog(`WARN: 재학습 파이프라인 등록 실패 — ${err?.message ?? err}`, false, true);
    }
    finally { setSubmitting(false); }
  };
  return <form className="pipeline-register" onSubmit={handleSubmit} aria-label="재학습 파이프라인 등록">
    <div className="pipeline-register-grid">
      <label><span>파이프라인 ID</span><input value={form.id} onChange={set("id")} placeholder="PL-POP-RETRAIN-01" required maxLength={128} /></label>
      <label><span>파이프라인 이름</span><input value={form.name} onChange={set("name")} placeholder="인구 변화율 요인분석 재학습" required maxLength={120} /></label>
      <label><span>대상 모델</span><select value={form.model_type} onChange={set("model_type")}>{MODEL_TYPES.map((type) => <option key={type.id} value={type.id}>{type.name}</option>)}</select></label>
      <label><span>등록 학습데이터</span><select value={form.dataset_id} onChange={set("dataset_id")} required>
        <option value="">선택하세요</option>{datasets.map((row) => <option key={row.id} value={row.id}>{row.name} · {TARGET_LABELS[row.target] ?? row.target} · {row.content_hash.slice(0, 12)}</option>)}
      </select></label>
      <label><span>트리거 조건</span><input value={form.trigger_policy} onChange={set("trigger_policy")} placeholder="수동" maxLength={120} /></label>
      <label><span>실험 ID</span><input value={form.experiment} onChange={set("experiment")} placeholder="EXP-POP-DECLINE-031" maxLength={EXPERIMENT_MAX} /></label>
    </div>
    {!datasets.length && <p role="status">DataOps에서 학습데이터를 등록한 뒤 선택하세요.</p>}
    {error && <p className="pipeline-register-error" role="alert"><i className="fa-solid fa-circle-exclamation" aria-hidden="true"></i> {error}</p>}
    <div className="pipeline-register-actions">
      <button type="submit" className="btn btn-primary" disabled={submitting || !datasets.length}><i className="fa-solid fa-plus" aria-hidden="true"></i> {submitting ? "등록 중…" : "등록"}</button>
      <button type="button" className="btn btn-secondary" onClick={onCancel} disabled={submitting}>취소</button>
    </div>
  </form>;
}
