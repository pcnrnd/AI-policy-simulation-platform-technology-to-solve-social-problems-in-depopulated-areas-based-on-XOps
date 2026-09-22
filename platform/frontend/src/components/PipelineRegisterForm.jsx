import { useState } from "react";
import { apiSend } from "../lib/api.js";

// 백엔드 PipelineCreateRequest.id 패턴과 같은 규칙 — 서버 422를 받기 전에 같은 문구로 막는다.
const ID_PATTERN = /^[A-Za-z0-9._-]{3,64}$/;

const EMPTY = { id: "", name: "", model_id: "", trigger_policy: "수동", experiment: "" };

/**
 * ML 재학습 파이프라인 등록 폼 — POST /api/v3/orchestration/pipelines (SQLite 영속화).
 * 등록 성공 시 onRegistered(정의)로 상위 카탈로그를 다시 불러오게 한다.
 */
export default function PipelineRegisterForm({ models, onRegistered, onCancel }) {
  const [form, setForm] = useState(EMPTY);
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  const set = (key) => (event) => setForm((prev) => ({ ...prev, [key]: event.target.value }));

  const validate = () => {
    if (!ID_PATTERN.test(form.id.trim())) return "파이프라인 ID는 영문·숫자·.-_ 3~64자여야 합니다.";
    if (!form.name.trim()) return "파이프라인 이름을 입력하세요.";
    if (!form.model_id) return "대상 모델을 선택하세요.";
    return null;
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (submitting) return;
    const invalid = validate();
    if (invalid) {
      setError(invalid);
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const created = await apiSend("POST", "/api/v3/orchestration/pipelines", {
        body: {
          id: form.id.trim(),
          name: form.name.trim(),
          model_id: form.model_id,
          trigger_policy: form.trigger_policy.trim() || "수동",
          experiment: form.experiment.trim()
        }
      });
      setForm(EMPTY);
      onRegistered?.(created);
    } catch (err) {
      setError(err?.message ?? "등록에 실패했습니다.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form className="pipeline-register" onSubmit={handleSubmit} aria-label="재학습 파이프라인 등록">
      <div className="pipeline-register-grid">
        <label>
          <span>파이프라인 ID</span>
          <input value={form.id} onChange={set("id")} placeholder="PL-POP-RETRAIN-01" required />
        </label>
        <label>
          <span>파이프라인 이름</span>
          <input value={form.name} onChange={set("name")} placeholder="인구이동 예측 재학습" required />
        </label>
        <label>
          <span>대상 모델</span>
          <select value={form.model_id} onChange={set("model_id")} required>
            <option value="">선택하세요</option>
            {models.map((m) => (
              <option key={m.model_id} value={m.model_id}>
                {m.model_id} ({m.version})
              </option>
            ))}
          </select>
        </label>
        <label>
          <span>트리거 조건</span>
          <input value={form.trigger_policy} onChange={set("trigger_policy")} placeholder="수동 · 드리프트(PSI > 0.2)" />
        </label>
        <label>
          <span>실험 ID</span>
          <input value={form.experiment} onChange={set("experiment")} placeholder="EXP-POP-DECLINE-031" />
        </label>
      </div>
      {error && (
        <p className="pipeline-register-error" role="alert">
          <i className="fa-solid fa-circle-exclamation" aria-hidden="true"></i> {error}
        </p>
      )}
      <div className="pipeline-register-actions">
        <button type="submit" className="btn btn-primary" aria-disabled={submitting}>
          <i className="fa-solid fa-plus" aria-hidden="true"></i> {submitting ? "등록 중…" : "등록"}
        </button>
        <button type="button" className="btn btn-secondary" onClick={onCancel}>
          취소
        </button>
      </div>
    </form>
  );
}
