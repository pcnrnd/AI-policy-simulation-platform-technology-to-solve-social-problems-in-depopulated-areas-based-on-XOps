import { useState } from "react";

// 신규 아카이브 등록 폼 — DataOps 라이프사이클의 "메타데이터 등록" 단계를 사용자 조작으로 실증.
// 저장소 유형 선택에 따라 Adapter·쿼리 언어(SQL/MQL)가 자동 결정되며,
// 등록 즉시 카탈로그·스키마·API 빌더(STEP ②③)에서 기본 소스와 동일하게 동작한다.

const SOURCE_TYPES = [
  { value: "RDB · PostgreSQL", prefix: "tb_", lang: "SQL" },
  { value: "NoSQL · MongoDB", prefix: "col_", lang: "MQL" },
  { value: "공간 DB · PostGIS", prefix: "geo_", lang: "SQL" },
  { value: "시계열 DB · TimescaleDB", prefix: "ts_", lang: "SQL" }
];

const TIERS = [
  { value: "Hot", hint: "고빈도 조회 · 고속 스토리지" },
  { value: "Warm", hint: "간헐 조회 · 중간 등급" },
  { value: "Cold", hint: "보존 위주 · 저비용 아카이브" }
];

const RETENTIONS = ["1년 보존", "3년 보존", "5년 보존", "10년 보존", "영구 보존"];

const EMPTY_COLUMN = { name: "", type: "", description: "" };

const INITIAL_FORM = {
  label: "",
  source: SOURCE_TYPES[0].value,
  object: "",
  description: "",
  tags: "",
  tier: "Warm",
  retention: "5년 보존",
  rangeColumn: "",
  rangeFrom: "",
  rangeTo: "",
  columns: [{ ...EMPTY_COLUMN }]
};

// 숫자형 range 값은 숫자로 보존 (MQL $gte/$lte·SQL BETWEEN 모두 타입 일치)
const coerce = (v) => {
  const n = Number(v);
  return v !== "" && Number.isFinite(n) ? n : v;
};

export default function ArchiveRegisterForm({ onRegister, onCancel }) {
  const [form, setForm] = useState(INITIAL_FORM);
  const [error, setError] = useState("");

  const setField = (key, value) => setForm((f) => ({ ...f, [key]: value }));

  const setColumn = (idx, key, value) =>
    setForm((f) => ({
      ...f,
      columns: f.columns.map((c, i) => (i === idx ? { ...c, [key]: value } : c))
    }));

  const addColumn = () => setForm((f) => ({ ...f, columns: [...f.columns, { ...EMPTY_COLUMN }] }));

  const removeColumn = (idx) =>
    setForm((f) => ({
      ...f,
      columns: f.columns.filter((_, i) => i !== idx),
      // 범위 기준 컬럼이 삭제되면 범위 설정도 해제
      ...(f.columns[idx]?.name === f.rangeColumn
        ? { rangeColumn: "", rangeFrom: "", rangeTo: "" }
        : {})
    }));

  const typeDef = SOURCE_TYPES.find((t) => t.value === form.source) ?? SOURCE_TYPES[0];
  const validColumns = form.columns.filter((c) => c.name.trim() && c.type.trim());

  const handleSubmit = () => {
    if (!form.label.trim()) return setError("데이터 소스명을 입력하세요.");
    if (!form.object.trim()) return setError(`데이터 객체명을 입력하세요. (예: ${typeDef.prefix}new_dataset)`);
    if (validColumns.length === 0) return setError("컬럼을 1개 이상 정의하세요 (이름·타입 필수).");
    if (form.rangeColumn && (form.rangeFrom === "" || form.rangeTo === ""))
      return setError("수집 범위를 설정하려면 from~to 값을 모두 입력하세요.");

    const id = `ds_user_${Date.now().toString(36)}`;
    const schema = {
      id,
      label: form.label.trim(),
      source: form.source,
      object: form.object.trim(),
      description: form.description.trim() || "사용자 등록 데이터 소스",
      archive: {
        tier: form.tier,
        retention: form.retention,
        loaded_at: new Date().toISOString().slice(0, 10),
        rows: 0
      },
      lineage: { origin: "사용자 등록 (수동 메타데이터)", version: "v1", commit: id.slice(-6) },
      ...(form.rangeColumn
        ? {
            range: {
              column: form.rangeColumn,
              from: coerce(form.rangeFrom.trim()),
              to: coerce(form.rangeTo.trim())
            }
          }
        : {}),
      tags: form.tags
        .split(",")
        .map((t) => t.trim().replace(/^#/, ""))
        .filter(Boolean),
      columns: validColumns.map((c) => ({
        name: c.name.trim(),
        type: c.type.trim(),
        description: c.description.trim() || "—"
      })),
      userRegistered: true
    };
    setError("");
    setForm(INITIAL_FORM);
    onRegister(schema);
  };

  return (
    <div className="archive-reg-form" role="form" aria-label="신규 아카이브 등록">
      <div className="archive-reg-head">
        <i className="fa-solid fa-box-archive" aria-hidden="true"></i> 신규 아카이브 등록 —
        메타데이터를 정의하면 적재(아카이빙) 후 즉시 가상화 API 대상이 됩니다
      </div>

      <div className="archive-reg-grid">
        <label className="field-inline">
          <span>소스명 *</span>
          <input
            className="input-control"
            placeholder="예: 빈집 실태조사"
            value={form.label}
            onChange={(e) => setField("label", e.target.value)}
          />
        </label>
        <label className="field-inline">
          <span>저장소 유형 *</span>
          <select
            className="select-control"
            value={form.source}
            onChange={(e) => setField("source", e.target.value)}
          >
            {SOURCE_TYPES.map((t) => (
              <option key={t.value} value={t.value}>
                {t.value} ({t.lang})
              </option>
            ))}
          </select>
        </label>
        <label className="field-inline">
          <span>데이터 객체 *</span>
          <input
            className="input-control"
            placeholder={`예: ${typeDef.prefix}new_dataset`}
            value={form.object}
            onChange={(e) => setField("object", e.target.value)}
          />
        </label>
        <label className="field-inline">
          <span>태그</span>
          <input
            className="input-control"
            placeholder="쉼표 구분 (예: 주거, 정형)"
            value={form.tags}
            onChange={(e) => setField("tags", e.target.value)}
          />
        </label>
        <label className="field-inline archive-reg-desc">
          <span>설명</span>
          <input
            className="input-control"
            placeholder="데이터 출처·용도 요약"
            value={form.description}
            onChange={(e) => setField("description", e.target.value)}
          />
        </label>
        <label className="field-inline">
          <span>아카이브 티어</span>
          <select
            className="select-control"
            value={form.tier}
            onChange={(e) => setField("tier", e.target.value)}
            title={TIERS.find((t) => t.value === form.tier)?.hint}
          >
            {TIERS.map((t) => (
              <option key={t.value} value={t.value}>
                {t.value} — {t.hint}
              </option>
            ))}
          </select>
        </label>
        <label className="field-inline">
          <span>보존 정책</span>
          <select
            className="select-control"
            value={form.retention}
            onChange={(e) => setField("retention", e.target.value)}
          >
            {RETENTIONS.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
        </label>
      </div>

      {/* 컬럼 정의 */}
      <div className="archive-reg-sub-label">
        컬럼 정의 * <span>(이름·타입 필수 — API 빌더의 SELECT/find 대상이 됩니다)</span>
      </div>
      {form.columns.map((col, idx) => (
        <div className="archive-reg-col-row" key={idx}>
          <input
            className="input-control"
            placeholder="컬럼명 (예: house_id)"
            value={col.name}
            onChange={(e) => setColumn(idx, "name", e.target.value)}
            aria-label={`컬럼 ${idx + 1} 이름`}
          />
          <input
            className="input-control"
            placeholder={typeDef.lang === "MQL" ? "타입 (예: String)" : "타입 (예: VARCHAR(10))"}
            value={col.type}
            onChange={(e) => setColumn(idx, "type", e.target.value)}
            aria-label={`컬럼 ${idx + 1} 타입`}
          />
          <input
            className="input-control"
            placeholder="설명"
            value={col.description}
            onChange={(e) => setColumn(idx, "description", e.target.value)}
            aria-label={`컬럼 ${idx + 1} 설명`}
          />
          <button
            type="button"
            className="btn btn-secondary archive-reg-col-del"
            onClick={() => removeColumn(idx)}
            disabled={form.columns.length === 1}
            aria-label={`컬럼 ${idx + 1} 삭제`}
          >
            <i className="fa-solid fa-minus" aria-hidden="true"></i>
          </button>
        </div>
      ))}
      <button type="button" className="btn btn-secondary archive-reg-col-add" onClick={addColumn}>
        <i className="fa-solid fa-plus" aria-hidden="true"></i> 컬럼 추가
      </button>

      {/* 수집 범위 (선택) */}
      <div className="archive-reg-sub-label">
        수집 범위 <span>(선택 — Adapter가 쿼리에 자동 주입하는 적재 스코프)</span>
      </div>
      <div className="archive-reg-range-row">
        <select
          className="select-control"
          value={form.rangeColumn}
          onChange={(e) => setField("rangeColumn", e.target.value)}
          aria-label="범위 기준 컬럼"
        >
          <option value="">범위 없음</option>
          {validColumns.map((c) => (
            <option key={c.name} value={c.name}>
              {c.name}
            </option>
          ))}
        </select>
        <input
          className="input-control"
          placeholder="from (예: 20240101)"
          value={form.rangeFrom}
          onChange={(e) => setField("rangeFrom", e.target.value)}
          disabled={!form.rangeColumn}
          aria-label="범위 시작값"
        />
        <input
          className="input-control"
          placeholder="to (예: 20261231)"
          value={form.rangeTo}
          onChange={(e) => setField("rangeTo", e.target.value)}
          disabled={!form.rangeColumn}
          aria-label="범위 종료값"
        />
      </div>

      {error && (
        <p className="archive-reg-error" role="alert">
          <i className="fa-solid fa-circle-exclamation" aria-hidden="true"></i> {error}
        </p>
      )}

      <div className="archive-reg-actions">
        <button type="button" className="btn btn-secondary" onClick={onCancel}>
          취소
        </button>
        <button type="button" className="btn btn-primary" onClick={handleSubmit}>
          <i className="fa-solid fa-tags" aria-hidden="true"></i> 메타데이터 등록 · 적재
        </button>
      </div>
    </div>
  );
}
