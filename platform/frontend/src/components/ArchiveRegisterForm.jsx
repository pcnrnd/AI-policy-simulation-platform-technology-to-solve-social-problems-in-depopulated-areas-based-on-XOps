import { useId, useRef, useState } from "react";

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

export default function ArchiveRegisterForm({ onRegister, onCancel, onSubmittingChange }) {
  const [form, setForm] = useState(INITIAL_FORM);
  const [errors, setErrors] = useState({});
  const [submitError, setSubmitError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const formId = useId();
  const labelRef = useRef(null);
  const objectRef = useRef(null);
  const columnRefs = useRef([]);
  const rangeFromRef = useRef(null);
  const rangeToRef = useRef(null);
  const errorSummaryRef = useRef(null);

  const setField = (key, value) => {
    setForm((f) => ({ ...f, [key]: value }));
    setErrors((current) => {
      const hasValue = String(value ?? "").trim() !== "";
      const clearsRangeError =
        (key === "rangeColumn" && !hasValue && current.range) ||
        (key === "rangeFrom" && hasValue && current.range?.field === "from") ||
        (key === "rangeTo" && hasValue && current.range?.field === "to");
      const clearsFieldError = current[key] && hasValue;
      if (!clearsFieldError && !clearsRangeError) return current;
      const next = { ...current };
      if (clearsFieldError) delete next[key];
      if (clearsRangeError) delete next.range;
      return next;
    });
    setSubmitError("");
  };

  const setColumn = (idx, key, value) => {
    setForm((f) => {
      const previousName = f.columns[idx]?.name;
      const updates =
        key === "name" && f.rangeColumn !== "" && previousName === f.rangeColumn
          ? { rangeColumn: String(value), rangeFrom: f.rangeFrom, rangeTo: f.rangeTo }
          : {};
      return {
        ...f,
        ...updates,
        columns: f.columns.map((c, i) => (i === idx ? { ...c, [key]: value } : c))
      };
    });
    setErrors((current) => {
      if (
        !String(value ?? "").trim() ||
        !current.columns ||
        current.columns.index !== idx ||
        current.columns.field !== key
      ) return current;
      const next = { ...current };
      delete next.columns;
      return next;
    });
    setSubmitError("");
  };

  const addColumn = () => setForm((f) => ({ ...f, columns: [...f.columns, { ...EMPTY_COLUMN }] }));

  const removeColumn = (idx) => {
    setForm((f) => ({
      ...f,
      columns: f.columns.filter((_, i) => i !== idx),
      // 범위 기준 컬럼이 삭제되면 범위 설정도 해제
      ...(f.columns[idx]?.name === f.rangeColumn
        ? { rangeColumn: "", rangeFrom: "", rangeTo: "" }
        : {})
    }));
    setErrors((current) => {
      if (!current.columns) return current;
      const next = { ...current };
      delete next.columns;
      return next;
    });
  };

  const typeDef = SOURCE_TYPES.find((t) => t.value === form.source) ?? SOURCE_TYPES[0];
  const validColumns = form.columns.filter((c) => c.name.trim() && c.type.trim());

  const validate = () => {
    const next = {};
    if (!form.label.trim()) next.label = "데이터 소스명을 입력하세요.";
    if (!form.object.trim()) next.object = `데이터 객체명을 입력하세요. 예: ${typeDef.prefix}new_dataset`;
    const incompleteColumn = form.columns.findIndex(
      (column) => !column.name.trim() || !column.type.trim()
    );
    if (validColumns.length === 0 || incompleteColumn >= 0) {
      const field = !form.columns[Math.max(0, incompleteColumn)]?.name.trim() ? "name" : "type";
      next.columns = {
        index: Math.max(0, incompleteColumn),
        field,
        message: `컬럼 ${Math.max(0, incompleteColumn) + 1}의 ${field === "name" ? "이름" : "타입"}을 입력하세요.`
      };
    }
    if (form.rangeColumn && !String(form.rangeFrom).trim()) {
      next.range = { field: "from", message: "수집 범위의 시작값을 입력하세요." };
    } else if (form.rangeColumn && !String(form.rangeTo).trim()) {
      next.range = { field: "to", message: "수집 범위의 종료값을 입력하세요." };
    }
    return next;
  };

  const focusFirstError = (nextErrors) => {
    const first = [
      ["label", labelRef.current],
      ["object", objectRef.current],
      ["columns", nextErrors.columns ? columnRefs.current[nextErrors.columns.index]?.[nextErrors.columns.field] : null],
      ["range", nextErrors.range?.field === "to" ? rangeToRef.current : rangeFromRef.current]
    ].find(([key]) => nextErrors[key]);
    requestAnimationFrame(() => first?.[1]?.focus());
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (submitting) return;
    const nextErrors = validate();
    if (Object.keys(nextErrors).length > 0) {
      setErrors(nextErrors);
      setSubmitError("");
      focusFirstError(nextErrors);
      return;
    }

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
              column: form.rangeColumn.trim(),
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
    setErrors({});
    setSubmitError("");
    setSubmitting(true);
    onSubmittingChange?.(true);
    try {
      await onRegister(schema);
      setForm({ ...INITIAL_FORM, columns: [{ ...EMPTY_COLUMN }] });
    } catch (error) {
      setSubmitError(error?.message || "등록하지 못했습니다. 입력값을 확인한 뒤 다시 시도하세요.");
      requestAnimationFrame(() => errorSummaryRef.current?.focus());
    } finally {
      setSubmitting(false);
      onSubmittingChange?.(false);
    }
  };

  return (
    <form
      className="archive-reg-form"
      aria-label="신규 아카이브 등록"
      aria-busy={submitting}
      onSubmit={handleSubmit}
      noValidate
    >
      <div className="archive-reg-head">
        <i className="fa-solid fa-box-archive" aria-hidden="true"></i> 신규 아카이브 등록 —
        메타데이터를 정의하면 적재(아카이빙) 후 즉시 가상화 API 대상이 됩니다
      </div>

      <div className="archive-reg-grid">
        <label className="field-inline">
          <span>소스명 *</span>
          <input
            ref={labelRef}
            id={`${formId}-label`}
            className="input-control"
            placeholder="예: 빈집 실태조사"
            value={form.label}
            onChange={(e) => setField("label", e.target.value)}
            onBlur={() => {
              if (!form.label.trim()) setErrors((current) => ({ ...current, label: "데이터 소스명을 입력하세요." }));
            }}
            required
            aria-invalid={Boolean(errors.label)}
            aria-describedby={errors.label ? `${formId}-label-error` : undefined}
          />
        </label>
        {errors.label && <span id={`${formId}-label-error`} className="field-error">{errors.label}</span>}
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
            ref={objectRef}
            id={`${formId}-object`}
            className="input-control"
            placeholder={`예: ${typeDef.prefix}new_dataset`}
            value={form.object}
            onChange={(e) => setField("object", e.target.value)}
            onBlur={() => {
              if (!form.object.trim()) {
                setErrors((current) => ({
                  ...current,
                  object: `데이터 객체명을 입력하세요. 예: ${typeDef.prefix}new_dataset`
                }));
              }
            }}
            required
            aria-invalid={Boolean(errors.object)}
            aria-describedby={errors.object ? `${formId}-object-error` : undefined}
          />
        </label>
        {errors.object && <span id={`${formId}-object-error`} className="field-error">{errors.object}</span>}
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
          <label className="archive-reg-field">
            <span>컬럼 {idx + 1} 이름 *</span>
            <input
              ref={(node) => {
                columnRefs.current[idx] = { ...(columnRefs.current[idx] ?? {}), name: node };
              }}
              className="input-control"
              placeholder="예: house_id"
              value={col.name}
              onChange={(e) => setColumn(idx, "name", e.target.value)}
              required
              onBlur={() => {
                if (!col.name.trim()) {
                  setErrors((current) => ({
                    ...current,
                    columns: { index: idx, field: "name", message: `컬럼 ${idx + 1}의 이름을 입력하세요.` }
                  }));
                }
              }}
              aria-invalid={errors.columns?.index === idx && errors.columns.field === "name"}
              aria-describedby={errors.columns?.index === idx && errors.columns.field === "name" ? `${formId}-column-${idx}-name-error` : undefined}
            />
            {errors.columns?.index === idx && errors.columns.field === "name" && (
              <span id={`${formId}-column-${idx}-name-error`} className="field-error">{errors.columns.message}</span>
            )}
          </label>
          <label className="archive-reg-field">
            <span>컬럼 {idx + 1} 타입 *</span>
            <input
              ref={(node) => {
                columnRefs.current[idx] = { ...(columnRefs.current[idx] ?? {}), type: node };
              }}
              className="input-control"
              placeholder={typeDef.lang === "MQL" ? "예: String" : "예: VARCHAR(10)"}
              value={col.type}
              onChange={(e) => setColumn(idx, "type", e.target.value)}
              required
              onBlur={() => {
                if (!col.type.trim()) {
                  setErrors((current) => ({
                    ...current,
                    columns: { index: idx, field: "type", message: `컬럼 ${idx + 1}의 타입을 입력하세요.` }
                  }));
                }
              }}
              aria-invalid={errors.columns?.index === idx && errors.columns.field === "type"}
              aria-describedby={errors.columns?.index === idx && errors.columns.field === "type" ? `${formId}-column-${idx}-type-error` : undefined}
            />
            {errors.columns?.index === idx && errors.columns.field === "type" && (
              <span id={`${formId}-column-${idx}-type-error`} className="field-error">{errors.columns.message}</span>
            )}
          </label>
          <label className="archive-reg-field">
            <span>컬럼 {idx + 1} 설명 (선택)</span>
            <input
              className="input-control"
              placeholder="컬럼 설명"
              value={col.description}
              onChange={(e) => setColumn(idx, "description", e.target.value)}
            />
          </label>
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
        <label className="archive-reg-field">
          <span>기준 컬럼 (선택)</span>
          <select
            className="select-control"
            value={form.rangeColumn}
            onChange={(e) => setField("rangeColumn", e.target.value)}
          >
            <option value="">범위 없음</option>
            {validColumns.map((c) => (
              <option key={c.name} value={c.name}>
                {c.name}
              </option>
            ))}
          </select>
        </label>
        <label className="archive-reg-field">
          <span>시작값 {form.rangeColumn ? "*" : "(선택)"}</span>
          <input
            ref={rangeFromRef}
            className="input-control"
            placeholder="예: 20240101"
            value={form.rangeFrom}
            onChange={(e) => setField("rangeFrom", e.target.value)}
            onBlur={() => {
              if (form.rangeColumn && !String(form.rangeFrom).trim()) {
                setErrors((current) => ({
                  ...current,
                  range: { field: "from", message: "수집 범위의 시작값을 입력하세요." }
                }));
              }
            }}
            disabled={!form.rangeColumn}
            required={Boolean(form.rangeColumn)}
            aria-invalid={errors.range?.field === "from"}
            aria-describedby={errors.range?.field === "from" ? `${formId}-range-from-error` : undefined}
          />
          {errors.range?.field === "from" && (
            <span id={`${formId}-range-from-error`} className="field-error">{errors.range.message}</span>
          )}
        </label>
        <label className="archive-reg-field">
          <span>종료값 {form.rangeColumn ? "*" : "(선택)"}</span>
          <input
            ref={rangeToRef}
            className="input-control"
            placeholder="예: 20261231"
            value={form.rangeTo}
            onChange={(e) => setField("rangeTo", e.target.value)}
            onBlur={() => {
              if (form.rangeColumn && !String(form.rangeTo).trim()) {
                setErrors((current) => ({
                  ...current,
                  range: { field: "to", message: "수집 범위의 종료값을 입력하세요." }
                }));
              }
            }}
            disabled={!form.rangeColumn}
            required={Boolean(form.rangeColumn)}
            aria-invalid={errors.range?.field === "to"}
            aria-describedby={errors.range?.field === "to" ? `${formId}-range-to-error` : undefined}
          />
          {errors.range?.field === "to" && (
            <span id={`${formId}-range-to-error`} className="field-error">{errors.range.message}</span>
          )}
        </label>
      </div>

      {(Object.keys(errors).length > 0 || submitError) && (
        <p ref={errorSummaryRef} className="archive-reg-error" role="alert" tabIndex="-1">
          <i className="fa-solid fa-circle-exclamation" aria-hidden="true"></i>{" "}
          {submitError || "입력한 내용을 확인해 주세요. 오류가 있는 첫 번째 항목으로 이동했습니다."}
        </p>
      )}

      <div className="archive-reg-actions">
        <button type="button" className="btn btn-secondary" onClick={onCancel} disabled={submitting}>
          취소
        </button>
        <button type="submit" className="btn btn-primary" disabled={submitting}>
          <i className={`fa-solid ${submitting ? "fa-spinner fa-spin" : "fa-tags"}`} aria-hidden="true"></i>{" "}
          {submitting ? "등록 중" : "메타데이터 등록 · 적재"}
        </button>
      </div>
    </form>
  );
}
