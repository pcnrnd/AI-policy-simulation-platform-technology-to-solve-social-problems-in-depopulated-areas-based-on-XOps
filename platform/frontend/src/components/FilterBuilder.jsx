import { useState } from "react";
import InfoTip from "./InfoTip.jsx";
import {
  FILTER_OPERATORS,
  FILTER_SYNTAX_HELP,
  MAX_FILTER_CONDITIONS,
  emptyFilterRow,
  formatFilterRows,
  isNumericColumn,
  parseFilterText
} from "../lib/filterExpression.js";

/** DataOps STEP③ 필터 조건 편집기 — 조건 행 목록(기본)과 문자열 직접 입력(고급).
 *
 * 바깥과는 항상 기존 필터 문자열 하나로만 주고받는다(요청 payload·SQL 프리뷰·발급 레코드 동일).
 * 소스를 바꾸면 컬럼 목록이 달라지므로 부모가 `key={소스 id}` 로 이 컴포넌트를 다시 만든다.
 */
export default function FilterBuilder({ value, columns = [], onChange, onBlur, error, errorId, inputRef }) {
  // 문자열이 행으로 표현되지 않으면(OR·괄호 등) 고급 입력으로 열어 내용을 잃지 않게 한다.
  const [rows, setRows] = useState(() => parseFilterText(value) ?? []);
  const [advanced, setAdvanced] = useState(() => parseFilterText(value) === null);
  const [modeNote, setModeNote] = useState(null);

  const emit = (nextRows) => {
    setRows(nextRows);
    onChange(formatFilterRows(nextRows, columns));
  };

  const updateRow = (index, patch) =>
    emit(rows.map((row, i) => (i === index ? { ...row, ...patch } : row)));

  const toggleAdvanced = () => {
    if (!advanced) {
      setModeNote(null);
      setAdvanced(true);
      return;
    }
    const parsed = parseFilterText(value);
    if (parsed === null) {
      setModeNote(
        `현재 식은 조건 행으로 표현할 수 없습니다(AND 결합·최대 ${MAX_FILTER_CONDITIONS}개 조건만 가능). 식을 고치거나 비운 뒤 전환하세요.`
      );
      return;
    }
    setModeNote(null);
    setRows(parsed);
    setAdvanced(false);
    // 행 기준으로 다시 직렬화해 표기(인용 등)를 통일한다.
    onChange(formatFilterRows(parsed, columns));
  };

  const atLimit = rows.length >= MAX_FILTER_CONDITIONS;

  return (
    <div className="filter-builder">
      <div className="filter-builder-head">
        <span className="filter-builder-label" id="dataops-filter-label">
          필터 조건 (선택)
          <InfoTip text={FILTER_SYNTAX_HELP} label="필터 조건 문법 설명 보기" />
        </span>
        <button
          type="button"
          className="btn btn-tertiary filter-mode-toggle"
          aria-pressed={advanced}
          onClick={toggleAdvanced}
        >
          <i className="fa-solid fa-terminal" aria-hidden="true"></i> 고급(직접 입력)
        </button>
      </div>

      {advanced ? (
        <input
          ref={inputRef}
          className="input-control mock-data-output"
          placeholder="예: in_flow_count > 100 AND reg_date = '20260101'"
          value={value}
          aria-labelledby="dataops-filter-label"
          aria-invalid={Boolean(error)}
          aria-describedby={error ? errorId : undefined}
          onChange={(e) => onChange(e.target.value)}
          onBlur={onBlur}
        />
      ) : (
        <>
          {rows.length === 0 && (
            <p className="filter-empty-note">조건이 없으면 적재 범위 전체를 조회합니다.</p>
          )}
          <ul className="filter-row-list">
            {rows.map((row, index) => (
              <li className="filter-row" key={index}>
                <span className="filter-row-join" aria-hidden={index === 0}>
                  {index === 0 ? "WHERE" : "AND"}
                </span>
                <select
                  className="select-control"
                  value={row.column}
                  aria-label={`조건 ${index + 1} 컬럼`}
                  onChange={(e) => updateRow(index, { column: e.target.value })}
                >
                  {columns.map((column) => (
                    <option key={column.name} value={column.name}>
                      {column.name}
                    </option>
                  ))}
                </select>
                <select
                  className="select-control filter-row-op"
                  value={row.op}
                  aria-label={`조건 ${index + 1} 연산자`}
                  onChange={(e) => updateRow(index, { op: e.target.value })}
                >
                  {FILTER_OPERATORS.map((op) => (
                    <option key={op} value={op}>
                      {op}
                    </option>
                  ))}
                </select>
                <input
                  ref={index === 0 ? inputRef : undefined}
                  className="input-control mock-data-output"
                  value={row.value}
                  inputMode={isNumericColumn(row.column, columns) ? "decimal" : "text"}
                  placeholder={isNumericColumn(row.column, columns) ? "숫자" : "값"}
                  aria-label={`조건 ${index + 1} 값`}
                  onChange={(e) => updateRow(index, { value: e.target.value })}
                  onBlur={onBlur}
                />
                <button
                  type="button"
                  className="btn btn-secondary filter-row-del"
                  aria-label={`조건 ${index + 1} 행 삭제`}
                  onClick={() => emit(rows.filter((_, i) => i !== index))}
                >
                  <i className="fa-solid fa-trash-can" aria-hidden="true"></i>
                </button>
              </li>
            ))}
          </ul>
          <div className="filter-builder-actions">
            <button
              type="button"
              className="btn btn-secondary filter-row-add"
              disabled={atLimit || columns.length === 0}
              onClick={() => emit([...rows, emptyFilterRow(columns)])}
            >
              <i className="fa-solid fa-plus" aria-hidden="true"></i> 조건 추가
            </button>
            {atLimit && (
              <span className="filter-limit-note">
                조건은 최대 {MAX_FILTER_CONDITIONS}개까지 결합할 수 있습니다.
              </span>
            )}
          </div>
        </>
      )}

      {modeNote && <span className="field-error">{modeNote}</span>}
      {error && (
        <span id={errorId} className="field-error">
          {error}
        </span>
      )}
    </div>
  );
}
