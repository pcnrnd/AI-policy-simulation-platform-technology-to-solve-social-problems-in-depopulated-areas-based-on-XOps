/** DataOps 필터 조건 — 조건 행 목록 ↔ 필터 문자열 변환과 검증(순수 함수).
 *
 * 요청 payload·생성 SQL 프리뷰·발급 API 레코드는 모두 기존 문자열 형식을 그대로 쓰므로,
 * 조건 행 UI(FilterBuilder)는 이 모듈을 거쳐 문자열로만 바깥과 주고받는다.
 *
 * 문법은 백엔드 `src/dataops/safety.py` 와 1:1로 맞춘다.
 *   조건    `컬럼 연산자 값`
 *   연산자  = != > >= < <=
 *   값      숫자 / 작은·큰따옴표 문자열(따옴표·세미콜론 불가) / 밑줄·영숫자 단어
 *   결합    ` AND ` 만(대소문자 무시). OR·괄호 미지원, 조건 최대 5개
 * 위반 시 백엔드는 UnsafeQueryError(HTTP 400)를 낸다.
 */

export const FILTER_OPERATORS = ["=", "!=", ">", ">=", "<", "<="];
/** 백엔드 safety.MAX_FILTER_CONDITIONS 와 같은 값. */
export const MAX_FILTER_CONDITIONS = 5;
/** 발급 API 레코드에 저장되는 filter 문자열 상한 — schemas/dataops.py 와 같은 값. */
export const MAX_FILTER_LENGTH = 256;

const FILTER_PATTERN = /^(\w+)\s*(>=|<=|!=|=|>|<)\s*('[^';]*'|"[^";]*"|-?\d+(?:\.\d+)?|\w+)$/;
const EMPTY_VALUE_PATTERN = /^\w+\s*(?:>=|<=|!=|=|>|<)\s*$/;
const NUMBER_PATTERN = /^-?\d+(?:\.\d+)?$/;
// 백엔드 _AND_SPLIT_RE 와 동일 — 공백으로 둘러싸인 AND 에서만 나눈다.
const AND_SPLIT_PATTERN = /\s+AND\s+/i;
// 컬럼 타입 앞머리로 수치형을 판정한다(INTEGER·BIGINT·NUMERIC(10,2)·float …).
const NUMERIC_TYPE_PATTERN = /^\s*(?:int|integer|bigint|smallint|serial|numeric|decimal|real|double|float|number)/i;

export const FILTER_SYNTAX_HELP = [
  "지원 연산자: =, !=, >, >=, <, <=",
  "값: 수치형 컬럼은 숫자 그대로(100, -3.5), 그 밖의 컬럼은 작은따옴표로 자동 인용됩니다. 값 안에 따옴표·세미콜론은 쓸 수 없습니다.",
  `결합: AND만 지원합니다(OR·괄호 미지원). 조건은 최대 ${MAX_FILTER_CONDITIONS}개까지 사용할 수 있습니다.`,
  "예시 ① in_flow_count > 100",
  "예시 ② reg_date = '20260101' AND in_flow_count >= 50"
].join("\n");

/** 필터 문자열을 ` AND ` 기준 단일 조건 목록으로. 빈 입력은 빈 배열. */
export function splitFilterConditions(text) {
  const expression = String(text ?? "").trim();
  if (!expression) return [];
  return expression.split(AND_SPLIT_PATTERN).map((part) => part.trim());
}

/** 컬럼이 수치형인지 — 값 인용 여부를 가른다. 스키마에 없는 컬럼은 수치형이 아니다. */
export function isNumericColumn(name, columns = []) {
  const column = columns.find((c) => c.name === name);
  return Boolean(column && NUMERIC_TYPE_PATTERN.test(String(column.type ?? "")));
}

const unquote = (raw) => raw.replace(/^'(.*)'$/s, "$1").replace(/^"(.*)"$/s, "$1");
const isNumberLiteral = (value) => NUMBER_PATTERN.test(String(value ?? "").trim());

/** 조건 행 하나 → `컬럼 연산자 값`. 값이 비면 연산자까지만 남겨 검증에서 걸리게 한다. */
function formatFilterRow(row, columns) {
  const value = String(row.value ?? "").trim();
  if (!value) return `${row.column} ${row.op}`;
  const literal = isNumericColumn(row.column, columns) ? value : `'${value}'`;
  return `${row.column} ${row.op} ${literal}`;
}

/** 조건 행 목록 → 기존 필터 문자열. 값이 빈 행도 지우지 않고 남긴다(조용한 누락 방지). */
export function formatFilterRows(rows = [], columns = []) {
  return rows
    .filter((row) => row && row.column && row.op)
    .map((row) => formatFilterRow(row, columns))
    .join(" AND ");
}

/** 필터 문자열 → 조건 행 목록. 행 UI로 표현할 수 없으면 null(고급 입력 전용). */
export function parseFilterText(text) {
  const conditions = splitFilterConditions(text);
  if (!conditions.length) return [];
  if (conditions.length > MAX_FILTER_CONDITIONS) return null;
  const rows = [];
  for (const condition of conditions) {
    const match = condition.match(FILTER_PATTERN);
    if (!match) return null;
    rows.push({ column: match[1], op: match[2], value: unquote(match[3]) });
  }
  return rows;
}

/** 전송 전 검증 — 문제가 없으면 null, 있으면 사용자에게 보일 문구 하나. */
export function filterValidationMessage(text, columns = []) {
  const conditions = splitFilterConditions(text);
  if (!conditions.length) return null;
  // 발급 API 레코드(schemas/dataops.py BuiltApiRequest.filter)의 상한 — 넘으면 등록이 422로 막힌다.
  if (String(text).trim().length > MAX_FILTER_LENGTH) {
    return `필터 식이 너무 깁니다(${String(text).trim().length}자 > ${MAX_FILTER_LENGTH}자). 조건을 줄이세요.`;
  }
  if (conditions.length > MAX_FILTER_CONDITIONS) {
    return `필터 조건은 최대 ${MAX_FILTER_CONDITIONS}개까지 사용할 수 있습니다. (현재 ${conditions.length}개)`;
  }
  for (const condition of conditions) {
    if (EMPTY_VALUE_PATTERN.test(condition)) {
      return "값이 비어 있는 조건이 있습니다. 값을 입력하거나 해당 조건 행을 삭제하세요.";
    }
    const match = condition.match(FILTER_PATTERN);
    if (!match) {
      // 결합 오류가 가장 흔하다 — 값 안의 정상 따옴표를 오류로 지목하지 않도록 먼저 가른다.
      if (/\bOR\b/i.test(condition) || /[()]/.test(condition)) {
        return "조건은 AND로만 연결할 수 있습니다(OR·괄호 미지원). 예: in_flow_count > 100 AND reg_date = '20260101'";
      }
      if (condition.includes(";")) {
        return `조건 “${condition}”의 값에는 세미콜론을 쓸 수 없습니다.`;
      }
      if (/['"]/.test(condition)) {
        return `조건 “${condition}”의 값에는 따옴표를 쓸 수 없습니다.`;
      }
      return "‘컬럼 연산자 값’ 형식으로 입력하고 조건은 AND로만 연결하세요. 예: in_flow_count > 100";
    }
    if (!columns.some((column) => column.name === match[1])) {
      return `현재 스키마에 ‘${match[1]}’ 컬럼이 없습니다. 목록에 있는 컬럼명을 사용하세요.`;
    }
    // 수치형 컬럼에 문자열을 넣으면 문법은 통과하지만 저장소에서 실패한다 — 전송 전에 막는다.
    if (isNumericColumn(match[1], columns) && !isNumberLiteral(match[3])) {
      return `‘${match[1]}’은 수치형 컬럼입니다. 숫자 값을 입력하세요. (입력값: ${match[3]})`;
    }
  }
  return null;
}

/** 새 조건 행의 초기값 — 첫 컬럼과 `=`. */
export function emptyFilterRow(columns = []) {
  return { column: columns[0]?.name ?? "", op: "=", value: "" };
}
