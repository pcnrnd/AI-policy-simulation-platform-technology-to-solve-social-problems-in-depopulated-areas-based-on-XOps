// FilterBuilder 변환·검증 순수 함수 소형 테스트 (vitest 미도입 → node 직접 실행).
import assert from "node:assert/strict";
import {
  filterValidationMessage,
  formatFilterRows,
  parseFilterText
} from "./filterExpression.js";

const COLUMNS = [
  { name: "reg_date", type: "VARCHAR(8)" },
  { name: "in_flow_count", type: "INTEGER" },
  { name: "sentiment_score", type: "float" }
];

let passed = 0;
const check = (name, fn) => {
  fn();
  passed += 1;
  console.log(`  ok  ${name}`);
};

// 케이스 1 — 수치형 단일 조건: 문자열 ↔ 행 왕복이 같은 문자열로 돌아온다.
check("수치형 단일 조건 왕복", () => {
  const rows = parseFilterText("in_flow_count > 100");
  assert.deepEqual(rows, [{ column: "in_flow_count", op: ">", value: "100" }]);
  assert.equal(formatFilterRows(rows, COLUMNS), "in_flow_count > 100");
});

// 케이스 2 — AND 결합 + 문자 컬럼 인용 유지.
check("AND 2조건 왕복(문자 컬럼은 작은따옴표 유지)", () => {
  const text = "reg_date = '20260101' AND in_flow_count >= 50";
  const rows = parseFilterText(text);
  assert.deepEqual(rows, [
    { column: "reg_date", op: "=", value: "20260101" },
    { column: "in_flow_count", op: ">=", value: "50" }
  ]);
  assert.equal(formatFilterRows(rows, COLUMNS), text);
  assert.equal(filterValidationMessage(text, COLUMNS), null);
});

// 케이스 3 — 행으로 표현할 수 없는 식은 null(고급 입력 전용)이고 검증에서 막힌다.
check("OR 결합은 행 변환 불가 + 검증 차단", () => {
  assert.equal(parseFilterText("in_flow_count > 100 OR reg_date = '20260101'"), null);
  assert.match(
    filterValidationMessage("in_flow_count > 100 OR reg_date = '20260101'", COLUMNS),
    /AND으?로만 연결/
  );
});

// 부가 검증 — 상한·빈 값·미지 컬럼·수치형 타입 불일치.
check("조건 6개는 상한 초과로 거부", () => {
  const over = Array.from({ length: 6 }, (_, i) => `in_flow_count > ${i}`).join(" AND ");
  assert.equal(parseFilterText(over), null);
  assert.match(filterValidationMessage(over, COLUMNS), /최대 5개/);
});

check("값이 빈 행은 조용히 누락되지 않고 문구로 걸린다", () => {
  const text = formatFilterRows(
    [
      { column: "in_flow_count", op: ">", value: "10" },
      { column: "reg_date", op: "=", value: "" }
    ],
    COLUMNS
  );
  assert.equal(text, "in_flow_count > 10 AND reg_date =");
  assert.match(filterValidationMessage(text, COLUMNS), /값이 비어 있는 조건/);
});

check("스키마에 없는 컬럼과 수치형 타입 불일치를 구분해 알린다", () => {
  assert.match(filterValidationMessage("ghost = 1", COLUMNS), /컬럼이 없습니다/);
  assert.match(filterValidationMessage("in_flow_count > abc", COLUMNS), /수치형 컬럼/);
  assert.equal(filterValidationMessage("", COLUMNS), null);
});

check("값의 세미콜론·따옴표는 결합 오류와 구분해 안내", () => {
  assert.match(filterValidationMessage("reg_date = 'a';DROP", COLUMNS), /세미콜론/);
  assert.match(filterValidationMessage("reg_date = it's", COLUMNS), /따옴표/);
  assert.match(filterValidationMessage("in_flow_count > 100 AND (a = 1)", COLUMNS), /OR·괄호 미지원/);
});

console.log(`\n${passed} passed`);
