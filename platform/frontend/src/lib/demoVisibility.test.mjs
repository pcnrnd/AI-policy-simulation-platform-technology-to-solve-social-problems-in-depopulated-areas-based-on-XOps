// 데모 데이터 표시 기본값 회귀 검사 (vitest 미도입 → node 직접 실행).
//
// 범위 문서(population-platform-scope.md)는 "데모 표시 OFF를 실데이터 기본 검증 상태"로 정한다.
// 저장값이 없는 첫 방문·손상된 저장값·스토리지 접근 불가 — 어느 경우든 화면은 OFF 로 떠야 하고,
// 사용자가 명시적으로 켠 값("true")만 ON 으로 복원한다.
import assert from "node:assert/strict";
import {
  MOCK_DATA_DEFAULT_VISIBLE,
  MOCK_DATA_STORAGE_KEY,
  readMockDataVisible
} from "./demoVisibility.js";

let passed = 0;
const check = (name, fn) => {
  fn();
  passed += 1;
  console.log(`  ok  ${name}`);
};

/** 메모리 스토리지 — 초기 항목을 주면 그 값으로 시작한다. */
const memoryStorage = (entries = {}) => {
  const map = new Map(Object.entries(entries));
  return { getItem: (key) => (map.has(key) ? map.get(key) : null) };
};

/** 접근 자체가 던지는 스토리지(프라이빗 모드·권한 거부). */
const throwingStorage = {
  getItem() {
    throw new DOMException("denied", "SecurityError");
  }
};

check("기본값 상수는 OFF(false)다", () => {
  assert.equal(MOCK_DATA_DEFAULT_VISIBLE, false);
});

check("저장값이 없으면(첫 방문) OFF", () => {
  assert.equal(readMockDataVisible(memoryStorage()), false);
});

check("저장값이 손상됐으면 OFF", () => {
  for (const raw of ["", "yes", "1", "TRUE", "null"]) {
    assert.equal(readMockDataVisible(memoryStorage({ [MOCK_DATA_STORAGE_KEY]: raw })), false, `raw=${raw}`);
  }
});

check("스토리지를 못 쓰면 OFF", () => {
  assert.equal(readMockDataVisible(throwingStorage), false);
  assert.equal(readMockDataVisible(undefined), false);
});

check("명시적으로 켠 값(\"true\")만 ON 으로 복원한다", () => {
  assert.equal(readMockDataVisible(memoryStorage({ [MOCK_DATA_STORAGE_KEY]: "true" })), true);
  assert.equal(readMockDataVisible(memoryStorage({ [MOCK_DATA_STORAGE_KEY]: "false" })), false);
});

console.log(`\n${passed} checks passed`);
