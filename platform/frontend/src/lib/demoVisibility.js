// 데모 데이터 표시 스위치의 저장·복원 규칙. JSX 와 분리해 node 로 바로 검사한다(demoVisibility.test.mjs).
//
// 범위 문서(population-platform-scope.md): "데모 표시 OFF를 실데이터 기본 검증 상태로 삼는다."
// 저장값이 없거나(첫 방문) 손상됐거나 스토리지를 못 쓰면 OFF 로 뜨고, 사용자가 켠 "true" 만 ON 으로 복원한다.

// 키를 올린 이유: 이전 키("decline-poc-mock-data")는 기본값이 ON 이었고 마운트 효과가 그 값을
// 그대로 저장했다. 같은 키를 쓰면 한 번이라도 열어 본 브라우저는 사용자가 켠 적이 없어도
// "true" 가 남아 기본값 변경이 적용되지 않는다. 새 키로 모든 브라우저가 OFF 에서 다시 시작한다.
export const MOCK_DATA_STORAGE_KEY = "decline-poc-mock-data.v2";

export const MOCK_DATA_DEFAULT_VISIBLE = false;

/**
 * 저장값 → 데모 표시 여부. 저장값이 정확히 "true" 일 때만 ON, 그 외는 전부 기본값(OFF).
 * @param {{ getItem(key: string): string | null } | undefined} storage 기본은 localStorage
 * @returns {boolean}
 */
export function readMockDataVisible(storage = globalThis.localStorage) {
  try {
    return storage.getItem(MOCK_DATA_STORAGE_KEY) === "true";
  } catch {
    return MOCK_DATA_DEFAULT_VISIBLE;
  }
}
