import { readMockDataVisible } from "./demoVisibility.js";
let mode = readMockDataVisible() ? "mock" : "real";
let epoch = 0;
const pending = new Set();
export const currentDataMode = () => mode;
export const dataModeEpoch = () => epoch;
export function setDataMode(next) {
  if (!new Set(["mock", "real"]).has(next)) throw new Error("데이터 모드를 확인하세요.");
  if (next === mode) return;
  mode = next;
  epoch += 1;
  for (const controller of pending) controller.abort();
  pending.clear();
}
export function beginDataRequest() {
  const controller = new AbortController();
  const requestEpoch = epoch;
  pending.add(controller);
  return { mode, signal: controller.signal, isCurrent: () => requestEpoch === epoch,
    release: () => pending.delete(controller) };
}
export const scenarioStorageKey = (isMock) => `decline_poc_scenarios.${isMock ? "mock" : "real"}.v2`;

// 셸 헤더의 데이터 모드 안내 — 화면 배지는 제거되었고, 문구는 화면 낭독용(sr-only) live 영역에서만 쓴다.
const SWITCH_HINT = "설정의 '데모 데이터 표시'에서 바꿀 수 있습니다.";
export function modeBadge(demoVisible) {
  return demoVisible
    ? { tone: "demo", icon: "fa-flask", text: "데모 데이터 표시 중", title: `화면의 값은 예시 데이터입니다. ${SWITCH_HINT}` }
    : { tone: "real", icon: "fa-database", text: "실데이터 표시 중", title: `등록한 학습데이터와 활성 모델 결과만 표시합니다. ${SWITCH_HINT}` };
}

// 모드 전환 직후 콘솔에 남기는 한 줄.
export const modeSwitchLog = (demoVisible) => demoVisible
  ? "INFO: 데모 데이터 표시로 전환했습니다. 화면의 값은 예시입니다."
  : "INFO: 실데이터 표시로 전환했습니다. 등록한 학습데이터와 활성 모델 결과만 표시합니다.";