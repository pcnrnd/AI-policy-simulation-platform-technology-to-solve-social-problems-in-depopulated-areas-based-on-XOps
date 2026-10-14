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
