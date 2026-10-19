import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { setDataMode, beginDataRequest, dataModeEpoch, scenarioStorageKey, modeBadge, modeSwitchLog, sourceChipText } from "./dataMode.js";
import { REAL_GUIDE_TEXT, REAL_CONTROLS_LOCK_TEXT, FACTOR_GUIDE_TEXT, FACTOR_PENDING_TEXT, realUnavailableText } from "./realSimulation.js";import { fetchReportData } from "./dataopsApi.js";
import { alertText } from "./alertText.js";
import { buildReportBlocks, buildReportRows } from "./reportContent.js";
setDataMode("real");
const request = beginDataRequest();
const start = dataModeEpoch();
setDataMode("mock");
assert.equal(request.signal.aborted, true);
assert.equal(request.isCurrent(), false);
assert.ok(dataModeEpoch() > start);
assert.equal(await fetchReportData({ id: "namwon" }), null);
const mock = beginDataRequest();
setDataMode("real");
setDataMode("mock");
assert.equal(mock.isCurrent(), false);
assert.notEqual(scenarioStorageKey(true), scenarioStorageKey(false));
const region = { id: "namwon", name: "남원", population: 987654, birthRate: 0.82, agingIndex: 31, riskIndex: 0.2 };
const template = { id: "template_analysis", title: "검증" };
const extra = { allowSeed: false, generatedAt: "2026-10-14T00:00:00Z", live: { mae: 1, wape: 0.1, baselineMae: 2, psi: [] } };
assert.doesNotMatch(JSON.stringify(buildReportBlocks(region, template, extra)), /987654|60%|10년 후|복지 예산/);
assert.equal(buildReportRows(region, template, extra).some((row) => row.includes(987654)), false);
// 상시 모드 배지: 문구·아이콘·tone 이 함께 바뀐다.
assert.deepEqual([modeBadge(true).text, modeBadge(true).tone], ["데모 데이터 표시 중", "demo"]);
assert.deepEqual([modeBadge(false).text, modeBadge(false).tone], ["실데이터 표시 중", "real"]);
assert.notEqual(modeBadge(true).icon, modeBadge(false).icon);
// 출처 칩 규칙(설계 §1-2 표 전 조합)
assert.equal(sourceChipText(true, "mock"), "데모");
assert.equal(sourceChipText(true, "api"), "데모 포함");
assert.equal(sourceChipText(true, "realdata"), "데모 포함");
assert.equal(sourceChipText(false, "api"), "실데이터");
assert.equal(sourceChipText(false, "realdata"), "실데이터");
assert.equal(sourceChipText(false, "mock"), null); // 실데이터 표시의 데모 출처(값 숨김) 카드에는 칩 없음
assert.equal(sourceChipText(false, undefined), null);
assert.equal(sourceChipText(true, "api", "데모"), "데모");
assert.equal(sourceChipText(false, undefined, "실데이터 · 등록본"), "실데이터 · 등록본");
// 화면 문구에 내부 구현명·제약 표현 금지
const FORBIDDEN = /mock|seed|시드|경로|modelops|\/api\/|정책 변수|VITE_|_KEY/i;
const shown = [true, false].flatMap((on) => [modeBadge(on).text, modeBadge(on).title, modeSwitchLog(on),
  ...["mock", "api", "realdata"].map((src) => sourceChipText(on, src)).filter(Boolean)]);
const guides = [REAL_GUIDE_TEXT, realUnavailableText("지자체 비교 분포는"), REAL_CONTROLS_LOCK_TEXT, FACTOR_GUIDE_TEXT, FACTOR_PENDING_TEXT];
for (const text of [...shown, ...guides]) assert.doesNotMatch(text, FORBIDDEN, text);
// 알림 문구: 서버·브라우저 원문은 고정 문구로, 사용자용 한글 안내(4xx·화면 코드)는 그대로
const FALLBACK = "잠시 뒤 다시 시도하세요.";
const httpError = (status, message) => Object.assign(new Error(message), { status });
assert.equal(alertText(httpError(500, "HTTP 500"), FALLBACK), FALLBACK);
assert.equal(alertText(new TypeError("Failed to fetch"), FALLBACK), FALLBACK);
assert.equal(alertText(new DOMException("signal timed out", "TimeoutError"), FALLBACK), FALLBACK);
assert.equal(alertText(httpError(503, "서버 점검 중입니다."), FALLBACK), FALLBACK);
assert.equal(alertText(httpError(422, "[object Object]"), FALLBACK), FALLBACK);
assert.equal(alertText(httpError(400, "Value error, 입력값을 확인하세요"), FALLBACK), FALLBACK);
assert.equal(alertText(new Error("RuntimeError: 학습 실패"), FALLBACK), FALLBACK);
assert.equal(alertText(httpError(409, "데이터가 갱신됐습니다. 목록을 다시 확인하세요."), FALLBACK), "데이터가 갱신됐습니다. 목록을 다시 확인하세요.");
assert.equal(alertText(new Error("파이프라인 ID를 입력하세요."), FALLBACK), "파이프라인 ID를 입력하세요.");
assert.equal(alertText(undefined, null), null);
// 화면 상태(오류·피드백·결과 문구)에 오류 원문을 직접 넣는 코드가 없다(리포터 탭은 담당 범위 밖이라 제외).
const srcRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
const RAW_INTO_STATE = /set\w*(?:Error|Feedback|Result)\([^;]*\b(?:err|error)\??\.message|message: `[^`]*\$\{(?:err|error)\??\.message/;
const rawHits = readdirSync(srcRoot, { recursive: true })
  .filter((file) => /\.jsx?$/.test(file) && !/ReporterPage\.jsx$/.test(file))
  .flatMap((file) => readFileSync(join(srcRoot, file), "utf-8").split("\n").filter((line) => RAW_INTO_STATE.test(line)).map((line) => `${file}: ${line.trim()}`));
assert.deepEqual(rawHits, []);
// 개요·모니터의 빈 상태는 값을 채우는 방법을 한 문장으로 알린다 — 상태만 알리던 옛 문구 금지
const srcText = (file) => readFileSync(join(srcRoot, file), "utf-8");
const STATE_ONLY = /실데이터가 아직 없습니다|요약할 수 없습니다|NO_DEMO_DATA|"데이터 없음"|비교 구간이 아직 없습니다|표시할 적재량이 없습니다|수집 경로가 아직 없어/;
for (const file of ["pages/MonitorPage.jsx", "pages/Overview.jsx", "components/RegionStatusCard.jsx", "components/PendingData.jsx", "components/realdata/RealdataEvaluationPanel.jsx"]) {
  assert.doesNotMatch(srcText(file), STATE_ONLY, file);
}
const emptyHints = [
  srcText("pages/MonitorPage.jsx").match(/const EMPTY_HINT = "([^"]+)";/)?.[1],
  srcText("pages/Overview.jsx").match(/const SOURCE_EMPTY_HINT = "([^"]+)";/)?.[1],
  srcText("components/PendingData.jsx").match(/export const DEMO_ONLY_HINT = "([^"]+)";/)?.[1]
];
for (const text of emptyHints) {
  assert.match(text ?? "", /면 [^.]*표시됩니다\.$/, text);
  assert.doesNotMatch(text, /미구현|구현되지 않|지원하지 않|제한|mock|seed|시드|경로|\/api\//i, text);
}
console.log("dataMode: 전환 요청 취소·epoch 격리·추천 저장키·mock 보고서 실측 호출 0·real 보고서 시드 값 차단·모드 배지·출처 칩·화면 문구 금지어·알림 원문 차단 PASS");
