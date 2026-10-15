import assert from "node:assert/strict";
import { setDataMode, beginDataRequest, dataModeEpoch, scenarioStorageKey, modeBadge, modeSwitchLog, sourceChipText } from "./dataMode.js";
import { REAL_GUIDE_TEXT, REAL_CONTROLS_LOCK_TEXT, REAL_DIAGNOSIS_NOTE, realUnavailableText } from "./realSimulation.js";import { fetchReportData } from "./dataopsApi.js";
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
assert.equal(sourceChipText(false, "mock"), "데모");
assert.equal(sourceChipText(false, undefined), null);
assert.equal(sourceChipText(true, "api", "데모"), "데모");
assert.equal(sourceChipText(false, undefined, "실데이터 · 등록본"), "실데이터 · 등록본");
// 화면 문구에 내부 구현명·제약 표현 금지
const FORBIDDEN = /mock|seed|시드|경로|modelops|\/api\/|정책 변수/i;
const shown = [true, false].flatMap((on) => [modeBadge(on).text, modeBadge(on).title, modeSwitchLog(on),
  ...["mock", "api", "realdata"].map((src) => sourceChipText(on, src))]);
const guides = [REAL_GUIDE_TEXT, realUnavailableText("문제 유형 진단은"), REAL_CONTROLS_LOCK_TEXT, REAL_DIAGNOSIS_NOTE];
for (const text of [...shown, ...guides]) assert.doesNotMatch(text, FORBIDDEN, text);
console.log("dataMode: 전환 요청 취소·epoch 격리·추천 저장키·mock 보고서 실측 호출 0·real 보고서 시드 값 차단·모드 배지·출처 칩·화면 문구 금지어 PASS");
