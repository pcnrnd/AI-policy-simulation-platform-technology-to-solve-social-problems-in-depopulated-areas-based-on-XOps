import assert from "node:assert/strict";
import { buildReportBlocks, buildReportRows, blocksToMarkdown, formatClock, formatDateTime, formatReportTimestamp } from "./reportContent.js";
import { buildDocx, buildXlsx, buildHwpHtml } from "./reportExport.js";

const generatedAt = "2026-10-14T04:45:00.000Z";
const expected = "2026-10-14 13:45 (KST)";
const region = { id: "namwon", name: "전북 남원시", population: 76000, birthRate: 0.82, agingIndex: 33.1, riskIndex: 0.21 };
const template = { id: "template_analysis", title: "보고서" };
const blocks = buildReportBlocks(region, template, { generatedAt });
const rows = buildReportRows(region, template, { generatedAt });
assert.equal(formatReportTimestamp(generatedAt), expected);
assert.equal(formatReportTimestamp("2026-10-14T15:00:00.000Z"), "2026-10-15 00:00 (KST)");
assert.equal(rows.find((row) => row[0] === "생성 일시")[1], expected);
assert.ok(blocksToMarkdown(template.title, blocks).includes(expected));
assert.ok((await buildHwpHtml({ title: template.title, blocks }).text()).includes(expected));
// OOXML ZIP은 비압축 저장이므로 XML 본문을 직접 확인할 수 있다.
for (const blob of [buildDocx({ title: template.title, blocks }), buildXlsx({ rows, sheetName: "지표요약" })]) {
  const text = new TextDecoder().decode(await blob.arrayBuffer());
  assert.ok(text.includes(expected));
  assert.ok(!text.includes("2026-05-23"));
}
// 화면 시각 형식: 날짜 포함은 한 줄 "YYYY-MM-DD HH:mm", 시각만은 24시간 "HH:mm:ss"(오전/오후 없음).
assert.equal(formatDateTime("2026-09-22T05:27:40Z"), "2026-09-22 14:27");
assert.equal(formatClock("2026-09-22T05:58:17Z"), "14:58:17");
assert.equal(formatClock("2026-09-22T15:05:09Z"), "00:05:09");
console.log("reportTimestamp: 미리보기 포맷·MD·HWP·DOCX·XLSX 생성 시각 KST 일치·자정 변환·화면 시각 형식 PASS");
