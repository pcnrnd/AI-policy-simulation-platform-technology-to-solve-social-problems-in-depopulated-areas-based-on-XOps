// 오케스트레이션 쓰기 호출 인증 회귀 검사 (vitest 미도입 → node 직접 실행).
//
// 백엔드는 /orchestration 쓰기(파이프라인 등록·삭제·실행, 재학습 이벤트)에 data:write 토큰을
// 요구한다. 토큰 없이 부르는 화면 코드가 남으면 그 버튼이 401로 깨진다 — 호출 표현식에
// `token` 이 함께 있는지 본다.
import assert from "node:assert/strict";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const srcRoot = join(here, "..");

function* sourceFiles(dir) {
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) yield* sourceFiles(full);
    else if (/\.(jsx?|mjs)$/.test(entry) && !entry.endsWith(".test.mjs")) yield full;
  }
}

/** `apiSend(` 호출 하나의 인자 텍스트를 괄호 균형으로 잘라 낸다(여러 줄 호출 대응). */
function apiSendCalls(text) {
  const calls = [];
  const needle = "apiSend(";
  let at = text.indexOf(needle);
  while (at !== -1) {
    let depth = 0;
    let end = at + needle.length - 1;
    for (; end < text.length; end += 1) {
      if (text[end] === "(") depth += 1;
      else if (text[end] === ")") {
        depth -= 1;
        if (depth === 0) break;
      }
    }
    calls.push({ index: at, text: text.slice(at, end + 1) });
    at = text.indexOf(needle, end + 1);
  }
  return calls;
}

const offenders = [];
let found = 0;
for (const file of sourceFiles(srcRoot)) {
  const text = readFileSync(file, "utf-8");
  for (const call of apiSendCalls(text)) {
    if (!call.text.includes("/api/v3/orchestration/")) continue;
    found += 1;
    if (/\btoken\b/.test(call.text)) continue;
    const lineNo = text.slice(0, call.index).split("\n").length;
    offenders.push(`${file.slice(srcRoot.length + 1)}:${lineNo}`);
  }
}

// 등록 폼·삭제·재학습 이벤트 3곳이 검사 대상에 잡혀야 검사 자체가 유효하다.
assert.ok(found >= 3, `오케스트레이션 쓰기 호출을 ${found}곳만 찾았다(3곳 이상이어야 한다)`);
assert.deepEqual(offenders, [], `토큰 없이 부르는 오케스트레이션 쓰기:\n${offenders.join("\n")}`);
console.log(`  ok  오케스트레이션 쓰기 ${found}곳이 모두 토큰을 보낸다`);
