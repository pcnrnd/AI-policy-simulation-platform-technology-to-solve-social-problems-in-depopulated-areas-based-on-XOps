// 목록 페이징 순수 계산 검사 (vitest 미도입 → node 직접 실행). 2026-10-23 독립 검토 F1~F3 회귀 방지.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { LIST_PAGE_SIZE, paginate, pageOfIndex, syncPagerState } from "./pager.js";

const rows = Array.from({ length: 12 }, (_, i) => i);
assert.equal(LIST_PAGE_SIZE, 5);
assert.deepEqual(paginate(rows, 2, 5), { pageRows: [5, 6, 7, 8, 9], safePage: 2, totalPages: 3 });
assert.deepEqual(paginate(rows.slice(0, 3), 3, 5).safePage, 1); // 목록이 줄면 마지막 쪽으로 보정
assert.deepEqual(paginate([], 1, 5), { pageRows: [], safePage: 1, totalPages: 1 });

// F1: 삭제 뒤 다음 대상(0부터 센 위치)이 있는 쪽 — 5개 단위(5번 → 2쪽, 12번 → 3쪽), 0~4번 → 1쪽.
assert.deepEqual([0, 4, 5, 9, 10, 12].map((i) => pageOfIndex(i)), [1, 1, 2, 2, 3, 3]);
assert.equal(pageOfIndex(12, 10), 2);

// F2·F3: 대상 키가 바뀌면 1쪽, 같은 키면 상태를 그대로(참조 동일 → 렌더 중 setState 반복 없음).
const onA2 = { key: "A", page: 2 };
assert.equal(syncPagerState(onA2, "A"), onA2);
const onB = syncPagerState(onA2, "B");
assert.deepEqual(onB, { key: "B", page: 1 });
// A→B→A: B에서 아무것도 누르지 않고 A로 돌아와도 A의 2쪽이 되살아나지 않는다.
assert.deepEqual(syncPagerState(onB, "A"), { key: "A", page: 1 });
assert.equal(syncPagerState({ key: undefined, page: 3 }, undefined).page, 3); // 키 없는 목록은 그대로

// 소스 가드: DataOps 삭제 후 이동 쪽에 10 고정값이 남지 않고 공통 계산을 쓴다. 지도 격자·스키마는 대상 키로 초기화.
const src = (...p) => readFileSync(join(dirname(fileURLToPath(import.meta.url)), "..", ...p), "utf-8");
const dataops = src("pages", "DataOpsPage.jsx");
assert.doesNotMatch(dataops, /\/ 10\)/);
assert.equal(dataops.match(/nextPage: nextIndexAfterDelete >= 0 \? pageOfIndex\(nextIndexAfterDelete, (?:LIST_PAGE_SIZE|BUILT_PAGE_SIZE)\) : 1/g)?.length, 2);
assert.match(dataops, /syncPagerState\(schemaPage, target\.id\)/);
assert.match(src("components", "SimulatorMap.jsx"), /usePager\(allowSeed \? buildRegionGrid\(currentRegion\) : \[\], undefined, currentRegion\.id\)/);
assert.match(src("components", "TablePager.jsx"), /const synced = syncPagerState\(state, resetKey\);\s*if \(synced !== state\) setState\(synced\);/);
console.log("pager: 5개 단위 슬라이스·삭제 후 이동 쪽·대상 변경 시 1쪽(A→B→A 포함)·DataOps/지도 격자/usePager 연결 PASS");
