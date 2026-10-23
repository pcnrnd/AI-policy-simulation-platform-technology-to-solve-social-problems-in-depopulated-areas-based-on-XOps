// 목록 페이징 순수 계산 — TablePager(usePager)·DataOps 삭제 후 이동 쪽이 같은 규칙을 쓴다(node 로 바로 검사: pager.test.mjs).

// 목록형 UI 공통 페이지 크기 — 항목이 이보다 많으면 이 단위로 나눠 보인다.
export const LIST_PAGE_SIZE = 5;

// 페이지 슬라이스 — 목록이 줄어 현재 페이지가 범위를 벗어나면 마지막 페이지로 보정.
export function paginate(rows, page, pageSize) {
  const totalPages = Math.max(1, Math.ceil(rows.length / pageSize));
  const safePage = Math.min(Math.max(1, page), totalPages);
  return {
    pageRows: rows.slice((safePage - 1) * pageSize, safePage * pageSize),
    safePage,
    totalPages
  };
}

/** 0부터 센 행 위치 → 그 행이 있는 쪽(1부터). 삭제 뒤 다음 대상으로 이동할 때 쓴다. */
export const pageOfIndex = (index, pageSize = LIST_PAGE_SIZE) => Math.floor(index / pageSize) + 1;

/**
 * 목록 대상 키(등록 모델·지역·소스 등)가 바뀌면 1쪽 상태로, 같으면 그대로 돌려준다.
 * 렌더할 때마다 맞추므로 A→B→A로 돌아와도 A의 이전 쪽이 되살아나지 않는다.
 */
export const syncPagerState = (state, key) => (state.key === key ? state : { key, page: 1 });
