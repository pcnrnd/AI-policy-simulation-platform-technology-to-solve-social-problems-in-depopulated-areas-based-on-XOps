import { useState } from "react";
import { LIST_PAGE_SIZE, paginate, syncPagerState } from "../lib/pager.js";

// 공통 페이지 크기·슬라이스는 lib/pager.js(순수 계산) — 기존 import 경로를 위해 여기서도 내보낸다.
export { LIST_PAGE_SIZE, paginate };

// 테이블 하단 페이저 — 한 페이지뿐이어도 전체 건수와 현재 표시 범위를 유지한다.
export default function TablePager({ page, totalPages, totalCount, pageSize = 5, onChange }) {
  const first = totalCount === 0 ? 0 : (page - 1) * pageSize + 1;
  const last = Math.min(totalCount, page * pageSize);
  return (
    <nav className="table-pager" aria-label="표 페이지 탐색">
      <span className="table-pager-info" aria-live="polite">
        총 {totalCount}건 · {first}–{last}건 표시
      </span>
      {/* 이전·쪽수·다음은 한 묶음 — 좁은 카드에서 줄바꿈돼도 버튼이 갈라지지 않는다. */}
      <span className="table-pager-nav">
      <button
        type="button"
        className="btn btn-secondary"
        style={{ padding: "3px 10px", fontSize: 11 }}
        disabled={page <= 1}
        onClick={() => onChange(page - 1)}
        aria-label="이전 페이지"
      >
        <i className="fa-solid fa-chevron-left" aria-hidden="true"></i> 이전
      </button>
      <span className="table-pager-page" aria-live="polite">
        {page} / {totalPages}
      </span>
      <button
        type="button"
        className="btn btn-secondary"
        style={{ padding: "3px 10px", fontSize: 11 }}
        disabled={page >= totalPages}
        onClick={() => onChange(page + 1)}
        aria-label="다음 페이지"
      >
        다음 <i className="fa-solid fa-chevron-right" aria-hidden="true"></i>
      </button>
      </span>
    </nav>
  );
}

// 목록 하나의 페이지 상태 — 현재 페이지 행과 <TablePager {...pager} /> 에 넘길 값을 함께 돌려준다.
// resetKey: 목록의 대상(등록 모델·지역 등)이 바뀌면 그 렌더에서 1쪽으로 맞춘다 — 돌아와도 이전 쪽을 되살리지 않는다.
export function usePager(rows, pageSize = LIST_PAGE_SIZE, resetKey) {
  const [state, setState] = useState({ key: resetKey, page: 1 });
  const synced = syncPagerState(state, resetKey);
  if (synced !== state) setState(synced); // 렌더 중 상태 맞춤(React 허용 패턴, 키가 같아지면 멈춘다)
  const { pageRows, safePage, totalPages } = paginate(rows, synced.page, pageSize);
  return { pageRows, offset: (safePage - 1) * pageSize,
    pager: { page: safePage, totalPages, totalCount: rows.length, pageSize, onChange: (next) => setState({ key: resetKey, page: next }) } };
}
