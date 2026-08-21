// 테이블 하단 페이저 — 한 페이지뿐이어도 전체 건수와 현재 표시 범위를 유지한다.
export default function TablePager({ page, totalPages, totalCount, pageSize = 5, onChange }) {
  const first = totalCount === 0 ? 0 : (page - 1) * pageSize + 1;
  const last = Math.min(totalCount, page * pageSize);
  return (
    <nav className="table-pager" aria-label="표 페이지 탐색">
      <span className="table-pager-info" aria-live="polite">
        총 {totalCount}건 · {first}–{last}건 표시
      </span>
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
    </nav>
  );
}

// 페이지 슬라이스 헬퍼 — 목록이 줄어 현재 페이지가 범위를 벗어나면 마지막 페이지로 보정.
export function paginate(rows, page, pageSize) {
  const totalPages = Math.max(1, Math.ceil(rows.length / pageSize));
  const safePage = Math.min(Math.max(1, page), totalPages);
  return {
    pageRows: rows.slice((safePage - 1) * pageSize, safePage * pageSize),
    safePage,
    totalPages
  };
}
