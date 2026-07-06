// 테이블 하단 페이저 — 행이 페이지 크기를 넘을 때만 렌더된다.
// 사용처: Model Store, 재학습 파이프라인 카탈로그, 발급된 API 목록 등 증가형 목록.
export default function TablePager({ page, totalPages, totalCount, onChange }) {
  if (totalPages <= 1) return null;
  return (
    <div className="table-pager">
      <span className="table-pager-info">총 {totalCount}건</span>
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
      <span className="table-pager-page">
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
    </div>
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
