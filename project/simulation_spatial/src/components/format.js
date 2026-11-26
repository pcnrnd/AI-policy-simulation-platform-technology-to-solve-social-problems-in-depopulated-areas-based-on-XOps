// 화면 숫자 서식(왼쪽 요약 카드와 오른쪽 상세가 같이 쓴다)
export const fmt = (n) => Math.round(n).toLocaleString("ko-KR");
export const km = (v) => (v == null || !Number.isFinite(v) ? "—" : `${v.toFixed(2)}km`);
export const pct = (v) => `${v.toFixed(1)}%`;
// 늘고 준 사람 수: "+120명"·"−35명"·"0명"(반올림한 값의 부호)
export const signedPeople = (n) => {
  const r = Math.round(n);
  return `${r > 0 ? "+" : r < 0 ? "−" : ""}${fmt(Math.abs(r))}명`;
};
