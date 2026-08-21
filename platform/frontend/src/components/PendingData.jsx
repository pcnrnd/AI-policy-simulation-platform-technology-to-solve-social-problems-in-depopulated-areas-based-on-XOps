// 요인분석 실행 전 데이터 대기 플레이스홀더 — UI 구조는 항상 노출, 데이터만 실행 후 공개.
export default function PendingData({ running = false, text }) {
  return (
    <p className="pl-pending">
      <i className="fa-solid fa-hourglass-half" aria-hidden="true"></i>
      {running ? "분석 결과를 준비 중입니다." : text ?? "[요인분석 실행] 후 데이터가 표시됩니다."}
    </p>
  );
}
