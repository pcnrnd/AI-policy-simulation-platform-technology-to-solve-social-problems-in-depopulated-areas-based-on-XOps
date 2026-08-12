// 요인분석 실행 전 데이터 대기 플레이스홀더 — UI 구조는 항상 노출, 데이터만 실행 후 공개.
export default function PendingData({ running = false, text }) {
  return (
    <p className="pl-pending" role="status">
      <i
        className={"fa-solid " + (running ? "fa-spinner fa-spin" : "fa-hourglass-half")}
        aria-hidden="true"
      ></i>
      {running ? "요인분석 진행 중..." : text ?? "[요인분석 실행] 후 데이터가 표시됩니다."}
    </p>
  );
}
