// 요인분석 실행 전 데이터 대기 플레이스홀더 — UI 구조는 항상 노출, 데이터만 실행 후 공개.
export default function PendingData({ running = false, text }) {
  return (
    <p className="pl-pending">
      <i className="fa-solid fa-hourglass-half" aria-hidden="true"></i>
      {running ? "분석 결과를 준비 중입니다." : text ?? "[요인분석 실행] 후 데이터가 표시됩니다."}
    </p>
  );
}

// 데모 표시 OFF에서 시드 파생 값 자리에 놓는 안내. 요소를 숨기지 않고 자리만 지킨다.
// 이 값들은 예시 데이터로만 채워지므로 값을 보는 방법(데모 표시)만 알린다.
export const DEMO_ONLY_HINT = "설정에서 '데모 데이터 표시'를 켜면 예시 값이 표시됩니다.";

// 차트 캔버스 위에 겹치는 빈 상태 안내 — 캔버스를 없애지 않고 그 자리에 사유만 남긴다.
// (MonitorPage가 같은 목적으로 쓰는 .monitor-empty-note 스타일을 그대로 쓴다.)
export function ChartEmptyNote({ children }) {
  return <p className="monitor-empty-note">{children}</p>;
}
