// 요인분석 실행 전 데이터 대기 플레이스홀더 — UI 구조는 항상 노출, 데이터만 실행 후 공개.
export default function PendingData({ running = false, text }) {
  return (
    <p className="pl-pending">
      <i className="fa-solid fa-hourglass-half" aria-hidden="true"></i>
      {running ? "분석 결과를 준비 중입니다." : text ?? "[요인분석 실행] 후 데이터가 표시됩니다."}
    </p>
  );
}

// 데모 표시 OFF에서 시드 파생 값 자리에 놓는 문구. 요소를 숨기지 않고 자리만 지킨다.
// 내부 사정(시드·저장소·엔드포인트)은 사용자 관심사가 아니므로 사유를 덧붙이지 않는다.
export const NO_DEMO_DATA = "데이터 없음";

// 차트 캔버스 위에 겹치는 빈 상태 안내 — 캔버스를 없애지 않고 그 자리에 사유만 남긴다.
// (MonitorPage가 같은 목적으로 쓰는 .monitor-empty-note 스타일을 그대로 쓴다.)
export function ChartEmptyNote({ children }) {
  return <p className="monitor-empty-note">{children}</p>;
}
