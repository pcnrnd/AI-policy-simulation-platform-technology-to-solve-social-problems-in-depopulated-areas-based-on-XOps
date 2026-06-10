import { useAppState } from "../context/AppStateContext.jsx";

export default function Header({ title, onToggleSidebar, sidebarOpen }) {
  const { driftInjected, pipelineRunning } = useAppState();

  let statusClass = "system-status";
  let statusText = "모델 모니터링 활성 (정상)";
  if (pipelineRunning) {
    statusClass = "system-status retraining";
    statusText = "자동 재학습 및 배포 파이프라인 수행 중...";
  } else if (driftInjected) {
    statusClass = "system-status drift-alert";
    statusText = "이상 현상: 데이터 드리프트 감지 (PSI: 0.384)";
  }

  return (
    <header className="main-header">
      <div className="header-left">
        <button
          type="button"
          className="sidebar-toggle-btn"
          onClick={onToggleSidebar}
          aria-label={sidebarOpen ? "메뉴 닫기" : "메뉴 열기"}
          aria-expanded={sidebarOpen}
          aria-controls="app-sidebar"
        >
          <i className="fa-solid fa-bars" aria-hidden="true"></i>
        </button>
        <div className="header-title-area">
          <h2>{title}</h2>
        </div>
      </div>
      <div className="header-controls">
        <div className={statusClass}>
          <span className="status-indicator"></span>
          <span>{statusText}</span>
        </div>
        <div className="alert-badge-container">
          <button className="alert-icon-btn" aria-label="알림">
            <i className="fa-solid fa-bell"></i>
          </button>
          {driftInjected && <div className="alert-dot"></div>}
        </div>
      </div>
    </header>
  );
}
