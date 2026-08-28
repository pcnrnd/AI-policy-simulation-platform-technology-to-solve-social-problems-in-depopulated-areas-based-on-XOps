import { useEffect, useRef, useState } from "react";
import { useAppState } from "../context/AppStateContext.jsx";

const SEVERITY_ICONS = {
  warn: { icon: "fa-triangle-exclamation", color: "var(--accent-red)" },
  success: { icon: "fa-circle-check", color: "var(--accent-teal)" },
  info: { icon: "fa-circle-info", color: "var(--accent-blue)" }
};

export default function Header({ title, onToggleSidebar, sidebarOpen, menuButtonRef }) {
  const {
    driftInjected,
    pipelineRunning,
    notifications,
    unreadCount,
    markNotificationsRead,
    mockDataVisible
  } = useAppState();

  // 알림 벨 드롭다운 — 열 때 읽음 처리, 외부 클릭 시 닫힘
  const [notifOpen, setNotifOpen] = useState(false);
  const bellRef = useRef(null);
  const notifRef = useRef(null);

  // 목업 표시를 끄면 벨 드롭다운도 함께 접는다 — 다시 켤 때 열린 채로 되살아나지 않도록.
  useEffect(() => {
    if (!mockDataVisible) setNotifOpen(false);
  }, [mockDataVisible]);

  useEffect(() => {
    if (!notifOpen) return undefined;
    const frame = requestAnimationFrame(() => notifRef.current?.focus());
    const onOutside = (e) => {
      if (bellRef.current && !bellRef.current.contains(e.target)) setNotifOpen(false);
    };
    const onKeyDown = (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        setNotifOpen(false);
        bellRef.current?.querySelector("button")?.focus();
      }
    };
    document.addEventListener("mousedown", onOutside);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      cancelAnimationFrame(frame);
      document.removeEventListener("mousedown", onOutside);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [notifOpen]);

  const toggleNotif = () => {
    setNotifOpen((open) => {
      if (!open) markNotificationsRead();
      return !open;
    });
  };

  let statusClass = "system-status";
  let statusText = "모델 모니터링 활성 (정상)";
  if (pipelineRunning) {
    statusClass = "system-status retraining";
    statusText = "자동 재학습 및 배포 파이프라인 수행 중...";
  } else if (driftInjected) {
    // PSI 수치는 모니터 화면의 실계산값(/monitoring/drift)이 단독 표기한다.
    // 헤더가 별도 상수를 표시하면 실제 판정값과 어긋나므로 정성 상태만 전달한다.
    statusClass = "system-status drift-alert";
    statusText = "이상 현상: 데이터 드리프트 감지 (임계 초과)";
  }

  if (!mockDataVisible) {
    statusClass = "system-status mock-data-visibility-status";
    statusText = "";
  }

  return (
    <header className="main-header">
      <div className="header-left">
        <button
          ref={menuButtonRef}
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
        <div className={statusClass} role="status" aria-live="polite">
          <span className="status-indicator" aria-hidden="true"></span>
          <span>{statusText}</span>
        </div>
        <div className="alert-badge-container" ref={bellRef}>
          <button
            className="alert-icon-btn"
            aria-label={mockDataVisible ? `알림 (읽지 않음 ${unreadCount}건)` : "알림"}
            aria-expanded={mockDataVisible && notifOpen}
            aria-controls="recent-notifications"
            onClick={toggleNotif}
            disabled={!mockDataVisible}
          >
            <i className="fa-solid fa-bell" aria-hidden="true"></i>
          </button>
          {mockDataVisible && unreadCount > 0 && <div className="alert-dot" aria-hidden="true"></div>}
          {notifOpen && (
            <div
              ref={notifRef}
              id="recent-notifications"
              className="notif-dropdown"
              role="region"
              aria-label="최근 알림"
              tabIndex="-1"
            >
              <div className="notif-head">최근 알림</div>
              {notifications.length === 0 ? (
                <div className="notif-empty">새 알림이 없습니다.</div>
              ) : (
                notifications.map((n) => {
                  const sev = SEVERITY_ICONS[n.severity] ?? SEVERITY_ICONS.info;
                  return (
                    <div key={n.id} className="notif-item">
                      <i
                        className={`fa-solid ${sev.icon}`}
                        style={{ color: sev.color }}
                        aria-hidden="true"
                      ></i>
                      <div className="notif-body">
                        <div className="notif-title">{n.title}</div>
                        <div className="notif-msg">{n.message}</div>
                      </div>
                      <span className="notif-time">{n.time}</span>
                    </div>
                  );
                })
              )}
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
