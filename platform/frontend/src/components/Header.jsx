import { useEffect, useRef, useState } from "react";
import { useAppState } from "../context/AppStateContext.jsx";

const SEVERITY_ICONS = {
  warn: { icon: "fa-triangle-exclamation", color: "var(--accent-red)" },
  success: { icon: "fa-circle-check", color: "var(--accent-teal)" },
  info: { icon: "fa-circle-info", color: "var(--accent-blue)" }
};

export default function Header({ title, onToggleSidebar, sidebarOpen }) {
  const { driftInjected, pipelineRunning, notifications, unreadCount, markNotificationsRead } =
    useAppState();

  // 알림 벨 드롭다운 — 열 때 읽음 처리, 외부 클릭 시 닫힘
  const [notifOpen, setNotifOpen] = useState(false);
  const bellRef = useRef(null);

  useEffect(() => {
    if (!notifOpen) return undefined;
    const onOutside = (e) => {
      if (bellRef.current && !bellRef.current.contains(e.target)) setNotifOpen(false);
    };
    document.addEventListener("mousedown", onOutside);
    return () => document.removeEventListener("mousedown", onOutside);
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
        <div className="alert-badge-container" ref={bellRef}>
          <button
            className="alert-icon-btn"
            aria-label={`알림 (읽지 않음 ${unreadCount}건)`}
            aria-expanded={notifOpen}
            aria-haspopup="true"
            onClick={toggleNotif}
          >
            <i className="fa-solid fa-bell"></i>
          </button>
          {unreadCount > 0 && <div className="alert-dot"></div>}
          {notifOpen && (
            <div className="notif-dropdown" role="region" aria-label="최근 알림">
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
