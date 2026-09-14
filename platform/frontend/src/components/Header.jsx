import { useEffect, useRef, useState } from "react";
import { useAppState } from "../context/AppStateContext.jsx";

const SEVERITY_ICONS = {
  warn: { icon: "fa-triangle-exclamation", color: "var(--accent-red)" },
  success: { icon: "fa-circle-check", color: "var(--accent-teal)" },
  info: { icon: "fa-circle-info", color: "var(--accent-blue)" }
};

// 모니터 화면의 실제 수집 판정(monitorCollectStatus)을 그대로 옮긴 색이다 — 기존 .system-status
// 팔레트 재사용(성공=teal 기본값, 실패=orange 경고색, 미수집=회색)이며 새 색을 만들지 않는다.
const COLLECT_CHIP = {
  ok: { icon: "fa-satellite-dish", label: "수집 성공", color: "var(--accent-teal)", bg: "rgba(var(--accent-teal-rgb), 0.02)" },
  fail: {
    icon: "fa-triangle-exclamation",
    label: "수집 실패",
    color: "var(--accent-orange)",
    bg: "rgba(var(--accent-orange-rgb), 0.02)"
  },
  unknown: { icon: "fa-satellite-dish", label: "미수집", color: "var(--text-muted)", bg: "rgba(128, 138, 154, 0.12)" }
};

export default function Header({ title, onToggleSidebar, sidebarOpen, menuButtonRef }) {
  const {
    driftInjected,
    pipelineRunning,
    monitorCollectStatus,
    notifications,
    unreadCount,
    markNotificationsRead,
    mockDataVisible
  } = useAppState();

  // 알림 벨 드롭다운 — 열 때 읽음 처리, 외부 클릭 시 닫힘
  const [notifOpen, setNotifOpen] = useState(false);
  const bellRef = useRef(null);
  const notifRef = useRef(null);

  // 데모 표시를 끄면 벨 드롭다운도 함께 접는다 — 다시 켤 때 열린 채로 되살아나지 않도록.
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
  // '(정상)' 판정은 실제 수집 결과와 무관하게 항상 붙어 있어 수집 실패 안내와 동시에 표시되면
  // '정상'이 실수집 성공처럼 읽혔다(QA-2026-09-11 #5). 활성 여부만 남기고, 실제 판정은 옆 칩으로 분리한다.
  let statusText = "모델 모니터링 활성";
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
    // 데모 기반 상태 문구(정상·드리프트·재학습)는 감추되 빈칸으로 두지 않는다.
    // 보이는 라벨은 OFF만, 보조기술에는 데모 데이터임을 남긴다.
    statusClass = "system-status mock-data-visibility-status";
    statusText = "OFF";
  }

  // 활성 문구와 분리된 별도 칩 — 모니터 화면과 같은 API 응답 기준으로 성공/실패/미수집만 알린다.
  // OFF에서는 위 활성 문구도 "OFF" 하나로 접히므로 칩도 함께 감춘다(데모 표시 계약과 일관).
  const collectChip = mockDataVisible ? COLLECT_CHIP[monitorCollectStatus] ?? COLLECT_CHIP.unknown : null;

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
        <div
          className={statusClass}
          role="status"
          aria-live="polite"
          aria-label={mockDataVisible ? undefined : "데모 데이터 OFF"}
        >
          <span className="status-indicator" aria-hidden="true"></span>
          <span>{statusText}</span>
        </div>
        {collectChip && (
          <span
            className="system-status"
            role="status"
            aria-live="polite"
            style={{
              padding: "2px 10px",
              fontSize: 11,
              color: collectChip.color,
              backgroundColor: collectChip.bg,
              borderColor: "currentColor"
            }}
          >
            <i className={`fa-solid ${collectChip.icon}`} aria-hidden="true"></i>
            <span>{collectChip.label}</span>
          </span>
        )}
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
