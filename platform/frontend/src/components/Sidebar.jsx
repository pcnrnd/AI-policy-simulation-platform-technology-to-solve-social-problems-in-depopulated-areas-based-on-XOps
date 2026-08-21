import { useRef } from "react";
import SettingsPanel from "./SettingsPanel.jsx";

export default function Sidebar({ sidebarRef, tabs, activeTab, onSelect, open, hidden }) {
  const tabRefs = useRef([]);

  const moveTo = (index) => {
    const next = (index + tabs.length) % tabs.length;
    onSelect(tabs[next].id, { closeDrawer: false });
    requestAnimationFrame(() => tabRefs.current[next]?.focus());
  };

  const handleKeyDown = (event, index) => {
    if (event.key === "ArrowDown" || event.key === "ArrowRight") {
      event.preventDefault();
      moveTo(index + 1);
    } else if (event.key === "ArrowUp" || event.key === "ArrowLeft") {
      event.preventDefault();
      moveTo(index - 1);
    } else if (event.key === "Home") {
      event.preventDefault();
      moveTo(0);
    } else if (event.key === "End") {
      event.preventDefault();
      moveTo(tabs.length - 1);
    }
  };

  return (
    <aside
      ref={sidebarRef}
      id="app-sidebar"
      className={"sidebar" + (open ? " open" : "")}
      role={open ? "dialog" : undefined}
      aria-modal={open ? "true" : undefined}
      aria-label="주요 내비게이션"
      aria-hidden={hidden || undefined}
      inert={hidden ? "" : undefined}
    >
      <div className="logo-section">
        <div className="logo-icon">🔴</div>
        <div>
          <div className="logo-text">인구감소 R&D</div>
          <div style={{ fontSize: 9, color: "var(--text-muted)", marginTop: 2, letterSpacing: "0.01em" }}>
            예측 시뮬레이션 기반 자원 최적화 플랫폼
          </div>
        </div>
      </div>

      <nav className="nav-menu" aria-label="주요 탭" role="tablist" aria-orientation="vertical">
        {tabs.map((tab, index) => {
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              ref={(node) => {
                tabRefs.current[index] = node;
              }}
              id={`nav-${tab.id}`}
              type="button"
              role="tab"
              aria-selected={isActive}
              aria-controls={`${tab.id}-panel`}
              aria-current={isActive ? "page" : undefined}
              tabIndex={isActive ? 0 : -1}
              className={"nav-item" + (isActive ? " active" : "")}
              onClick={() => onSelect(tab.id)}
              onKeyDown={(event) => handleKeyDown(event, index)}
            >
              <i className={"fa-solid " + tab.icon} aria-hidden="true"></i>
              <span>{tab.label}</span>
            </button>
          );
        })}
      </nav>

      <SettingsPanel />

      <div className="sidebar-footer">
        <p>인구감소 R&D R-Center</p>
        <p className="sidebar-version">
          v3.1.0 (React + Vite)
        </p>
      </div>
    </aside>
  );
}
