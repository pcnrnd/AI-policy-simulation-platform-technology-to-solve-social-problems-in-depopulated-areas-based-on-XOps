import SettingsPanel from "./SettingsPanel.jsx";

export default function Sidebar({ tabs, activeTab, onSelect, open }) {
  return (
    <aside id="app-sidebar" className={"sidebar" + (open ? " open" : "")}>
      <div className="logo-section">
        <div className="logo-icon">🔴</div>
        <div>
          <div className="logo-text">인구감소 R&D</div>
          <div style={{ fontSize: 9, color: "var(--text-muted)", marginTop: 2, letterSpacing: "0.01em" }}>
            예측 시뮬레이션 기반 자원 최적화 플랫폼
          </div>
        </div>
      </div>

      <nav className="nav-menu" aria-label="주요 탭" role="tablist">
        {tabs.map((tab) => {
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              type="button"
              role="tab"
              aria-selected={isActive}
              aria-current={isActive ? "page" : undefined}
              className={"nav-item" + (isActive ? " active" : "")}
              onClick={() => onSelect(tab.id)}
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
        <p style={{ fontSize: "9px", marginTop: 4, opacity: 0.7 }}>
          v3.1.0 (React + Vite)
        </p>
      </div>
    </aside>
  );
}
