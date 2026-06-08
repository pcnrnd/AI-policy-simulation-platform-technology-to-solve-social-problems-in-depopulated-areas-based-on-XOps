export default function Sidebar({ tabs, activeTab, onSelect }) {
  return (
    <aside className="sidebar">
      <div className="logo-section">
        <div className="logo-icon">🔴</div>
        <div className="logo-text">인구감소 R&D</div>
      </div>

      <nav className="nav-menu" aria-label="주요 탭">
        {tabs.map((tab) => (
          <a
            key={tab.id}
            className={"nav-item" + (activeTab === tab.id ? " active" : "")}
            onClick={() => onSelect(tab.id)}
          >
            <i className={"fa-solid " + tab.icon}></i>
            <span>{tab.label}</span>
          </a>
        ))}
      </nav>

      <div className="sidebar-footer">
        <p>인구감소 R&D R-Center</p>
        <p style={{ fontSize: "9px", marginTop: 4, opacity: 0.7 }}>
          v3.1.0 (React + Vite)
        </p>
      </div>
    </aside>
  );
}
