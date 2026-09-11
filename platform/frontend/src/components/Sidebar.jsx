import { useEffect, useRef, useState } from "react";
import SettingsPanel from "./SettingsPanel.jsx";
import {
  flattenNavTabs,
  readNavGroupExpanded,
  visibleNavTabs,
  writeNavGroupExpanded
} from "../lib/nav.js";

/**
 * 그룹별 펼침 초기값 — 저장된 값을 쓰되, 현재 탭이 그 그룹에 있으면 무조건 펼친다.
 */
function initialExpanded(sections, activeTab) {
  const map = {};
  sections.forEach((section) => {
    if (!section.id) return;
    const stored = readNavGroupExpanded(section.id);
    const hasActive = section.tabs.some((tab) => tab.id === activeTab);
    map[section.id] = hasActive ? true : stored;
  });
  return map;
}

export default function Sidebar({ sidebarRef, sections, activeTab, onSelect, open, hidden }) {
  const tabRefs = useRef([]);
  const tabs = flattenNavTabs(sections);
  const [expanded, setExpanded] = useState(() => initialExpanded(sections, activeTab));
  const visibleTabs = visibleNavTabs(sections, expanded);

  const prevActiveRef = useRef(activeTab);

  // 그룹 밖에서 자식 화면으로 들어올 때만 펼친다. 이미 그룹 안에 있으면 접힌 상태를 유지한다.
  useEffect(() => {
    const prev = prevActiveRef.current;
    prevActiveRef.current = activeTab;
    if (prev === activeTab) return;
    const host = sections.find((section) => section.id && section.tabs.some((tab) => tab.id === activeTab));
    if (!host) return;
    const cameFromInside = host.tabs.some((tab) => tab.id === prev);
    if (cameFromInside) return;
    setExpanded((current) => ({ ...current, [host.id]: true }));
    writeNavGroupExpanded(host.id, true);
  }, [activeTab, sections]);

  const toggleGroup = (sectionId) => {
    setExpanded((prev) => {
      const next = !prev[sectionId];
      writeNavGroupExpanded(sectionId, next);
      return { ...prev, [sectionId]: next };
    });
  };

  const moveTo = (visibleIndex) => {
    const next = visibleTabs[(visibleIndex + visibleTabs.length) % visibleTabs.length];
    const index = tabs.findIndex((tab) => tab.id === next.id);
    onSelect(next.id, { closeDrawer: false });
    requestAnimationFrame(() => tabRefs.current[index]?.focus());
  };

  const handleKeyDown = (event, tabId) => {
    const visibleIndex = visibleTabs.findIndex((tab) => tab.id === tabId);
    if (visibleIndex < 0) return;
    if (event.key === "ArrowDown" || event.key === "ArrowRight") {
      event.preventDefault();
      moveTo(visibleIndex + 1);
    } else if (event.key === "ArrowUp" || event.key === "ArrowLeft") {
      event.preventDefault();
      moveTo(visibleIndex - 1);
    } else if (event.key === "Home") {
      event.preventDefault();
      moveTo(0);
    } else if (event.key === "End") {
      event.preventDefault();
      moveTo(visibleTabs.length - 1);
    }
  };

  const renderTab = (tab, groupLabelId) => {
    const index = tabs.findIndex((entry) => entry.id === tab.id);
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
        aria-describedby={groupLabelId}
        tabIndex={isActive ? 0 : -1}
        className={"nav-item" + (groupLabelId ? " nav-item-nested" : "") + (isActive ? " active" : "")}
        onClick={() => onSelect(tab.id)}
        onKeyDown={(event) => handleKeyDown(event, tab.id)}
      >
        <i className={"fa-solid " + tab.icon} aria-hidden="true"></i>
        <span>{tab.label}</span>
      </button>
    );
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

      <nav className="nav-menu" aria-label="주요 탭">
        {sections.map((section) => {
          if (!section.label) {
            return (
              <div key={section.tabs[0].id} role="tablist" aria-orientation="vertical">
                {section.tabs.map((tab) => renderTab(tab))}
              </div>
            );
          }
          const isOpen = expanded[section.id] !== false;
          const itemsId = `${section.id}-items`;
          return (
            <div key={section.id} className={"nav-group" + (isOpen ? "" : " is-collapsed")}>
              <button
                type="button"
                className="nav-group-toggle"
                id={section.id}
                aria-expanded={isOpen}
                aria-controls={itemsId}
                onClick={() => toggleGroup(section.id)}
              >
                <i className="fa-solid fa-chevron-down nav-group-chevron" aria-hidden="true"></i>
                <span>{section.label}</span>
              </button>
              <div
                id={itemsId}
                className="nav-group-items"
                role="tablist"
                aria-orientation="vertical"
                aria-labelledby={section.id}
                hidden={!isOpen}
              >
                {section.tabs.map((tab) => renderTab(tab, section.id))}
              </div>
            </div>
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
