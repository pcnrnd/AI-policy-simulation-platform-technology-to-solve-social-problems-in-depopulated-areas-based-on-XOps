import { useState, useMemo, useCallback, useEffect } from "react";
import Sidebar from "./components/Sidebar.jsx";
import Header from "./components/Header.jsx";
import AlertPopupContainer from "./components/AlertPopup.jsx";
import Overview from "./pages/Overview.jsx";
import MonitorPage from "./pages/MonitorPage.jsx";
import OrchestratorPage from "./pages/OrchestratorPage.jsx";
import DataOpsPage from "./pages/DataOpsPage.jsx";
import SimulatorPage from "./pages/SimulatorPage.jsx";
import ReporterPage from "./pages/ReporterPage.jsx";
import { useAppState } from "./context/AppStateContext.jsx";
import { useResizableSidebar } from "./hooks/useResizableSidebar.js";

const TABS = [
  { id: "tab-overview", label: "종합 대시보드", icon: "fa-chart-line", Component: Overview },
  { id: "tab-mlops-monitor", label: "MLOps 성능 모니터", icon: "fa-gauge-high", Component: MonitorPage },
  { id: "tab-mlops-orch", label: "오케스트레이터", icon: "fa-diagram-project", Component: OrchestratorPage },
  { id: "tab-dataops", label: "DataOps", icon: "fa-database", Component: DataOpsPage },
  { id: "tab-simulator", label: "정책 시뮬레이터 & 추천", icon: "fa-map-location-dot", Component: SimulatorPage },
  { id: "tab-reporter", label: "자동화 리포팅", icon: "fa-file-invoice", Component: ReporterPage }
];

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const { ready, activeTab, setActiveTab } = useAppState();
  const { width, resizing, startResize, resizeBy, resetWidth } = useResizableSidebar();

  const handleResizerKeyDown = useCallback(
    (event) => {
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        resizeBy(-16);
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        resizeBy(16);
      }
    },
    [resizeBy]
  );

  const handleSelect = useCallback((id) => {
    setActiveTab(id);
    setSidebarOpen(false); // 모바일: 탭 선택 시 드로어 닫기
  }, []);

  const closeSidebar = useCallback(() => setSidebarOpen(false), []);
  const toggleSidebar = useCallback(() => setSidebarOpen((prev) => !prev), []);

  // 모바일 드로어 열림 시 ESC 로 닫기
  useEffect(() => {
    if (!sidebarOpen) return undefined;
    const onKeyDown = (event) => {
      if (event.key === "Escape") closeSidebar();
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [sidebarOpen, closeSidebar]);

  const ActiveComponent = useMemo(
    () => TABS.find((t) => t.id === activeTab)?.Component ?? Overview,
    [activeTab]
  );

  const activeLabel = useMemo(
    () => TABS.find((t) => t.id === activeTab)?.label ?? "",
    [activeTab]
  );

  if (!ready) {
    return (
      <div className="loading-screen">
        <div>R&D 데이터 자산을 로딩 중...</div>
      </div>
    );
  }

  return (
    <div
      className={"app-container" + (resizing ? " resizing" : "")}
      style={{ "--sidebar-width": `${width}px` }}
    >
      <Sidebar
        tabs={TABS}
        activeTab={activeTab}
        onSelect={handleSelect}
        open={sidebarOpen}
      />
      <div
        className={"sidebar-resizer" + (resizing ? " active" : "")}
        onPointerDown={startResize}
        onKeyDown={handleResizerKeyDown}
        onDoubleClick={resetWidth}
        role="separator"
        aria-orientation="vertical"
        aria-valuenow={width}
        aria-valuemin={200}
        aria-valuemax={420}
        aria-label="사이드바 너비 조절 (드래그 또는 좌우 방향키, 더블클릭 시 초기화)"
        tabIndex={0}
      />
      <div
        className={"sidebar-backdrop" + (sidebarOpen ? " visible" : "")}
        onClick={closeSidebar}
        aria-hidden="true"
      />
      <main className="main-content">
        <Header
          title={activeLabel}
          onToggleSidebar={toggleSidebar}
          sidebarOpen={sidebarOpen}
        />
        <div className="content-body">
          <section className="page-tab active" key={activeTab}>
            <ActiveComponent />
          </section>
        </div>
      </main>
      <AlertPopupContainer />
    </div>
  );
}
