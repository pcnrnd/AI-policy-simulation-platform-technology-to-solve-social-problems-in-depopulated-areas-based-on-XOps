import { useState, useMemo, useCallback, useEffect, useRef } from "react";
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
  const [isMobile, setIsMobile] = useState(() => window.matchMedia("(max-width: 768px)").matches);
  const sidebarRef = useRef(null);
  const resizerRef = useRef(null);
  const menuButtonRef = useRef(null);
  const mainRef = useRef(null);
  const alertsRef = useRef(null);
  const contentBodyRef = useRef(null);
  const handledTabFocusRequestRef = useRef(0);
  const { ready, activeTab, setActiveTab, tabFocusRequest, mockDataVisible } = useAppState();
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

  const handleSelect = useCallback((id, { closeDrawer = true } = {}) => {
    setActiveTab(id);
    if (closeDrawer) setSidebarOpen(false); // 모바일: 실행(클릭/Enter/Space) 시 드로어 닫기
  }, []);

  const closeSidebar = useCallback(() => setSidebarOpen(false), []);
  const toggleSidebar = useCallback(() => setSidebarOpen((prev) => !prev), []);

  useEffect(() => {
    const media = window.matchMedia("(max-width: 768px)");
    const onChange = (event) => {
      const focusWasInSidebar =
        sidebarRef.current?.contains(document.activeElement) || document.activeElement === resizerRef.current;
      setIsMobile(event.matches);
      if (!event.matches) setSidebarOpen(false);
      if (event.matches && focusWasInSidebar) {
        requestAnimationFrame(() => menuButtonRef.current?.focus());
      }
    };
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, []);

  // 모바일 드로어: 초점 진입·내부 순환·ESC 닫기·배경 비활성화·트리거 복귀
  useEffect(() => {
    if (!sidebarOpen || !isMobile) return undefined;
    const sidebar = sidebarRef.current;
    const main = mainRef.current;
    const alerts = alertsRef.current;
    const focusableSelector =
      'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])';

    main?.setAttribute("inert", "");
    main?.setAttribute("aria-hidden", "true");
    alerts?.setAttribute("inert", "");
    alerts?.setAttribute("aria-hidden", "true");
    document.body.classList.add("drawer-open");
    const frame = requestAnimationFrame(() => {
      sidebar?.querySelector('[role="tab"][aria-selected="true"]')?.focus();
    });

    const onKeyDown = (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        closeSidebar();
        return;
      }
      if (event.key !== "Tab") return;
      const focusable = [...(sidebar?.querySelectorAll(focusableSelector) ?? [])];
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (!sidebar?.contains(document.activeElement)) {
        event.preventDefault();
        first.focus();
      } else if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => {
      cancelAnimationFrame(frame);
      document.removeEventListener("keydown", onKeyDown);
      main?.removeAttribute("inert");
      main?.removeAttribute("aria-hidden");
      alerts?.removeAttribute("inert");
      alerts?.removeAttribute("aria-hidden");
      document.body.classList.remove("drawer-open");
      requestAnimationFrame(() => menuButtonRef.current?.focus());
    };
  }, [sidebarOpen, isMobile, closeSidebar]);

  const ActiveComponent = useMemo(
    () => TABS.find((t) => t.id === activeTab)?.Component ?? Overview,
    [activeTab]
  );

  const activeLabel = useMemo(
    () => TABS.find((t) => t.id === activeTab)?.label ?? "",
    [activeTab]
  );

  // 탭을 바꾸면 본문 스크롤을 맨 위로 되돌린다. 스크롤 위치는 이전 탭의 것이라,
  // 그대로 두면 새 탭의 헤더·툴바·상태 안내가 화면 밖에 남는다(§10 UI-10 "같은 맥락에서 안내").
  // 초점 이동(아래 effect)은 rAF 뒤에 실행되므로 이 초기화가 먼저 반영된다.
  useEffect(() => {
    const contentBody = contentBodyRef.current;
    if (!contentBody) return;
    contentBody.scrollTop = 0;
    contentBody.scrollLeft = 0;
  }, [activeTab]);

  useEffect(() => {
    if (!tabFocusRequest || handledTabFocusRequestRef.current === tabFocusRequest) return;
    handledTabFocusRequestRef.current = tabFocusRequest;
    requestAnimationFrame(() => document.getElementById(`${activeTab}-panel`)?.focus());
  }, [activeTab, tabFocusRequest]);

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
        sidebarRef={sidebarRef}
        tabs={TABS}
        activeTab={activeTab}
        onSelect={handleSelect}
        open={sidebarOpen}
        hidden={isMobile && !sidebarOpen}
      />
      <div
        ref={resizerRef}
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
      <main
        ref={mainRef}
        className={"main-content" + (mockDataVisible ? "" : " mock-values-hidden")}
      >
        <Header
          title={activeLabel}
          onToggleSidebar={toggleSidebar}
          sidebarOpen={sidebarOpen}
          menuButtonRef={menuButtonRef}
        />
        <div className="content-body" ref={contentBodyRef}>
          {TABS.map((tab) => {
            const isActive = tab.id === activeTab;
            return (
              <section
                id={`${tab.id}-panel`}
                className={`page-tab${isActive ? " active" : ""}`}
                key={tab.id}
                role="tabpanel"
                aria-labelledby={`nav-${tab.id}`}
                tabIndex={isActive ? 0 : -1}
                hidden={!isActive}
              >
                {isActive && <ActiveComponent />}
              </section>
            );
          })}
        </div>
      </main>
      <div ref={alertsRef} className={mockDataVisible ? "" : "mock-values-hidden"}>
        <AlertPopupContainer />
      </div>
    </div>
  );
}
