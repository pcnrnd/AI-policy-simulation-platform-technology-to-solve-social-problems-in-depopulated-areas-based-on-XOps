import { useState, useMemo } from "react";
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

const TABS = [
  { id: "tab-overview", label: "종합 대시보드", icon: "fa-chart-line", Component: Overview },
  { id: "tab-mlops-monitor", label: "MLOps 성능 모니터", icon: "fa-gauge-high", Component: MonitorPage },
  { id: "tab-mlops-orch", label: "재학습 오케스트레이션", icon: "fa-diagram-project", Component: OrchestratorPage },
  { id: "tab-dataops", label: "DataOps 스키마 API", icon: "fa-database", Component: DataOpsPage },
  { id: "tab-simulator", label: "정책 시뮬레이터", icon: "fa-map-location-dot", Component: SimulatorPage },
  { id: "tab-reporter", label: "자동화 리포팅", icon: "fa-file-invoice", Component: ReporterPage }
];

export default function App() {
  const [activeTab, setActiveTab] = useState("tab-overview");
  const { ready } = useAppState();

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
    <div className="app-container">
      <Sidebar tabs={TABS} activeTab={activeTab} onSelect={setActiveTab} />
      <main className="main-content">
        <Header title={activeLabel} />
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
