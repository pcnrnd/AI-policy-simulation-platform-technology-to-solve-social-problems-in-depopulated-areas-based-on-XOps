import { useMemo } from "react";
import { Bar, Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from "chart.js";
import Card from "../components/Card.jsx";
import { useAppState } from "../context/AppStateContext.jsx";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const AXIS_OPTS = {
  responsive: true,
  maintainAspectRatio: false,
  scales: {
    y: { grid: { color: "rgba(255, 255, 255, 0.05)" }, ticks: { color: "#9ca3af" } },
    x: { grid: { display: false }, ticks: { color: "#9ca3af" } }
  },
  plugins: { legend: { labels: { color: "#f3f4f6" } } }
};

export default function MonitorPage() {
  const { appData, driftInjected, accuracyOverride, f1Override } = useAppState();

  const driftData = useMemo(() => {
    const current = driftInjected
      ? appData.drift_distribution.current_drifted
      : appData.drift_distribution.current_normal;
    return {
      labels: appData.drift_distribution.buckets,
      datasets: [
        {
          label: "참조 분포 (Reference Dataset)",
          data: appData.drift_distribution.reference,
          backgroundColor: "rgba(59, 130, 246, 0.4)",
          borderColor: "rgba(59, 130, 246, 1)",
          borderWidth: 1.5,
          borderRadius: 4
        },
        {
          label: "실시간 유입 데이터 분포",
          data: current,
          backgroundColor: driftInjected ? "rgba(239, 68, 68, 0.65)" : "rgba(16, 185, 129, 0.5)",
          borderColor: driftInjected ? "rgba(239, 68, 68, 1)" : "rgba(16, 185, 129, 1)",
          borderWidth: 1.5,
          borderRadius: 4
        }
      ]
    };
  }, [appData, driftInjected]);

  const metricsData = useMemo(
    () => ({
      labels: appData.metrics_history.timestamps,
      datasets: [
        {
          label: "Accuracy",
          data: appData.metrics_history.accuracy,
          borderColor: "rgba(16, 185, 129, 1)",
          backgroundColor: "rgba(16, 185, 129, 0.05)",
          fill: true,
          tension: 0.35,
          borderWidth: 2
        },
        {
          label: "F1-Score",
          data: appData.metrics_history.f1,
          borderColor: "rgba(59, 130, 246, 1)",
          borderWidth: 2,
          pointStyle: "circle",
          tension: 0.35
        },
        {
          label: "MSE",
          data: appData.metrics_history.mse,
          borderColor: "rgba(239, 68, 68, 1)",
          borderWidth: 1.5,
          borderDash: [5, 5],
          tension: 0.35
        }
      ]
    }),
    [appData]
  );

  const shapData = useMemo(
    () => ({
      labels: appData.shap_features.map((f) => f.feature.split(" (")[0]),
      datasets: [
        {
          label: "SHAP 기여도 기여 지수 (음수일수록 유출 기여)",
          data: appData.shap_features.map((f) => f.value),
          backgroundColor: appData.shap_features.map((f) =>
            f.value > 0 ? "rgba(16, 185, 129, 0.65)" : "rgba(239, 68, 68, 0.65)"
          ),
          borderColor: appData.shap_features.map((f) =>
            f.value > 0 ? "rgba(16, 185, 129, 1)" : "rgba(239, 68, 68, 1)"
          ),
          borderWidth: 1.5,
          borderRadius: 4
        }
      ]
    }),
    [appData]
  );

  const shapOpts = {
    indexAxis: "y",
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: { grid: { color: "rgba(255, 255, 255, 0.05)" }, ticks: { color: "#9ca3af" } },
      y: { grid: { display: false }, ticks: { color: "#f3f4f6", font: { size: 10 } } }
    },
    plugins: { legend: { display: false } }
  };

  const accVal = accuracyOverride !== null ? accuracyOverride.toFixed(3) : "0.892";
  const f1Val = f1Override !== null ? f1Override.toFixed(3) : "0.884";

  const psiVal = driftInjected ? "0.384" : "0.045";
  const psiColor = driftInjected ? "var(--accent-red)" : "var(--accent-teal)";
  const driftLabel = driftInjected ? "위험 (Drift)" : "정상";
  const driftLabelStyle = driftInjected
    ? { backgroundColor: "rgba(239, 68, 68, 0.15)", color: "var(--accent-red)" }
    : { backgroundColor: "rgba(16, 185, 129, 0.1)", color: "var(--accent-teal)" };

  const outlierRows = driftInjected
    ? [
        { time: "13:02", target: "의성군 데이터 (인구비율)", z: "3.45", outlier: true },
        { time: "13:01", target: "고흥군 데이터 (일자리수)", z: "2.89", outlier: true },
        { time: "12:34", target: "의성군 데이터", z: "1.24", outlier: false }
      ]
    : [
        { time: "12:34", target: "의성군 데이터", z: "1.24", outlier: false },
        { time: "11:20", target: "고흥군 데이터", z: "0.98", outlier: false },
        { time: "10:05", target: "봉화군 데이터", z: "1.67", outlier: false }
      ];

  const outlierCount = driftInjected ? "3건" : "0건";
  const outlierCountColor = driftInjected ? "var(--accent-red)" : "var(--text-primary)";

  return (
    <>
      <div className="grid-cols-3" style={{ marginBottom: 24 }}>
        <div className="card" style={{ padding: 18 }}>
          <div className="stat-label">Model Accuracy / F1-Score</div>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "baseline",
              marginTop: 10
            }}
          >
            <span className="stat-value" style={{ fontSize: 32 }}>
              {accVal}
            </span>
            <span className="trend-up">F1: {f1Val}</span>
          </div>
          <p style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 8 }}>
            6개 지표 평가지표 실시간 자동 집계
          </p>
        </div>

        <div className={"card" + (driftInjected ? " glow-red" : "")} style={{ padding: 18 }}>
          <div className="stat-label">Data Drift Status (PSI)</div>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "baseline",
              marginTop: 10
            }}
          >
            <span className="stat-value" style={{ fontSize: 32, color: psiColor }}>
              {psiVal}
            </span>
            <span
              className="system-status"
              style={{ padding: "2px 8px", fontSize: 11, ...driftLabelStyle }}
            >
              {driftLabel}
            </span>
          </div>
          <p style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 8 }}>
            임계치 PSI {">"} 0.2 초과 시 자동 Alert 트리거
          </p>
        </div>

        <div className="card" style={{ padding: 18 }}>
          <div className="stat-label">Outlier Detection (Z-Score)</div>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "baseline",
              marginTop: 10
            }}
          >
            <span className="stat-value" style={{ fontSize: 32, color: outlierCountColor }}>
              {outlierCount}
            </span>
            <span
              className="system-status"
              style={{
                padding: "2px 8px",
                fontSize: 11,
                backgroundColor: driftInjected ? "rgba(239, 68, 68, 0.1)" : "rgba(16, 185, 129, 0.1)",
                color: driftInjected ? "var(--accent-red)" : "var(--accent-teal)"
              }}
            >
              {driftInjected ? "경고" : "정상"}
            </span>
          </div>
          <p style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 8 }}>
            IQR 및 Z-score 기반 다차원 이상치 필터링
          </p>
        </div>
      </div>

      <div className="grid-details-split">
        <Card title="데이터 분포 변화 시각화 (참조 vs 최근유입)" icon="fa-chart-area">
          <div style={{ position: "relative", height: 320, width: "100%" }}>
            <Bar data={driftData} options={AXIS_OPTS} />
          </div>
        </Card>

        <Card title="이상값 검출 로그" icon="fa-filter">
          <div style={{ maxHeight: 320, overflowY: "auto" }}>
            <table style={{ width: "100%" }}>
              <thead>
                <tr>
                  <th>시간</th>
                  <th>지자체</th>
                  <th>Z-score</th>
                  <th>상태</th>
                </tr>
              </thead>
              <tbody>
                {outlierRows.map((row, idx) => (
                  <tr key={idx}>
                    <td
                      style={{
                        color: row.outlier ? "var(--accent-red)" : "var(--text-secondary)"
                      }}
                    >
                      {row.time}
                    </td>
                    <td>{row.target}</td>
                    <td>{row.z}</td>
                    <td>
                      {row.outlier ? (
                        <span className="outlier-tag">Outlier</span>
                      ) : (
                        <span
                          className="system-status"
                          style={{
                            padding: "1px 6px",
                            fontSize: 10,
                            backgroundColor: "rgba(16, 185, 129, 0.1)",
                            color: "var(--accent-teal)"
                          }}
                        >
                          Normal
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>

      <div className="grid-cols-2">
        <Card title="MLOps 6대 핵심 평가지표 실시간 모니터링" icon="fa-chart-column">
          <div style={{ position: "relative", height: 280, width: "100%" }}>
            <Line data={metricsData} options={AXIS_OPTS} />
          </div>
        </Card>

        <Card title="SHAP 기반 인구 유출 기여 특징 중요도 분석" icon="fa-brain">
          <div style={{ position: "relative", height: 280, width: "100%" }}>
            <Bar data={shapData} options={shapOpts} />
          </div>
        </Card>
      </div>
    </>
  );
}
