import { Doughnut } from "react-chartjs-2";
import { useTheme } from "../context/ThemeContext.jsx";

// 게이지(반원) — Doughnut 차트를 반원으로 렌더하여 0~1 비율을 표시.
// lowerIsBetter: 지연시간처럼 낮을수록 좋은 지표(임계 대비 사용률)에 사용.
// displayText: % 대신 표시할 텍스트(예: "120ms").
export default function GaugeChart({ value, label, goodThreshold = 0.85, lowerIsBetter = false, displayText }) {
  const { isDark } = useTheme();
  const ratio = Math.max(0, Math.min(1, value));
  const ok = lowerIsBetter ? ratio <= goodThreshold : ratio >= goodThreshold;
  const warn = lowerIsBetter ? ratio <= Math.min(1, goodThreshold + 0.2) : ratio >= 0.6;
  const color = ok ? "rgba(16, 185, 129, 1)" : warn ? "rgba(245, 158, 11, 1)" : "rgba(239, 68, 68, 1)";
  const trackColor = isDark ? "rgba(255, 255, 255, 0.06)" : "rgba(15, 23, 42, 0.08)";

  const data = {
    labels: [label, "잔여"],
    datasets: [
      {
        data: [ratio, 1 - ratio],
        backgroundColor: [color, trackColor],
        borderColor: "transparent",
        circumference: 180,
        rotation: 270,
        cutout: "72%"
      }
    ]
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false }, tooltip: { enabled: false } }
  };

  return (
    <div style={{ position: "relative", height: 150, width: "100%" }}>
      <Doughnut data={data} options={options} />
      <div
        style={{
          position: "absolute",
          bottom: 8,
          left: 0,
          width: "100%",
          textAlign: "center"
        }}
      >
        <div style={{ fontSize: 28, fontWeight: 700, fontFamily: "'Outfit', sans-serif", color }}>
          {displayText ?? `${(ratio * 100).toFixed(1)}%`}
        </div>
        <div style={{ fontSize: 11, color: "var(--text-secondary)" }}>{label}</div>
      </div>
    </div>
  );
}
