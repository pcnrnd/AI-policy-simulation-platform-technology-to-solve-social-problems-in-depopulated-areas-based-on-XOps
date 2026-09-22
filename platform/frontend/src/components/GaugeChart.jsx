import { Doughnut } from "react-chartjs-2";
import { useTheme } from "../context/ThemeContext.jsx";

// 게이지(반원) — Doughnut 차트를 반원으로 렌더하여 0~1 비율을 표시.
// lowerIsBetter: 지연시간처럼 낮을수록 좋은 지표(임계 대비 사용률)에 사용.
// displayText: % 대신 표시할 텍스트(예: "120ms").
// rest: 호출부가 게이지 껍데기에 직접 붙이는 DOM 속성(예: data-values-source).
export default function GaugeChart({
  value,
  label,
  goodThreshold = 0.85,
  lowerIsBetter = false,
  displayText,
  ...rest
}) {
  const { isDark } = useTheme();
  // 측정값이 없으면(데모 표시 OFF·미수집) 0으로 강제해 판정하지 않는다 — value ?? 0 으로 두면
  // 값이 없는 게이지가 "위험"(또는 lowerIsBetter에서 "정상")이라고 단정해 버린다.
  const measured = Number.isFinite(value);
  const ratio = measured ? Math.max(0, Math.min(1, value)) : 0;
  const ok = measured && (lowerIsBetter ? ratio <= goodThreshold : ratio >= goodThreshold);
  const warn = measured && (lowerIsBetter ? ratio <= Math.min(1, goodThreshold + 0.2) : ratio >= 0.6);
  const color = !measured
    ? (isDark ? "rgba(255, 255, 255, 0.10)" : "rgba(15, 23, 42, 0.12)")
    : ok
      ? "rgba(16, 185, 129, 1)"
      : warn
        ? "rgba(245, 158, 11, 1)"
        : "rgba(239, 68, 68, 1)";
  const status = !measured
    ? { label: "판정 없음", icon: "fa-minus", color: "var(--text-muted)" }
    : ok
      ? { label: "정상", icon: "fa-circle-check", color: "var(--accent-teal)" }
      : warn
        ? { label: "주의", icon: "fa-triangle-exclamation", color: "var(--accent-orange)" }
        : { label: "위험", icon: "fa-circle-exclamation", color: "var(--accent-red)" };
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
        cutout: "78%"
      }
    ]
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false }, tooltip: { enabled: false } }
  };

  // 보조 지표용 컴팩트 게이지 — 칸 폭에 끌려 커지지 않도록 폭 상한 + 중앙 정렬
  return (
    <div
      style={{ position: "relative", height: 100, maxWidth: 180, width: "100%", margin: "0 auto" }}
      {...rest}
    >
      <Doughnut data={data} options={options} />
      <div
        style={{
          position: "absolute",
          bottom: 4,
          left: 0,
          width: "100%",
          textAlign: "center"
        }}
      >
        <div className="gauge-value" style={{ fontSize: 20, fontWeight: 700, fontFamily: "'Outfit', sans-serif", color: "var(--text-primary)" }}>
          {displayText ?? (measured ? `${(ratio * 100).toFixed(1)}%` : "–")}
        </div>
        <div style={{ fontSize: 10, color: "var(--text-secondary)" }}>{label}</div>
        <div className="gauge-status" style={{ color: status.color }}>
          <i className={`fa-solid ${status.icon}`} aria-hidden="true"></i> {status.label}
        </div>
      </div>
    </div>
  );
}
