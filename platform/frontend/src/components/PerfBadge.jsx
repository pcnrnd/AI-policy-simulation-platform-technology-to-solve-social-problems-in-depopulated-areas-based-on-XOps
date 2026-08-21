import { PERF_BUDGET_MS } from "../lib/perf.js";

// 처리 응답속도 표시 배지 — 목표(2000ms) 이내면 teal, 초과면 red.
export default function PerfBadge({ ms, label = "처리 응답속도", budget = PERF_BUDGET_MS }) {
  const measured = typeof ms === "number";
  const ok = measured && ms <= budget;
  const color = !measured ? "var(--text-muted)" : ok ? "var(--accent-teal)" : "var(--accent-red)";
  const text = measured ? `${ms.toFixed(0)}ms` : "측정 중…";

  return (
    <span
      className="system-status"
      style={{
        padding: "2px 10px",
        fontSize: 11,
        backgroundColor: ok ? "rgba(var(--accent-teal-rgb), 0.02)" : "var(--surface-hover)",
        borderColor: "currentColor",
        color
      }}
      title={`성능 목표 ≤ ${budget}ms`}
    >
      <i className="fa-solid fa-stopwatch" aria-hidden="true"></i>
      {label}: {text} {measured && <span className="perf-badge-target">/ ≤{budget}ms</span>}
    </span>
  );
}
