import { useLayoutEffect, useRef, useState } from "react";
import { ChartEmptyNote } from "../PendingData.jsx";

const H = 150, L = 46, R = 14, T = 20, B = 22;
const BLUE = "var(--accent-blue)";
const num = (v) => v.toLocaleString("ko-KR");
const fmtMan = (v, d) => `${v.toFixed(d)}만`;

function niceTicks(min, max) {
  const raw = (max - min) / 4;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(+v.toFixed(6));
  return out;
}
const pathOf = (pts) => pts.map(([x, y], k) => `${k ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join("");

/** 상자 폭(px) — 그래프를 그 폭에 맞춰 다시 그린다. */
function useWidth() {
  const ref = useRef(null);
  const [width, setWidth] = useState(0);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const read = () => setWidth(el.clientWidth);
    read();
    const ro = new ResizeObserver(read);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, width];
}

/**
 * 전망 그래프 하나 — 관측(실선) → 현재 구분선 → 전망 12분기.
 * 고른 사업이 있으면 반영 후(점선)와 반영 전과의 차이 띠, 끝 글씨 "반영 후 값 · 반영 전보다 +n명".
 * 사업이 없으면 반영 전 전망(점선)과 끝 글씨 "전망 값".
 */
export default function OutlookChart({ chart, guide }) {
  const [ref, width] = useWidth();
  if (chart.empty) return <div className="reco-chart-box" ref={ref}><ChartEmptyNote>{guide}</ChartEmptyNote></div>;
  const W = Math.max(240, width || 240);
  const { obs, base, reflect, labels } = chart;
  const ahead = reflect ?? base;
  const n = labels.length, cur = obs.length - 1;
  const all = [...obs, ...base, ...ahead];
  const lo = Math.min(...all), hi = Math.max(...all), pad = (hi - lo || 0.1) * 0.1;
  const min = lo - pad, max = hi + pad;
  const dec = hi - lo < 1 ? 2 : 1;
  const x = (i) => L + (i * (W - L - R)) / (n - 1);
  const y = (v) => T + ((max - v) * (H - T - B)) / (max - min);
  const up = [[x(cur), y(obs[cur])], ...ahead.map((v, k) => [x(cur + 1 + k), y(v)])];
  const down = [[x(cur), y(obs[cur])], ...base.map((v, k) => [x(cur + 1 + k), y(v)])].reverse();
  const endText = reflect ? `반영 후 ${fmtMan(chart.end, dec)} · 반영 전보다 +${num(chart.increase)}명` : `전망 ${fmtMan(chart.end, dec)}`;
  const ey = y(chart.end);
  const halo = { stroke: "var(--reco-halo)", paintOrder: "stroke", strokeLinejoin: "round" };
  return (
    <div className="reco-chart-box" ref={ref}>
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={`${chart.title} 그래프, ${endText}`}>
        {niceTicks(min, max).map((v) => (
          <g key={v}>
            <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} stroke="var(--reco-grid)" />
            <text x={L - 6} y={y(v) + 3.5} textAnchor="end" fontSize="10.5" fill="var(--text-muted)">{`${+v.toFixed(2)}만`}</text>
          </g>
        ))}
        {[[0, "start"], [cur, "middle"], [n - 1, "end"]].map(([i, a]) => (
          <text key={i} x={x(i)} y={H - 6} textAnchor={a} fontSize="10.5" fill="var(--text-secondary)">{labels[i]}</text>
        ))}
        {reflect && <polygon points={[...up, ...down].map((pt) => pt.map((c) => c.toFixed(1)).join(",")).join(" ")} fill="rgba(var(--accent-blue-rgb), 0.22)" />}
        <line x1={x(cur)} x2={x(cur)} y1={T - 6} y2={H - B} stroke="var(--text-muted)" strokeOpacity="0.5" strokeDasharray="2 3" />
        <text x={x(cur)} y={T - 9} textAnchor="middle" fontSize="11" fontWeight="600" fill="var(--text-secondary)" strokeWidth="3" {...halo}>{`현재 ${fmtMan(obs[cur], dec)}`}</text>
        <path d={pathOf(obs.map((v, i) => [x(i), y(v)]))} fill="none" stroke={BLUE} strokeWidth="2" />
        <path d={pathOf(up)} fill="none" stroke={BLUE} strokeWidth="2" strokeDasharray="5 4" />
        {obs.map((v, i) => <circle key={`o${i}`} cx={x(i)} cy={y(v)} r="2" fill={BLUE}><title>{`${labels[i]} · ${fmtMan(v, dec)}`}</title></circle>)}
        {ahead.map((v, k) => (
          <circle key={`f${k}`} cx={x(cur + 1 + k)} cy={y(v)} r="2" fill={BLUE}>
            <title>{`${labels[cur + 1 + k]} · ${fmtMan(v, dec)}${chart.increases ? ` · +${num(chart.increases[k])}명` : ""}`}</title>
          </circle>
        ))}
        <text x={W - R} y={ey < T + 24 ? ey + 16 : ey - 9} textAnchor="end" fontSize="12" fontWeight="700" fill={BLUE} strokeWidth="4" {...halo}>{endText}</text>
      </svg>
    </div>
  );
}
