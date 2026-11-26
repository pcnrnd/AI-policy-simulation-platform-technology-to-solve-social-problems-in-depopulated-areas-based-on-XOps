import { useMemo } from "react";
import { Bar } from "react-chartjs-2";
import { BarElement, CategoryScale, Chart, Legend, LinearScale, Tooltip } from "chart.js";
import { maxRate, scenarioState, typeResult, usageChanges } from "../lib/scenario.js";
import { radiusText } from "../lib/spatial.js";
import { fmt, km } from "./format.js";

Chart.register(BarElement, CategoryScale, LinearScale, Tooltip, Legend);

export const ratePct = (r) => (r == null ? "—" : `${Math.round(r * 100)}%`);

/** 처음(동작 0) ↔ 지금 비교 행: ◉ 유형 접근·사용률과 정주·생활 유입. 선택 상세 미리 보기도 같은 행을 쓴다. */
export function compareRows(type, env, before, after, stBefore, stAfter) {
  const changed = usageChanges(before.usage, after.usage);
  const ids = changed.map((c) => c.id);
  return [
    [`${radiusText(env)} 밖 대상 인구`, `${fmt(before.summary.total - before.summary.covered)}명`, `${fmt(after.summary.total - after.summary.covered)}명`],
    ["부족 읍면동", `${before.summary.shortage.length}곳`, `${after.summary.shortage.length}곳`],
    ["평균 거리", km(before.summary.meanDistKm), km(after.summary.meanDistKm)],
    ["영향 시설 최고 사용률", ratePct(maxRate(before.usage, ids)), ratePct(maxRate(after.usage, ids))],
    ["정주인구 유입", `${fmt(stBefore.settledTotal)}명`, `${fmt(stAfter.settledTotal)}명`],
    ["생활인구 유입", `${fmt(stBefore.livingTotal)}명`, `${fmt(stAfter.livingTotal)}명`]
  ];
}

export function CompareTable({ head, rows }) {
  return (
    <table className="compare">
      <thead><tr>{head.map((h) => <th key={h}>{h}</th>)}</tr></thead>
      <tbody>{rows.map(([k, ...v], r) => <tr key={r}><td>{k}</td>{v.map((x, i) => <td key={i}>{x}</td>)}</tr>)}</tbody>
    </table>
  );
}

/** 사용률 전·후 가로 막대(바뀐 시설, 많이 바뀐 순 최대 8곳). */
export function UsageChart({ changes, nameOf, color, labels }) {
  const top = changes.slice(0, 8);
  if (!top.length) return null;
  const data = {
    labels: top.map((c) => nameOf(c.id)),
    datasets: [
      { label: labels[0], data: top.map((c) => (c.before == null ? 0 : Math.round(c.before * 100))), backgroundColor: "#b8b2a6" },
      { label: labels[1], data: top.map((c) => (c.after == null ? 0 : Math.round(c.after * 100))), backgroundColor: color }
    ]
  };
  const options = {
    indexAxis: "y", responsive: true, maintainAspectRatio: false, animation: { duration: 250 },
    plugins: { legend: { position: "bottom", labels: { boxWidth: 12 } }, tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${c.parsed.x}%` } } },
    scales: { x: { beginAtZero: true, title: { display: true, text: "사용률(%)" } }, y: { ticks: { font: { size: 11 } } } }
  };
  // 범례·축 제목 자리(약 90px) + 시설마다 28px
  return <div className="chart-box" style={{ height: 90 + top.length * 28 }}><Bar data={data} options={options} aria-label="시설 사용률 전후 막대그래프" /></div>;
}

/** "추가 시설" 탭: 추가·전환 목록(지우기·되돌리기)과 처음 ↔ 지금 누적 효과(◉ 유형, 분석 기준이 없으면 목록만). */
export function ScenarioTab({ scenario, type, env, result, selection, onSelectFacility, onRemoveAction }) {
  const { actions, baseFacilities, units, settings, capacities, state } = scenario;
  const baseState = useMemo(() => scenarioState(baseFacilities, [], units, settings), [baseFacilities, units, settings]);
  const before = useMemo(() => typeResult(baseState, type, env, capacities[type?.id]), [baseState, type, env, capacities]);
  if (!actions.length) return <p className="hint">지도에서 위치를 눌러 시설을 추가하세요.</p>;
  const nameOfDong = Object.fromEntries(units.map((u) => [u.id, u.name]));
  const items = state.facilities.filter((f) => f.actionId);
  const allFacs = Object.fromEntries([...baseFacilities, ...state.facilities].map((f) => [f.id, f.name]));
  return (
    <section>
      <ul className="added-list">
        {actions.map((a) => {
          const item = items.find((x) => x.actionId === a.id);
          const id = `act-${a.id}`;
          return (
            <li key={a.id}>
              <button type="button" className={selection?.id === id ? "on" : ""} onClick={() => onSelectFacility(id)}>
                {a.name}{item?.dong && ` · ${nameOfDong[item.dong] ?? ""}`}
              </button>
              <button type="button" className="ghost" aria-label={`${a.name} ${a.kind === "add" ? "삭제" : "되돌리기"}`} onClick={() => onRemoveAction(a.id)}>
                {a.kind === "add" ? "삭제" : "되돌리기"}
              </button>
            </li>
          );
        })}
      </ul>
      {type && (
        <>
          <h3 className="tab-sub">{type.label} · 처음과 지금</h3>
          <CompareTable head={["지표", "처음", "지금"]} rows={compareRows(type, env, before, result, baseState, state)} />
          <UsageChart changes={usageChanges(before.usage, result.usage)} nameOf={(id) => allFacs[id] ?? id} color={type.color} labels={["처음", "지금"]} />
        </>
      )}
    </section>
  );
}
