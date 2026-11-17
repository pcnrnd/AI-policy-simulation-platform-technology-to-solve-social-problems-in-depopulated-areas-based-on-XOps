import { useMemo, useState } from "react";
import { Bar, Line } from "react-chartjs-2";
import Card from "./Card.jsx";
import InfoTip from "./InfoTip.jsx";
import PerfBadge from "./PerfBadge.jsx";
import TablePager, { usePager } from "./TablePager.jsx";
import { ChartEmptyNote } from "./PendingData.jsx";
import { useChartTheme } from "../hooks/useChartTheme.js";
import { usePublicTrends } from "../hooks/usePublicTrends.js";
import { useRenderTiming } from "../lib/perf.js";
import { formatUnitValue } from "../lib/realSimulation.js";
import {
  COMPARE_COLUMNS, DEFAULT_CRITERION, SORT_CRITERIA, conversionCompareView, evidenceChartsView, facilityRecommendationView
} from "../lib/facilityRecommendation.js";

const MEDAL = ["🥇", "🥈", "🥉"];
const SERIES_COLORS = [["rgba(59, 130, 246, 0.75)", "rgba(59, 130, 246, 1)"], ["rgba(14, 165, 233, 0.45)", "rgba(14, 165, 233, 1)"], ["rgba(16, 185, 129, 0.45)", "rgba(16, 185, 129, 1)"]];

/** 근거 그래프 하나(선 그래프) — 값이 없으면 캔버스 자리를 그대로 두고 행동 안내만 겹친다. */
function EvidenceChart({ chart, ct }) {
  const tickUnit = chart.unit.replace(/ 명$/, "");
  // 분기 눈금이 많아 듬성듬성(관측 구간은 실선, 예측 구간은 점선).
  const dense = chart.labels.length > 12;
  // 눈금은 5개만 — 처음과 마지막 달은 항상 보이게 고르게 나눈다.
  const shownTicks = new Set(Array.from({ length: 5 }, (_, k) => Math.round((k * (chart.labels.length - 1)) / 4)));
  const data = {
    labels: chart.labels,
    datasets: (chart.series.length ? chart.series : [{ label: chart.title, values: [] }]).map((s, i) => {
      const line = (SERIES_COLORS[i] ?? SERIES_COLORS[0])[1];
      return { label: s.label || chart.title, data: s.values, borderColor: line, backgroundColor: line, borderWidth: 2, pointRadius: 2, pointHoverRadius: 5, tension: 0.25, segment: { borderDash: (c) => (c.p1DataIndex >= chart.observed ? [5, 4] : undefined) }, ranges: s.ranges };
    })
  };
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    scales: {
      x: { grid: { display: false }, ticks: { color: ct.tickStrong, font: { size: 11 }, ...(dense && { autoSkip: false, maxRotation: 0, callback: (_, i) => (shownTicks.has(i) ? chart.labels[i] : "") }) } },
      // 단위는 눈금과 툴팁 값에 붙인다(축 제목·부제는 두지 않는다).
      y: { beginAtZero: true, grid: { color: ct.grid }, ticks: { color: ct.tick, callback: (v) => (v === 0 ? "0" : `${v.toLocaleString("ko-KR")}${tickUnit}`) } }
    },
    plugins: {
      legend: { display: chart.series.length > 1, position: "bottom", labels: { color: ct.tick, boxWidth: 10, font: { size: 10 } } },
      tooltip: {
        callbacks: {
          label: (ctx) => {
            const range = ctx.dataset.ranges?.[ctx.dataIndex];
            const text = `${ctx.dataset.label} ${ctx.parsed.y.toLocaleString("ko-KR")}${tickUnit}`;
            return range ? `${text} (${range[0].toLocaleString("ko-KR")}~${range[1].toLocaleString("ko-KR")}${tickUnit})` : text;
          }
        }
      }
    }
  };
  return (
    <figure className="reco-evidence-chart">
      <figcaption>
        <strong>{chart.title}</strong>
      </figcaption>
      <div className="reco-evidence-canvas">
        <Line data={data} options={options} aria-label={`${chart.title} 그래프`} role="img" />
        {chart.empty && <ChartEmptyNote>{chart.empty}</ChartEmptyNote>}
      </div>
    </figure>
  );
}

/** 전환 방안 비교 — 고른 시설의 방안 2~3개를 같은 표 구조로(값이 없으면 같은 자리에 행동 안내). */
function ConversionCompare({ compare }) {
  return (
    <section className="reco-compare" data-slot="sim.report.compare" aria-label={compare.title}>
      <div className="reco-compare-head">
        <strong>{compare.title}</strong>
        <span>{compare.scope}</span>
      </div>
      <div className="table-container">
        <table>
          <caption className="sr-only">{compare.scope || compare.title}</caption>
          <thead>
            <tr>{COMPARE_COLUMNS.map((label) => <th key={label} scope="col">{label}</th>)}</tr>
          </thead>
          <tbody>
            {compare.rows.length ? compare.rows.map((row) => (
              <tr key={row.id}>
                {row.cells.map((cell, i) => (
                  <td key={i} style={i === 0 ? { fontWeight: 600 } : undefined}>
                    {i === row.cells.length - 1 ? <span className="reco-verdict" data-verdict={row.verdict}>{cell}</span> : cell}
                  </td>
                ))}
              </tr>
            )) : (
              <tr><td colSpan={COMPARE_COLUMNS.length} style={{ color: "var(--text-muted)" }}>{compare.empty}</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

/**
 * 추천 결과 패널 — Top3 요약 + 막대 차트 + 근거 그래프 묶음(순위 그래프 바로 아래) + 전환 방안 비교 + 상세 표.
 * 데모 표시: facilityRows(시설 × 활용 방안) — 추천 지수 순위로 Top3·차트를 그리고, 표는 열 제목으로 고른 기준으로 정렬한다.
 * 실데이터 표시: view(sensitivityRecommendation 결과) — 같은 형식에 모델 예측상 변화를 넣는다. 전환 방안 비교·시설 기반 그래프는 같은 자리에 행동 안내.
 * 생활인구·정주인구 전망의 관측 값은 두 표시 모두 플랫폼 공표 통계 API(usePublicTrends)에서 온다(정책 반영 선은 시설 표시만).
 * @param {{ region: object, facilityRows?: object[], view?: object }} props
 */
export default function PolicyRecommendation({ region, facilityRows = null, view: realView = null }) {
  const ct = useChartTheme();
  const [criterion, setCriterion] = useState(DEFAULT_CRITERION);
  const view = useMemo(
    () => realView ?? facilityRecommendationView(facilityRows, criterion),
    [realView, facilityRows, criterion]
  );
  const ranked = view.items;
  // 1~3순위 카드로 고른 시설(기본 1순위) — 전환 방안 비교와 정책 반영 전망 선이 따라 바뀐다.
  const [selectedId, setSelectedId] = useState(null);
  const selected = ranked.find((s) => s.id === selectedId) ?? ranked[0] ?? null;
  const compare = useMemo(() => conversionCompareView(realView ? null : selected), [realView, selected]);
  const trends = usePublicTrends(region?.name);
  const evidenceCharts = useMemo(
    () => (realView ? evidenceChartsView(trends, null) : evidenceChartsView(trends, ranked, selected)),
    [realView, trends, ranked, selected]
  );
  const vizMs = useRenderTiming([ranked]);
  const { scoreLabel, scoreUnit, columns } = view;
  const cellsOf = (s) => [s.name, ...s.cells];
  const scoreText = (s) => s.scoreText ?? formatUnitValue(s.score, scoreUnit);
  const rankText = view.rankText ?? ((i) => `${MEDAL[i]} 추천 ${i + 1}`);
  const statusOf = view.status ?? ((_, i) => ({ text: i < 3 ? "추천" : "검토", strong: i < 3 }));
  const sortOf = (key) => (key === view.criterion ? (SORT_CRITERIA[key].ascending ? "ascending" : "descending") : "none");
  const sortIcon = { none: "fa-sort", ascending: "fa-sort-up", descending: "fa-sort-down" };
  // 정렬 기준을 고를 수 있는 열 제목은 버튼으로(열 제목 글자 그대로).
  const headLabel = (col) => (col.sortKey ? (
    <button type="button" className="reco-sort" onClick={() => setCriterion(col.sortKey)}>
      {col.label} <i className={`fa-solid ${sortIcon[sortOf(col.sortKey)]}`} aria-hidden="true"></i>
    </button>
  ) : col.label);
  const scoreColumn = view.scoreColumn;

  const chartData = useMemo(
    () => ({
      labels: ranked.map((s) => s.chartLabel ?? s.name),
      datasets: [
        {
          label: scoreLabel,
          data: ranked.map((s) => s.score),
          backgroundColor: ranked.map((_, i) =>
            i < 3 ? "rgba(59, 130, 246, 0.85)" : "rgba(139, 92, 246, 0.55)"
          ),
          borderColor: ranked.map((_, i) =>
            i < 3 ? "rgba(59, 130, 246, 1)" : "rgba(139, 92, 246, 1)"
          ),
          borderWidth: 1.5,
          borderRadius: 4
        }
      ]
    }),
    [ranked, scoreLabel]
  );

  const chartOpts = {
    indexAxis: "y",
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    scales: {
      // 시설 추천은 값 축에 정렬 기준과 단위를 적는다.
      x: { grid: { color: ct.grid }, ticks: { color: ct.tick, ...(view.scoreUnit && { callback: (v) => `${v.toLocaleString("ko-KR")}${view.scoreUnit}` }) }, ...(view.criterion && { title: { display: true, text: scoreLabel, color: ct.tick } }),
        ...(view.scoreMax && { min: 0, max: view.scoreMax }) },
      y: { grid: { display: false }, ticks: { color: ct.tickStrong, font: { size: 11 } } }
    },
    plugins: { legend: { display: false }, tooltip: { callbacks: { label: (ctx) => `${scoreLabel} ${ctx.parsed.x.toLocaleString("ko-KR")}${view.scoreUnit ?? ""}` } } }
  };

  const top3 = ranked.slice(0, 3);
  // 표는 고른 정렬 기준의 순서(시설 추천) — 순위 칸·검토 구분은 추천 순위 그대로.
  const rows = usePager(view.tableItems ?? ranked, undefined, view.criterion);
  // 산정 근거는 두 모드 모두 제목 옆 툴팁 하나로 — 본문 설명 문단은 두지 않는다.
  const basisTip = view.basisTip
    ?? `관측 지표의 예측 민감도를 근거로 추천합니다. 변수끼리의 관계와 실제 실현 가능성을 함께 확인하세요. ${view.infoTip}`;

  return (
    <Card
      title={<>{region.name} {view.title ?? "맞춤 정책 추천"}<InfoTip text={basisTip} label="추천 산정 근거 보기" /></>}
      icon="fa-lightbulb"
      headerRight={<PerfBadge ms={vizMs} />}
      dataSource={realView ? "api" : "mock"}
      data-slot="sim.report.panel"
    >

      {/* Top 3 추천 요약 */}
      <div className="reco-top3">
        {top3.map((s, i) => (
          <div
            key={s.id}
            className="reco-card"
            role="button"
            tabIndex={0}
            aria-pressed={s.id === selected?.id}
            onClick={() => setSelectedId(s.id)}
            onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setSelectedId(s.id); } }}
          >
            <div className="reco-rank">{rankText(i)}</div>
            <div className="reco-name">{s.name}</div>
            <div className="reco-cat">{s.category}</div>
            <div className="reco-effect">{s.effectLine ?? ""}</div>
            <div className="reco-score">
              {scoreText(s)} <span>{s.scoreUnit ?? scoreUnit}</span>
            </div>
          </div>
        ))}
      </div>

      <div style={{ position: "relative", height: 300, width: "100%", marginBottom: 20 }}>
        <Bar data={chartData} options={chartOpts} />
      </div>

      {/* 근거 그래프 묶음 — 순위 그래프 바로 아래, 두 모드 같은 두 자리 */}
      <section className="reco-evidence-charts" data-slot="sim.report.charts" aria-label="추천 참고 그래프">
        {evidenceCharts.map((chart) => <EvidenceChart key={chart.id} chart={chart} ct={ct} />)}
      </section>

      {/* 전환 방안 비교 — 근거 그래프 묶음 아래·표 위, 두 모드 같은 표 구조 */}
      <ConversionCompare compare={compare} />

      <div className={view.criterion ? "table-container reco-page-table" : "table-container"}>
        <table>
          <caption className="sr-only">{view.caption}</caption>
          <thead>
            <tr>
              <th scope="col" aria-sort={view.rankColumn ? sortOf(view.rankColumn.sortKey) : undefined}>{headLabel(view.rankColumn ?? { label: "순위" })}</th>
              {columns.map((col) => (
                <th key={col.label} scope="col" className={col.numeric ? "cell-num" : undefined} aria-sort={col.sortKey ? sortOf(col.sortKey) : undefined}>{headLabel(col)}</th>
              ))}
              {scoreColumn && <th scope="col" className="cell-num" aria-sort={scoreColumn.sortKey ? sortOf(scoreColumn.sortKey) : undefined}>{headLabel(scoreColumn)}</th>}
              <th scope="col">{view.statusLabel ?? "추천"}</th>
            </tr>
          </thead>
          <tbody>
            {rows.pageRows.map((s, pageIndex) => {
              const i = rows.offset + pageIndex;
              const status = statusOf(s, i);
              return (
              <tr key={s.id}>
                <td style={{ fontWeight: 700 }}>{view.rankText ? s.rank ?? i + 1 : MEDAL[i] ?? i + 1}</td>
                {cellsOf(s).map((cell, index) => (
                  <td
                    key={index}
                    className={columns[index].numeric ? "cell-num" : undefined}
                    style={index === 0 ? { fontWeight: 600 } : undefined}
                  >
                    {cell}
                  </td>
                ))}
                {scoreColumn && (
                  <td className="cell-num" style={{ fontWeight: 700, color: "var(--accent-blue)" }}>
                    {s.scoreCell}
                  </td>
                )}
                <td>
                  {status.strong ? (
                    <span className="system-status" style={{ padding: "1px 8px", fontSize: 10 }}>
                      {status.text}
                    </span>
                  ) : (
                    <span style={{ color: "var(--text-muted)", fontSize: 11 }}>{status.text}</span>
                  )}
                </td>
              </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <TablePager {...rows.pager} />
    </Card>
  );
}
