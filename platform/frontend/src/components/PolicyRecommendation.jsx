import { useMemo, useState } from "react";
import { Bar } from "react-chartjs-2";
import Card from "./Card.jsx";
import InfoTip from "./InfoTip.jsx";
import PerfBadge from "./PerfBadge.jsx";
import TablePager, { usePager } from "./TablePager.jsx";
import { useChartTheme } from "../hooks/useChartTheme.js";
import { useRenderTiming } from "../lib/perf.js";
import { formatUnitValue } from "../lib/realSimulation.js";
import { DEFAULT_CRITERION, SORT_CRITERIA, facilityRecommendationView } from "../lib/facilityRecommendation.js";

const MEDAL = ["🥇", "🥈", "🥉"];

/**
 * 추천 결과 패널 — Top3 요약 + 막대 차트 + 상세 표.
 * 데모 표시: facilityRows(시설 × 활용 방안) — 사용자가 표의 열 제목으로 고른 정렬 기준 하나로 순서를 정한다.
 * 실데이터 표시: view(sensitivityRecommendation 결과) — 같은 형식에 모델 예측상 변화를 넣는다.
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
      labels: ranked.map((s) => s.name),
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
    scales: {
      // 시설 추천은 값 축에 정렬 기준과 단위를 적는다.
      x: { grid: { color: ct.grid }, ticks: { color: ct.tick }, ...(view.criterion && { title: { display: true, text: scoreLabel, color: ct.tick } }) },
      y: { grid: { display: false }, ticks: { color: ct.tickStrong, font: { size: 11 } } }
    },
    plugins: { legend: { display: false } }
  };

  const top3 = ranked.slice(0, 3);
  const rows = usePager(ranked);
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
          <div key={s.id} className="reco-card">
            <div className="reco-rank">{rankText(i)}</div>
            <div className="reco-name">{s.name}</div>
            <div className="reco-cat">{s.category}</div>
            <div className="reco-score">
              {scoreText(s)} <span>{s.scoreUnit ?? scoreUnit}</span>
            </div>
          </div>
        ))}
      </div>

      <div style={{ position: "relative", height: 300, width: "100%", marginBottom: 20 }}>
        <Bar data={chartData} options={chartOpts} />
      </div>

      <div className={view.criterion ? "table-container reco-page-table" : "table-container"}>
        <table>
          <caption className="sr-only">{view.caption}</caption>
          <thead>
            <tr>
              <th scope="col">순위</th>
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
                <td style={{ fontWeight: 700 }}>{view.rankText ? i + 1 : MEDAL[i] ?? i + 1}</td>
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
      <p style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 12 }}>
        {view.footnote}
      </p>
    </Card>
  );
}
