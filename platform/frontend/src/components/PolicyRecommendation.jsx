import { useMemo } from "react";
import { Bar } from "react-chartjs-2";
import Card from "./Card.jsx";
import InfoTip from "./InfoTip.jsx";
import PerfBadge from "./PerfBadge.jsx";
import TablePager, { usePager } from "./TablePager.jsx";
import { useChartTheme } from "../hooks/useChartTheme.js";
import { useRenderTiming } from "../lib/perf.js";
import { formatUnitValue } from "../lib/realSimulation.js";

const MEDAL = ["🥇", "🥈", "🥉"];
const RICE_COLUMNS = [
  { label: "Reach", numeric: true },
  { label: "Impact(보정)", numeric: true },
  { label: "Confidence", numeric: true },
  { label: "Effort", numeric: true }
];

/**
 * 정책 추천 결과 패널 — Top3 요약 + 막대 차트 + 상세 표.
 * 데모 표시: 시뮬레이터 입력(지자체 + 정책 변수)으로 도출된 RICE 랭킹(ranked).
 * 실데이터 표시: view(sensitivityRecommendation 결과) — 같은 형식에 모델 예측상 변화를 넣고 RICE·점수는 쓰지 않는다.
 * @param {{ region: object, ranked?: object[], view?: object }} props
 */
export default function PolicyRecommendation({ region, ranked: riceRanked, view = null }) {
  const ct = useChartTheme();
  const ranked = view ? view.items : riceRanked;
  const vizMs = useRenderTiming([ranked]);
  const scoreLabel = view ? view.scoreLabel : "정책 추천 점수";
  const scoreUnit = view ? view.scoreUnit : "점";
  const columns = view ? view.columns : [{ label: "정책 전략" }, { label: "대응 유형" }, ...RICE_COLUMNS];
  const cellsOf = (s) => view ? [s.name, ...s.cells] : [s.name, s.category, formatUnitValue(s.reach, "명"), s.adjImpact, s.confidence, s.effort];
  const scoreText = (s) => s.scoreText ?? formatUnitValue(s.score, scoreUnit);

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
      x: { grid: { color: ct.grid }, ticks: { color: ct.tick } },
      y: { grid: { display: false }, ticks: { color: ct.tickStrong, font: { size: 11 } } }
    },
    plugins: { legend: { display: false } }
  };

  const top3 = ranked.slice(0, 3);
  const rows = usePager(ranked);
  // 산정 근거는 두 모드 모두 제목 옆 툴팁 하나로 — 본문 설명 문단은 두지 않는다.
  const basisTip = view
    ? `관측 지표의 예측 민감도를 근거로 추천합니다. 변수끼리의 관계와 실제 실현 가능성을 함께 확인하세요. ${view.infoTip}`
    : "표시 점수 = 안내식 × 예산 보정계수 √(총예산/600억). 남원 2,000억 예시: 3,061 × 1.83 ≈ 5,589";

  return (
    <Card
      title={<>{region.name} 맞춤 정책 추천<InfoTip text={basisTip} label="추천 산정 근거 보기" /></>}
      icon="fa-lightbulb"
      headerRight={<PerfBadge ms={vizMs} />}
      dataSource={view ? "api" : "mock"}
      data-slot="sim.report.panel"
    >

      {/* Top 3 추천 요약 */}
      <div className="reco-top3">
        {top3.map((s, i) => (
          <div key={s.id} className="reco-card">
            <div className="reco-rank">{MEDAL[i]} 추천 {i + 1}</div>
            <div className="reco-name">{s.name}</div>
            <div className="reco-cat">{s.category}</div>
            <div className="reco-score">
              {scoreText(s)} <span>{scoreUnit}</span>
            </div>
          </div>
        ))}
      </div>

      <div style={{ position: "relative", height: 300, width: "100%", marginBottom: 20 }}>
        <Bar data={chartData} options={chartOpts} />
      </div>

      <div className="table-container">
        <table>
          <caption className="sr-only">{view ? view.caption : "RICE 방식으로 산정한 맞춤 정책 추천 순위와 구성 점수"}</caption>
          <thead>
            <tr>
              <th scope="col">순위</th>
              {columns.map((col) => (
                <th key={col.label} scope="col" className={col.numeric ? "cell-num" : undefined}>{col.label}</th>
              ))}
              {!view && <th scope="col" className="cell-num">추천 점수</th>}
              <th scope="col">추천</th>
            </tr>
          </thead>
          <tbody>
            {rows.pageRows.map((s, pageIndex) => {
              const i = rows.offset + pageIndex;
              return (
              <tr key={s.id}>
                <td style={{ fontWeight: 700 }}>{MEDAL[i] ?? i + 1}</td>
                {cellsOf(s).map((cell, index) => (
                  <td
                    key={index}
                    className={columns[index].numeric ? "cell-num" : undefined}
                    style={index === 0 ? { fontWeight: 600 } : !view && index === 1 ? { color: "var(--text-secondary)", fontSize: 12 } : undefined}
                  >
                    {cell}
                  </td>
                ))}
                {!view && (
                  <td className="cell-num" style={{ fontWeight: 700, color: "var(--accent-blue)" }}>
                    {scoreText(s)}
                  </td>
                )}
                <td>
                  {i < 3 ? (
                    <span className="system-status" style={{ padding: "1px 8px", fontSize: 10 }}>
                      추천
                    </span>
                  ) : (
                    <span style={{ color: "var(--text-muted)", fontSize: 11 }}>검토</span>
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
        {view ? view.footnote : "※ Impact는 지자체 정책 영향 가중치와 시뮬레이터 슬라이더 강조도로 보정된 값입니다."}
      </p>
    </Card>
  );
}
