import { useMemo } from "react";
import { Bar } from "react-chartjs-2";
import Card from "./Card.jsx";
import PerfBadge from "./PerfBadge.jsx";
import { useChartTheme } from "../hooks/useChartTheme.js";
import { useRenderTiming } from "../lib/perf.js";

const MEDAL = ["🥇", "🥈", "🥉"];

/**
 * 정책 추천 결과 패널 — 시뮬레이터 입력(지자체 + 정책 변수)으로 도출된 RICE 랭킹을
 * Top3 요약 + 막대 차트 + 상세 표로 표시한다.
 * @param {{ region: object, ranked: object[] }} props
 */
export default function PolicyRecommendation({ region, ranked }) {
  const ct = useChartTheme();
  const vizMs = useRenderTiming([ranked]);

  const chartData = useMemo(
    () => ({
      labels: ranked.map((s) => s.name),
      datasets: [
        {
          label: "정책 추천 점수",
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
    [ranked]
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

  return (
    <Card
      title={`${region.name} 맞춤 정책 추천 (RICE 기반)`}
      icon="fa-lightbulb"
      style={{ marginTop: 24 }}
      headerRight={<PerfBadge ms={vizMs} />}
    >
      <p style={{ fontSize: 13, color: "var(--text-secondary)", marginBottom: 16 }}>
        시뮬레이터의 정책 변수(복지·산업·주거 가중치)와 지자체 인구감소 특성을 반영해{" "}
        <strong>RICE = (Reach × Impact × Confidence) ÷ Effort</strong> 점수로 맞춤 정책을 추천합니다.
      </p>

      {/* Top 3 추천 요약 */}
      <div className="reco-top3">
        {top3.map((s, i) => (
          <div key={s.id} className="reco-card">
            <div className="reco-rank">{MEDAL[i]} 추천 {i + 1}</div>
            <div className="reco-name">{s.name}</div>
            <div className="reco-cat">{s.category}</div>
            <div className="reco-score">
              {s.score.toLocaleString()} <span>점</span>
            </div>
          </div>
        ))}
      </div>

      <div style={{ position: "relative", height: 300, width: "100%", marginBottom: 20 }}>
        <Bar data={chartData} options={chartOpts} />
      </div>

      <div className="table-container">
        <table>
          <thead>
            <tr>
              <th>순위</th>
              <th>정책 전략</th>
              <th>대응 유형</th>
              <th>Reach</th>
              <th>Impact(보정)</th>
              <th>Confidence</th>
              <th>Effort</th>
              <th>추천 점수</th>
              <th>추천</th>
            </tr>
          </thead>
          <tbody>
            {ranked.map((s, i) => (
              <tr key={s.id}>
                <td style={{ fontWeight: 700 }}>{MEDAL[i] ?? i + 1}</td>
                <td style={{ fontWeight: 600 }}>{s.name}</td>
                <td style={{ color: "var(--text-secondary)", fontSize: 12 }}>{s.category}</td>
                <td>{s.reach.toLocaleString()}</td>
                <td>{s.adjImpact}</td>
                <td>{s.confidence}</td>
                <td>{s.effort}</td>
                <td style={{ fontWeight: 700, color: "var(--accent-blue)" }}>
                  {s.score.toLocaleString()}
                </td>
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
            ))}
          </tbody>
        </table>
      </div>
      <p style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 12 }}>
        ※ Impact는 지자체 정책 영향 가중치와 시뮬레이터 슬라이더 강조도로 보정된 값입니다.
      </p>
    </Card>
  );
}
