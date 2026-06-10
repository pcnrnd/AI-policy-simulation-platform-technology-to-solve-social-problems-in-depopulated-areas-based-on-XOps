import { useMemo } from "react";
import Card from "./Card.jsx";

// 소멸위험지수(낮을수록 위험) → 등급/색상 매핑.
function riskGrade(riskIndex) {
  if (riskIndex < 0.15) return { label: "소멸 고위험", color: "var(--accent-red)", rgb: "239, 68, 68" };
  if (riskIndex < 0.18) return { label: "소멸 주의", color: "var(--accent-orange)", rgb: "245, 158, 11" };
  return { label: "관찰 단계", color: "var(--accent-teal)", rgb: "16, 185, 129" };
}

// 10개년 history 첫값 대비 최근값 감소율(%).
function declineRate(history) {
  if (!Array.isArray(history) || history.length < 2) return null;
  const first = history[0];
  const last = history[history.length - 1];
  if (!first) return null;
  return (((last - first) / first) * 100).toFixed(1);
}

/**
 * 지자체별 인구감소 현황 요약 — 종합 대시보드의 핵심 도메인 패널.
 * 위험지수 오름차순(고위험 우선) 정렬, 행 클릭 시 해당 지자체로 시뮬레이터 이동.
 * @param {{ regions: object[], currentRegionId?: string, onSelectRegion: (region: object) => void }} props
 */
export default function RegionStatusCard({ regions, currentRegionId, onSelectRegion }) {
  const sorted = useMemo(
    () => [...regions].sort((a, b) => a.riskIndex - b.riskIndex),
    [regions]
  );

  return (
    <Card
      title="지자체별 인구감소 현황"
      icon="fa-triangle-exclamation"
      headerRight={
        <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
          위험지수 오름차순 · 행 클릭 시 시뮬레이터 이동
        </span>
      }
    >
      <div className="table-container" style={{ marginTop: 10 }}>
        <table>
          <thead>
            <tr>
              <th>지자체</th>
              <th>인구수</th>
              <th>10년 증감</th>
              <th>고령화지수</th>
              <th>출산율</th>
              <th>위험등급</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((region) => {
              const grade = riskGrade(region.riskIndex);
              const decline = declineRate(region.history);
              const isActive = region.id === currentRegionId;
              return (
                <tr
                  key={region.id}
                  onClick={() => onSelectRegion(region)}
                  style={{
                    cursor: "pointer",
                    backgroundColor: isActive ? "rgba(59,130,246,0.08)" : undefined
                  }}
                  title={`${region.name} 정책 시뮬레이터로 이동`}
                >
                  <td>
                    <strong>{region.name}</strong>
                    {region.theme && (
                      <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>
                        <i className="fa-solid fa-circle-exclamation" style={{ marginRight: 4 }}></i>
                        {region.theme}
                      </div>
                    )}
                  </td>
                  <td>{region.population.toLocaleString()}명</td>
                  <td style={{ color: "var(--accent-red)" }}>
                    {decline !== null ? `${decline}%` : "—"}
                  </td>
                  <td>{region.agingIndex}%</td>
                  <td>{region.birthRate}</td>
                  <td>
                    <span
                      className="system-status"
                      style={{
                        padding: "2px 8px",
                        fontSize: 11,
                        backgroundColor: `rgba(${grade.rgb}, 0.1)`,
                        borderColor: `rgba(${grade.rgb}, 0.25)`,
                        color: grade.color
                      }}
                    >
                      {grade.label} ({region.riskIndex})
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
