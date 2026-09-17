import { useMemo } from "react";
import Card from "./Card.jsx";
import { NO_DEMO_DATA } from "./PendingData.jsx";

// 소멸위험지수(낮을수록 위험) → 등급/색상 매핑.
function riskGrade(riskIndex) {
  if (riskIndex < 0.15) return { label: "소멸 고위험", color: "var(--accent-red)", rgbToken: "--accent-red-rgb" };
  if (riskIndex < 0.18) return { label: "소멸 주의", color: "var(--accent-orange)", rgbToken: "--accent-orange-rgb" };
  return { label: "관찰 단계", color: "var(--accent-teal)", rgbToken: "--accent-teal-rgb" };
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
 *
 * allowSeed=false(데모 표시 OFF)이면 표 구조는 그대로 두고 행만 비운다 — 지자체 인구·고령화지수·
 * 출산율·위험등급은 mock_data.json 전용 항목이라 실저장소에 대응 값이 없다.
 * @param {{ regions: object[], currentRegionId?: string, onSelectRegion: (region: object) => void, allowSeed?: boolean }} props
 */
export default function RegionStatusCard({ regions, currentRegionId, onSelectRegion, allowSeed = true }) {
  const sorted = useMemo(
    () => (allowSeed ? [...regions].sort((a, b) => a.riskIndex - b.riskIndex) : []),
    [regions, allowSeed]
  );

  return (
    <Card
      title="지자체별 인구감소 현황"
      icon="fa-triangle-exclamation"
      className="page-section"
      headerRight={
        <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
          위험지수 오름차순 · 지자체명 선택 시 시뮬레이터 이동
        </span>
      }
    >
      <div className="table-container" style={{ marginTop: 10 }}>
        <table>
          <caption className="sr-only">지자체별 인구감소 지표와 소멸 위험등급</caption>
          <thead>
            <tr>
              <th scope="col">지자체</th>
              <th scope="col" className="cell-num">인구수</th>
              <th scope="col" className="cell-num">10년 증감</th>
              <th scope="col" className="cell-num">고령화지수</th>
              <th scope="col" className="cell-num">출산율</th>
              <th scope="col">위험등급</th>
            </tr>
          </thead>
          <tbody>
            {sorted.length === 0 && (
              <tr>
                <td className="empty-table-cell" colSpan={6}>
                  {NO_DEMO_DATA} — 지자체 인구·고령화지수·출산율·위험등급은 시드 전용 항목이라
                  실저장소에 대응 값이 없습니다.
                </td>
              </tr>
            )}
            {sorted.map((region) => {
              const grade = riskGrade(region.riskIndex);
              const decline = declineRate(region.history);
              const isActive = region.id === currentRegionId;
              return (
                <tr
                  key={region.id}
                  className={isActive ? "table-row-selected" : ""}
                    style={undefined}
                >
                  <td>
                    <button
                      type="button"
                      className="table-row-action"
                      onClick={() => onSelectRegion(region)}
                      aria-current={isActive ? "true" : undefined}
                    >
                      {region.name}
                      <span className="sr-only"> 정책 시뮬레이터로 이동</span>
                    </button>
                    {region.theme && (
                      <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>
                        <i className="fa-solid fa-circle-exclamation" style={{ marginRight: 4 }}></i>
                        {region.theme}
                      </div>
                    )}
                  </td>
                  <td className="cell-num">{region.population.toLocaleString()}명</td>
                  <td className="cell-num" style={{ color: "var(--accent-red)" }}>
                    {decline !== null ? `${decline}%` : "—"}
                  </td>
                  <td className="cell-num">{region.agingIndex}%</td>
                  <td className="cell-num">{region.birthRate}</td>
                  <td>
                    <span
                      className="system-status"
                      style={{
                        padding: "2px 8px",
                        fontSize: 11,
                        backgroundColor: `rgba(var(${grade.rgbToken}), 0.02)`,
                        borderColor: grade.color,
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
