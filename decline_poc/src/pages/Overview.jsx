import { useMemo } from "react";
import { Doughnut, Radar } from "react-chartjs-2";
import { useAppState } from "../context/AppStateContext.jsx";
import { useChartTheme } from "../hooks/useChartTheme.js";
import StatCard from "../components/StatCard.jsx";
import Card from "../components/Card.jsx";
import RegionStatusCard from "../components/RegionStatusCard.jsx";

// 데이터 소스 수에 맞춰 순환 사용하는 단계 색상 팔레트.
const SOURCE_PALETTE = [
  "rgba(59, 130, 246, 0.75)",
  "rgba(16, 185, 129, 0.75)",
  "rgba(139, 92, 246, 0.75)",
  "rgba(245, 158, 11, 0.75)",
  "rgba(236, 72, 153, 0.75)",
  "rgba(20, 184, 166, 0.75)"
];

export default function Overview() {
  const { appData, currentRegion, f1Override, focusRegion } = useAppState();
  const ct = useChartTheme();

  const f1Value = f1Override !== null ? f1Override.toFixed(3) : "0.884";
  const f1Label =
    f1Override !== null ? "최적 (SOTA v3.1)" : "최적 (SOTA)";
  const f1Sub = f1Override !== null ? "연합 재학습 성공" : "데이터 소스 통합 기준";

  // 도넛: 연동 데이터 소스 구성 (스키마 컬럼 수 기반, 데이터 소스 수에 동적 대응)
  const sourceData = useMemo(() => {
    const schemas = appData.metadata_schemas;
    return {
      labels: schemas.map((s) => s.label ?? s.id),
      datasets: [
        {
          data: schemas.map((s) => s.columns.length),
          backgroundColor: schemas.map((_, i) => SOURCE_PALETTE[i % SOURCE_PALETTE.length]),
          borderColor: "rgba(8, 13, 26, 1)",
          borderWidth: 2
        }
      ]
    };
  }, [appData]);

  const sourceCount = appData.metadata_schemas.length;

  const doughnutOpts = {
    responsive: true,
    maintainAspectRatio: false,
    cutout: "60%",
    plugins: { legend: { position: "right", labels: { color: ct.legend, boxWidth: 12, font: { size: 11 } } } }
  };

  // 레이더: 선택 지자체 정책 영향 프로파일
  const radarData = useMemo(() => {
    const p = currentRegion.policyImpacts;
    return {
      labels: ["복지 영향", "산업 영향", "주거 영향", "출산율", "위험 완화 여력"],
      datasets: [
        {
          label: currentRegion.name,
          data: [
            p.welfare * 100,
            p.industry * 100,
            p.housing * 100,
            currentRegion.birthRate * 100,
            (1 - currentRegion.riskIndex) * 100
          ],
          backgroundColor: "rgba(59, 130, 246, 0.18)",
          borderColor: "rgba(59, 130, 246, 1)",
          borderWidth: 2,
          pointBackgroundColor: "rgba(16, 185, 129, 1)"
        }
      ]
    };
  }, [currentRegion]);

  const radarOpts = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      r: {
        angleLines: { color: ct.angleLines },
        grid: { color: ct.angleLines },
        pointLabels: { color: ct.tick, font: { size: 11 } },
        ticks: { display: false, backdropColor: "transparent" },
        suggestedMin: 0,
        suggestedMax: 100
      }
    },
    plugins: { legend: { labels: { color: ct.legend, boxWidth: 12 } } }
  };

  return (
    <>
      <div className="grid-cols-3">
        <StatCard
          label="AI 예측 소멸위기 지역수"
          icon="fa-triangle-exclamation"
          value="89개소"
          footer={
            <>
              <span className="trend-up">
                <i className="fa-solid fa-caret-up"></i> 4개소
              </span>
              <span className="text-secondary">전분기 대비</span>
            </>
          }
        />
        <StatCard
          label="글로벌 모델 F1-score"
          icon="fa-bullseye"
          value={f1Value}
          footer={
            <>
              <span className="trend-up" style={{ color: "var(--accent-teal)" }}>
                <i className="fa-solid fa-circle-check"></i> {f1Label}
              </span>
              <span className="text-secondary">{f1Sub}</span>
            </>
          }
        />
        <StatCard
          label="연동 데이터 소스"
          icon="fa-network-wired"
          value={`${sourceCount}개 실시간`}
          footer={
            <>
              <span className="trend-up">
                <i className="fa-solid fa-arrow-right"></i> Active
              </span>
              <span className="text-secondary">주민·복지·산업·공간·스마트팜·시설</span>
            </>
          }
        />
      </div>

      <RegionStatusCard
        regions={appData.regions}
        currentRegionId={currentRegion.id}
        onSelectRegion={focusRegion}
      />

      <div className="grid-cols-2">
        <Card title="연동 데이터 소스 구성" icon="fa-chart-pie">
          <div style={{ position: "relative", height: 240, width: "100%" }}>
            <Doughnut data={sourceData} options={doughnutOpts} />
          </div>
        </Card>

        <Card title={`${currentRegion.name} 정책 영향 프로파일`} icon="fa-bullseye">
          <div style={{ position: "relative", height: 240, width: "100%" }}>
            <Radar data={radarData} options={radarOpts} />
          </div>
        </Card>
      </div>
    </>
  );
}
