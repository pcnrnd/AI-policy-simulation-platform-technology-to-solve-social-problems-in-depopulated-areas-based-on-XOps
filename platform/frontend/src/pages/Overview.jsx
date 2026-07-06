import { useMemo } from "react";
import { Doughnut, Radar } from "react-chartjs-2";
import { useAppState } from "../context/AppStateContext.jsx";
import { useTheme } from "../context/ThemeContext.jsx";
import { useChartTheme } from "../hooks/useChartTheme.js";
import StatCard from "../components/StatCard.jsx";
import Card from "../components/Card.jsx";
import RegionStatusCard from "../components/RegionStatusCard.jsx";

// 데이터 소스 수에 맞춰 순환 사용하는 색상 팔레트 —
// 플랫폼 액센트(블루·시안·바이올렛·틸) 한 계열로 통일해 글래스모피즘 톤앤매너에 맞춘다.
const SOURCE_PALETTE = [
  "rgba(59, 130, 246, 0.85)",
  "rgba(34, 211, 238, 0.8)",
  "rgba(139, 92, 246, 0.8)",
  "rgba(16, 185, 129, 0.8)",
  "rgba(99, 102, 241, 0.8)",
  "rgba(45, 212, 191, 0.8)"
];

export default function Overview() {
  const { appData, currentRegion, setCurrentRegion, f1Override, focusRegion } = useAppState();
  const ct = useChartTheme();
  const { isDark } = useTheme();

  const f1Value = f1Override !== null ? f1Override.toFixed(3) : "0.884";
  const f1Label =
    f1Override !== null ? "최적 (SOTA v3.1)" : "최적 (SOTA)";
  const f1Sub = f1Override !== null ? "연합 재학습 성공" : "데이터 소스 통합 기준";

  // 도넛: 소스별 아카이브 적재 행 수 — "어떤 소스가 얼마나 적재돼 있는가"를 보여준다.
  const sourceData = useMemo(() => {
    const schemas = appData.metadata_schemas;
    return {
      labels: schemas.map((s) => s.label ?? s.id),
      datasets: [
        {
          data: schemas.map((s) => s.archive?.rows ?? 0),
          backgroundColor: schemas.map((_, i) => SOURCE_PALETTE[i % SOURCE_PALETTE.length]),
          // 보더는 카드 배경과 동화되도록 테마별 분기 (라이트에서 검은 띠 방지)
          borderColor: isDark ? "rgba(8, 13, 26, 1)" : "#ffffff",
          borderWidth: 2
        }
      ]
    };
  }, [appData, isDark]);

  const sourceCount = appData.metadata_schemas.length;

  const doughnutOpts = {
    responsive: true,
    maintainAspectRatio: false,
    cutout: "60%",
    plugins: {
      legend: { position: "right", labels: { color: ct.legend, boxWidth: 12, font: { size: 11 } } },
      tooltip: {
        callbacks: {
          label: (cx) => ` ${cx.label}: ${cx.parsed.toLocaleString()}행 적재`
        }
      }
    }
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
          label="AI 예측 소멸위기 지역수 (전국 기준)"
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
        <Card title="연동 데이터 소스 아카이브 적재 현황 (행 수)" icon="fa-chart-pie">
          <div style={{ position: "relative", height: 240, width: "100%" }}>
            <Doughnut data={sourceData} options={doughnutOpts} />
          </div>
        </Card>

        <Card
          title={`${currentRegion.name} 정책 영향 프로파일`}
          icon="fa-bullseye"
          headerRight={
            <select
              className="select-control"
              style={{ padding: "5px 8px", fontSize: 12, width: "auto" }}
              value={currentRegion.id}
              onChange={(e) => {
                const region = appData.regions.find((r) => r.id === e.target.value);
                if (region) setCurrentRegion(region);
              }}
              aria-label="모니터링 대상 지자체 선택"
            >
              {appData.regions.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.name}
                </option>
              ))}
            </select>
          }
        >
          <div style={{ position: "relative", height: 240, width: "100%" }}>
            <Radar data={radarData} options={radarOpts} />
          </div>
        </Card>
      </div>
    </>
  );
}
