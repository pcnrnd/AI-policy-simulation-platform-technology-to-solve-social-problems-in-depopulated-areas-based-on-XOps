import { useEffect, useMemo, useState } from "react";
import { Doughnut, Radar } from "react-chartjs-2";
import { useAppState } from "../context/AppStateContext.jsx";
import { useTheme } from "../context/ThemeContext.jsx";
import { useChartTheme } from "../hooks/useChartTheme.js";
import StatCard from "../components/StatCard.jsx";
import Card from "../components/Card.jsx";
import RegionStatusCard from "../components/RegionStatusCard.jsx";

// 데이터 소스 수에 맞춰 순환 사용하는 색상 팔레트 —
// 플랫폼 액센트(블루·시안·바이올렛·틸) 한 계열로 통일해 글래스모피즘 톤앤매너에 맞춘다.
// f1Override(연합 재학습 승급 지표)를 내보내는 모델 — Model Store에서 운영 버전을 찾을 때 사용
const POPULATION_MODEL_ID = "population-forecast";

const SOURCE_PALETTE = [
  "rgba(59, 130, 246, 0.85)",
  "rgba(34, 211, 238, 0.8)",
  "rgba(139, 92, 246, 0.8)",
  "rgba(16, 185, 129, 0.8)",
  "rgba(99, 102, 241, 0.8)",
  "rgba(45, 212, 191, 0.8)"
];

export default function Overview() {
  const { appData, currentRegion, setCurrentRegion, f1Override, focusRegion, modelStore } = useAppState();
  const ct = useChartTheme();
  const { isDark } = useTheme();
  const [compactChart, setCompactChart] = useState(() => window.matchMedia("(max-width: 768px)").matches);

  useEffect(() => {
    const media = window.matchMedia("(max-width: 768px)");
    const onChange = (event) => setCompactChart(event.matches);
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, []);

  // 승급 버전은 Model Store의 현재 운영 버전을 따른다(하드코딩하면 실제 승급 결과와 어긋난다).
  const servingVersion = modelStore.find(
    (m) => m.modelId === POPULATION_MODEL_ID && m.status === "운영"
  )?.version;
  const f1Value = f1Override !== null ? f1Override.toFixed(3) : "0.884";
  const f1Label =
    f1Override !== null && servingVersion ? `최적 (SOTA ${servingVersion})` : "최적 (SOTA)";
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
      legend: { position: compactChart ? "bottom" : "right", labels: { color: ct.legend, boxWidth: 12, font: { size: 11 } } },
      tooltip: {
        callbacks: {
          label: (cx) => ` ${cx.label}: ${cx.parsed.toLocaleString()}행 적재`
        }
      }
    }
  };

  const sourceRows = sourceData.labels.map((label, index) => ({
    label,
    rows: sourceData.datasets[0].data[index]
  }));
  const sourceTotal = sourceRows.reduce((sum, source) => sum + source.rows, 0);
  const largestSource = sourceRows.reduce((largest, source) => source.rows > (largest?.rows ?? -1) ? source : largest, null);
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

  const radarSummary = radarData.labels
    .map((label, index) => `${label} ${radarData.datasets[0].data[index].toFixed(1)}`)
    .join(", ");

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
    // 단일 데이터셋 — 지자체명이 카드 제목·select와 중복되므로 범례 숨김(차트 수직 중앙 정렬)
    plugins: { legend: { display: false } }
  };

  return (
    <>
      <div className="grid-cols-3">
        <StatCard
          label="AI 예측 소멸위기 지역수 (전국 기준)"
          icon="fa-triangle-exclamation"
          value="89"
          unit="개소"
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
          value={sourceCount}
          unit="개 실시간"
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
          <p className="chart-summary">
            총 {sourceTotal.toLocaleString()}행 중 가장 큰 소스는 {largestSource?.label ?? "–"} {largestSource?.rows.toLocaleString() ?? 0}행입니다.
            소스별 적재량: {sourceRows.map((source) => `${source.label} ${source.rows.toLocaleString()}행`).join(", ")}.
          </p>
        </Card>

        <Card
          title={`${currentRegion.name} 정책 영향 프로파일`}
          icon="fa-bullseye"
          headerRight={
            <label className="compact-select-field">
              <span>대상 지자체</span>
              <select
                className="select-control"
                value={currentRegion.id}
                onChange={(e) => {
                  const region = appData.regions.find((r) => r.id === e.target.value);
                  if (region) setCurrentRegion(region);
                }}
              >
                {appData.regions.map((r) => (
                  <option key={r.id} value={r.id}>{r.name}</option>
                ))}
              </select>
            </label>
          }
        >
          <div style={{ position: "relative", height: 240, width: "100%" }}>
            <Radar data={radarData} options={radarOpts} />
          </div>
          <p className="chart-summary">{currentRegion.name} 지표: {radarSummary}.</p>
        </Card>
      </div>
    </>
  );
}
