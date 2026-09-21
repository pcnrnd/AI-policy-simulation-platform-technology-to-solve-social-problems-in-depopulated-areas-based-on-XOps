import { useEffect, useMemo, useState } from "react";
import { Doughnut, Radar } from "react-chartjs-2";
import { useAppState } from "../context/AppStateContext.jsx";
import { useTheme } from "../context/ThemeContext.jsx";
import { useChartTheme } from "../hooks/useChartTheme.js";
import StatCard from "../components/StatCard.jsx";
import Card from "../components/Card.jsx";
import RegionStatusCard from "../components/RegionStatusCard.jsx";
import { ChartEmptyNote, NO_DEMO_DATA } from "../components/PendingData.jsx";

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
  const {
    appData,
    currentRegion,
    setCurrentRegion,
    f1Override,
    focusRegion,
    modelStore,
    overviewSummary,
    mockDataVisible
  } = useAppState();
  const ct = useChartTheme();
  const { isDark } = useTheme();
  const [compactChart, setCompactChart] = useState(() => window.matchMedia("(max-width: 768px)").matches);

  useEffect(() => {
    const media = window.matchMedia("(max-width: 768px)");
    const onChange = (event) => setCompactChart(event.matches);
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, []);

  // 데모 표시 토글은 화면 구조가 아니라 **데이터 유무**만 바꾼다. OFF에서는 시드(mock_data.json)
  // 유래 값이 실값 자리에 들어가지 않도록 여기 데이터 계층에서 차단하고, 빈 자리는 "–"와 사유 문구로 남긴다.
  const allowSeed = mockDataVisible;

  // 승급 버전은 Model Store의 현재 운영 버전을 따른다(하드코딩하면 실제 승급 결과와 어긋난다).
  const servingVersion = modelStore.find(
    (m) => m.modelId === POPULATION_MODEL_ID && m.status === "운영"
  )?.version;
  // F1은 두 경로로 실측된다 — 이번 세션의 재학습 승급(f1Override)과 롤업이 실어 주는 레지스트리
  // 스냅샷(metrics_source === "trained"). 둘 다 없으면 남는 0.884는 시드 상수다.
  const summaryModel = overviewSummary?.model ?? null;
  const measuredF1 =
    f1Override !== null
      ? f1Override
      : summaryModel?.metrics_source === "trained" && typeof summaryModel.f1 === "number"
        ? summaryModel.f1
        : null;
  const f1Value = measuredF1 !== null ? measuredF1.toFixed(3) : allowSeed ? "0.884" : "–";
  // 값이 "–" 인데 "최적 (SOTA)" 배지를 붙이면 측정이 없는 상태를 최고 성능이라고 단정한다.
  const f1Measured = measuredF1 !== null || allowSeed;
  const f1ServingVersion = servingVersion ?? summaryModel?.serving_version;
  const f1Origin = measuredF1 !== null ? "api" : "mock";
  const f1Label = !f1Measured
    ? "측정 없음"
    : measuredF1 !== null && f1ServingVersion
      ? `실측 (${f1ServingVersion})`
      : "최적 (SOTA)";
  const f1Sub =
    measuredF1 !== null
      ? f1Override !== null
        ? "연합 재학습 성공"
        : "학습 아티팩트 실측값"
      : allowSeed
        ? "데이터 소스 통합 기준"
        : "재학습 후 표시됩니다";

  // 카탈로그 롤업(GET /api/v3/overview/summary)이 실 저장소를 물고 있을 때만 API 값을 쓴다.
  // 요청 실패(overviewSummary === null)나 In-Memory degrade면 mock_data.json 으로 폴백하고,
  // 어느 쪽 값인지는 카드 컨테이너의 data-values-source 로 드러낸다(mock 스위치 계약).
  const liveSources =
    overviewSummary?.source_kind === "database" && Array.isArray(overviewSummary.sources)
      ? overviewSummary.sources
      : null;
  const catalogOrigin = liveSources ? "api" : "mock";

  // 도넛: 소스별 아카이브 적재 행 수 — "어떤 소스가 얼마나 적재돼 있는가"를 보여준다.
  // 실측 롤업에서 archive_rows 가 null 인 소스는 '확인 불가'다 — 0으로 그리면 적재량이 없는
  // 것처럼 보이므로 도넛에서 아예 뺀다. 몇 건을 못 셌는지는 아래 unknownSources 로 알린다.
  const sourceRows = useMemo(
    () =>
      liveSources
        ? liveSources
            .filter((s) => typeof s.archive_rows === "number")
            .map((s) => ({ label: s.label ?? s.id, rows: s.archive_rows }))
        : allowSeed
          ? appData.metadata_schemas.map((s) => ({ label: s.label ?? s.id, rows: s.archive?.rows ?? 0 }))
          : [],
    [liveSources, appData, allowSeed]
  );
  // 도넛에 그릴 값이 하나도 없을 때 — 캔버스는 그대로 두고 그 위에 사유만 얹는다.
  const sourceRowsEmpty = sourceRows.length === 0;
  const unknownSources = liveSources ? overviewSummary.archive_rows_unknown ?? 0 : 0;

  const sourceData = useMemo(
    () => ({
      labels: sourceRows.map((s) => s.label),
      datasets: [
        {
          data: sourceRows.map((s) => s.rows),
          backgroundColor: sourceRows.map((_, i) => SOURCE_PALETTE[i % SOURCE_PALETTE.length]),
          // 보더는 카드 배경과 동화되도록 테마별 분기 (라이트에서 검은 띠 방지)
          borderColor: isDark ? "rgba(8, 13, 26, 1)" : "#ffffff",
          borderWidth: 2
        }
      ]
    }),
    [sourceRows, isDark]
  );

  const sourceCount = liveSources
    ? overviewSummary.source_count ?? liveSources.length
    : allowSeed
      ? appData.metadata_schemas.length
      : "–";

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

  const sourceTotal = sourceRows.reduce((sum, source) => sum + source.rows, 0);
  const largestSource = sourceRows.reduce((largest, source) => source.rows > (largest?.rows ?? -1) ? source : largest, null);
  // 레이더: 선택 지자체 정책 영향 프로파일.
  // 지자체 정책영향·출산율·위험지수는 시드(mock_data.json regions) 전용이라 실저장소 대응값이 없다 —
  // 데모 OFF에서는 축 구조만 남기고 값 계열을 비운다(카드·선택 UI는 그대로 조작 가능).
  const radarData = useMemo(() => {
    const p = currentRegion.policyImpacts;
    return {
      labels: ["복지 영향", "산업 영향", "주거 영향", "출산율", "위험 완화 여력"],
      datasets: [
        {
          label: currentRegion.name,
          data: allowSeed
            ? [
                p.welfare * 100,
                p.industry * 100,
                p.housing * 100,
                currentRegion.birthRate * 100,
                (1 - currentRegion.riskIndex) * 100
              ]
            : [],
          backgroundColor: "rgba(59, 130, 246, 0.18)",
          borderColor: "rgba(59, 130, 246, 1)",
          borderWidth: 2,
          pointBackgroundColor: "rgba(16, 185, 129, 1)"
        }
      ]
    };
  }, [currentRegion, allowSeed]);

  const radarSummary = allowSeed
    ? radarData.labels
        .map((label, index) => `${label} ${radarData.datasets[0].data[index].toFixed(1)}`)
        .join(", ")
    : null;

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
        {/* 전국 소멸위기 지역수는 시드 상수다 — 전국 집계를 내려주는 실 엔드포인트가 아직 없어
            데모 OFF에서는 값을 비우고 사유만 같은 자리에 남긴다(카드 구조는 그대로). */}
        <StatCard
          label="AI 예측 소멸위기 지역수 (전국 기준)"
          icon="fa-triangle-exclamation"
          value={allowSeed ? "89" : "–"}
          unit={allowSeed ? "개소" : undefined}
          footer={
            allowSeed ? (
              <>
                <span className="trend-up">
                  <i className="fa-solid fa-caret-up"></i> 4개소
                </span>
                <span className="text-secondary">전분기 대비</span>
              </>
            ) : (
              <span className="text-secondary">{NO_DEMO_DATA}</span>
            )
          }
        />
        <StatCard
          label="글로벌 모델 F1-score"
          icon="fa-bullseye"
          dataSource={f1Origin}
          value={f1Value}
          footer={
            <>
              <span
                className="trend-up"
                style={{ color: f1Measured ? "var(--accent-teal)" : "var(--text-muted)" }}
              >
                <i className={`fa-solid ${f1Measured ? "fa-circle-check" : "fa-minus"}`}></i> {f1Label}
              </span>
              <span className="text-secondary">{f1Sub}</span>
            </>
          }
        />
        <StatCard
          label="연동 데이터 소스"
          icon="fa-network-wired"
          dataSource={catalogOrigin}
          value={sourceCount}
          unit={liveSources || allowSeed ? "개 실시간" : undefined}
          footer={
            liveSources || allowSeed ? (
              <>
                <span className="trend-up">
                  <i className="fa-solid fa-arrow-right"></i> Active
                </span>
                {/* 카테고리 문구는 시드 6종 이름이다 — 실데이터 롤업에는 그 구성이 없다. */}
                <span className="text-secondary">
                  {allowSeed ? "주민·복지·산업·공간·스마트팜·시설" : "실적재 데이터 소스"}
                </span>
              </>
            ) : (
              <span className="text-secondary">{NO_DEMO_DATA}</span>
            )
          }
        />
      </div>

      <RegionStatusCard
        regions={appData.regions}
        currentRegionId={currentRegion.id}
        onSelectRegion={focusRegion}
        allowSeed={allowSeed}
      />

      <div className="grid-cols-2">
        <Card
          title="연동 데이터 소스 아카이브 적재 현황 (행 수)"
          icon="fa-chart-pie"
          dataSource={catalogOrigin}
        >
          <div style={{ position: "relative", height: 240, width: "100%" }}>
            <Doughnut data={sourceData} options={doughnutOpts} />
            {sourceRowsEmpty && (
              <ChartEmptyNote>
                {NO_DEMO_DATA}
              </ChartEmptyNote>
            )}
          </div>
          <p className="chart-summary">
            {sourceRowsEmpty ? (
              `${NO_DEMO_DATA} — 표시할 적재량이 없습니다.`
            ) : (
              <>
                총 {sourceTotal.toLocaleString()}행 중 가장 큰 소스는 {largestSource?.label ?? "–"} {largestSource?.rows.toLocaleString() ?? 0}행입니다.
                소스별 적재량: {sourceRows.map((source) => `${source.label} ${source.rows.toLocaleString()}행`).join(", ")}.
                {/* 저장소에 닿지 못한 소스는 0으로 단정하지 않고 건수만 알린다(L0 감사 G3). */}
                {unknownSources > 0 && ` 적재 여부를 확인하지 못한 소스 ${unknownSources}건은 합계에서 제외했습니다.`}
              </>
            )}
          </p>
        </Card>

        <Card
          // 지자체 목록·이름은 시드(mock_data.json regions) 전용이라 실저장소 대응값이 없다 —
          // 데모 OFF에서는 제목·요약문에서 이름을 빼고, 선택 자체는 조작 UI라 그대로 둔다.
          title={allowSeed ? `${currentRegion.name} 정책 영향 프로파일` : "정책 영향 프로파일"}
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
            {radarSummary === null && (
              <ChartEmptyNote>
                {NO_DEMO_DATA}
              </ChartEmptyNote>
            )}
          </div>
          <p className="chart-summary">
            {radarSummary === null
              ? NO_DEMO_DATA
              : `${currentRegion.name} 지표: ${radarSummary}.`}
          </p>
        </Card>
      </div>
    </>
  );
}
