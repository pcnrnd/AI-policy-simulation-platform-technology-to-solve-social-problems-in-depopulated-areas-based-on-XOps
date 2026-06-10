import { useEffect, useMemo, useRef, useState } from "react";
import { Line, Scatter } from "react-chartjs-2";
import L from "leaflet";
import Card from "../components/Card.jsx";
import PolicyRecommendation from "../components/PolicyRecommendation.jsx";
import { useAppState } from "../context/AppStateContext.jsx";
import { useChartTheme } from "../hooks/useChartTheme.js";
import { buildRegionGrid, densityColor, DENSITY_LEGEND } from "../lib/geo.js";
import { rankStrategies } from "../constants/policyStrategies.js";

const YEAR_LABELS = ["2026", "2027", "2028", "2029", "2030", "2031", "2032", "2033", "2034", "2035"];
const NATURAL_DECLINE = 0.98;

function computeTrends(region, welfare, industry, housing) {
  const basePop = region.population;
  const combined =
    region.policyImpacts.welfare * (welfare / 100) +
    region.policyImpacts.industry * (industry / 100) +
    region.policyImpacts.housing * (housing / 100);

  const growthModifier = combined * 0.06;
  const simRate = NATURAL_DECLINE + growthModifier;

  const baseTrend = [];
  const simTrend = [];
  let basePopTracker = basePop;
  let simPopTracker = basePop;

  for (let i = 0; i < 10; i++) {
    basePopTracker = Math.round(basePopTracker * NATURAL_DECLINE);
    simPopTracker = Math.round(simPopTracker * simRate);
    baseTrend.push(basePopTracker);
    simTrend.push(simPopTracker);
  }
  return { baseTrend, simTrend };
}

export default function SimulatorPage() {
  const {
    appData,
    currentRegion,
    setCurrentRegion,
    welfareWeight,
    setWelfareWeight,
    industryWeight,
    setIndustryWeight,
    housingWeight,
    setHousingWeight,
    addConsoleLog
  } = useAppState();
  const ct = useChartTheme();

  // 정책 추천(실행 버튼으로 도출) 상태
  const [recommendation, setRecommendation] = useState(null);
  const [recoSig, setRecoSig] = useState("");
  const currentSig = `${currentRegion.id}|${welfareWeight}|${industryWeight}|${housingWeight}`;
  const recoStale = recommendation !== null && recoSig !== currentSig;

  const handleRecommend = () => {
    const ranked = rankStrategies(currentRegion, {
      welfare: welfareWeight,
      industry: industryWeight,
      housing: housingWeight
    });
    setRecommendation(ranked);
    setRecoSig(currentSig);
    addConsoleLog(
      `INFO: ${currentRegion.name} 정책 추천 도출 완료 — 1순위 '${ranked[0].name}' (점수 ${ranked[0].score.toLocaleString()}).`
    );
  };

  const mapContainerRef = useRef(null);
  const mapRef = useRef(null);
  const markersRef = useRef([]);

  // Initialize map once
  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return undefined;

    const map = L.map(mapContainerRef.current).setView([35.1, 126.9], 8);

    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
      subdomains: "abcd",
      maxZoom: 20
    }).addTo(map);

    // 레이어 1: 위험등급 마커 / 레이어 2: 인구밀도 격자 단계구분도
    const markerLayer = L.layerGroup();
    const gridLayer = L.layerGroup();

    appData.regions.forEach((region) => {
      const color = region.riskIndex < 0.15 ? "#ef4444" : "#f59e0b";

      const marker = L.circleMarker([region.lat, region.lng], {
        radius: 10,
        fillColor: color,
        color: "#ffffff",
        weight: 1,
        opacity: 1,
        fillOpacity: 0.8
      });

      marker.bindPopup(`
        <div style="font-family: 'Inter', sans-serif;">
          <h4 style="margin-bottom:6px; color: ${color};">${region.name}</h4>
          <p style="font-size:12px; margin: 2px 0;">인구수: <strong>${region.population.toLocaleString()}명</strong></p>
          <p style="font-size:12px; margin: 2px 0;">위험지수: <strong>${region.riskIndex}</strong> (소멸위기)</p>
          <p style="font-size:11px; margin-top:6px; color: var(--text-muted);">클릭하여 정책 가이드를 설정하세요.</p>
        </div>
      `);

      marker.on("click", () => {
        setCurrentRegion(region);
      });

      marker.addTo(markerLayer);
      markersRef.current.push(marker);

      // 격자 폴리곤(인구밀도 단계구분도)
      buildRegionGrid(region).forEach((cell) => {
        const rect = L.rectangle(cell.bounds, {
          color: "#0b1220",
          weight: 0.5,
          fillColor: densityColor(cell.density),
          fillOpacity: 0.55
        });
        rect.bindPopup(
          `<div style="font-family:'Inter',sans-serif;font-size:12px;">` +
            `격자 ${cell.gridId}<br/>㎢당 인구밀도: <strong>${cell.density}</strong></div>`
        );
        rect.addTo(gridLayer);
      });
    });

    markerLayer.addTo(map);
    L.control
      .layers(
        null,
        { "위험등급 마커": markerLayer, "인구밀도 격자 단계구분도": gridLayer },
        { collapsed: false, position: "topright" }
      )
      .addTo(map);

    mapRef.current = map;

    // Force layout recompute after mount
    setTimeout(() => map.invalidateSize(), 100);

    return () => {
      markersRef.current.forEach((m) => m.remove());
      markersRef.current = [];
      map.remove();
      mapRef.current = null;
    };
  }, [appData, setCurrentRegion]);

  const { baseTrend, simTrend } = useMemo(
    () => computeTrends(currentRegion, welfareWeight, industryWeight, housingWeight),
    [currentRegion, welfareWeight, industryWeight, housingWeight]
  );

  const chartData = useMemo(
    () => ({
      labels: YEAR_LABELS,
      datasets: [
        {
          label: "자연 감소 예측 추이 (Base Model)",
          data: baseTrend,
          borderColor: "rgba(239, 68, 68, 0.65)",
          borderDash: [5, 5],
          fill: false,
          tension: 0.2
        },
        {
          label: "정책 시뮬레이션 적용 예측 추이",
          data: simTrend,
          borderColor: "rgba(59, 130, 246, 1)",
          backgroundColor: "rgba(59, 130, 246, 0.05)",
          fill: true,
          tension: 0.2
        }
      ]
    }),
    [baseTrend, simTrend]
  );

  const chartOpts = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: { grid: { color: ct.grid }, ticks: { color: ct.tick } },
      x: { grid: { display: false }, ticks: { color: ct.tick } }
    },
    plugins: { legend: { labels: { color: ct.legend, boxWidth: 12 } } }
  };

  const finalPop = simTrend[9] ?? currentRegion.population;
  const growthPercent = (((finalPop - currentRegion.population) / currentRegion.population) * 100).toFixed(1);
  const growthClass = parseFloat(growthPercent) >= 0 ? "trend-up" : "trend-down";

  // 산점도: 전 지자체 위험지수(x) vs 인구수(y), 선택 지자체 강조
  const scatterData = useMemo(
    () => ({
      datasets: [
        {
          label: "지자체",
          data: appData.regions.map((r) => ({ x: r.riskIndex, y: r.population, name: r.name })),
          backgroundColor: appData.regions.map((r) =>
            r.id === currentRegion.id ? "rgba(59, 130, 246, 1)" : "rgba(245, 158, 11, 0.7)"
          ),
          pointRadius: appData.regions.map((r) => (r.id === currentRegion.id ? 10 : 6)),
          pointHoverRadius: 12
        }
      ]
    }),
    [appData, currentRegion]
  );

  const scatterOpts = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        title: { display: true, text: "인구소멸 위험지수 (낮을수록 위험)", color: ct.tick },
        grid: { color: ct.grid },
        ticks: { color: ct.tick }
      },
      y: {
        title: { display: true, text: "인구수(명)", color: ct.tick },
        grid: { color: ct.grid },
        ticks: { color: ct.tick }
      }
    },
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: (ctx) =>
            `${ctx.raw.name}: 위험 ${ctx.raw.x}, 인구 ${ctx.raw.y.toLocaleString()}명`
        }
      }
    }
  };

  return (
    <>
    <div className="grid-details-split">
      <Card
        title="지자체별 인구감소 위험등급 분포 지도"
        icon="fa-map-location-dot"
        headerRight={
          <span className="system-status" style={{ fontSize: 11 }}>
            Leaflet 공간정보 매핑 연동
          </span>
        }
      >
        <p style={{ fontSize: 13, color: "var(--text-secondary)", marginBottom: 16 }}>
          전국 89개 인구감소지역 중 실시간 파라미터가 공유되는 {appData.regions.length}개 시범 분석
          대상 지자체를 맵에 핀으로 표시합니다. 핀을 클릭하면 해당 지역의 인구 현황 및{" "}
          <strong>정책 효과 예측 시뮬레이션</strong>이 연계 활성화됩니다.
        </p>
        <div id="map" ref={mapContainerRef}></div>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 10,
            marginTop: 12,
            flexWrap: "wrap"
          }}
        >
          <span style={{ fontSize: 11, color: "var(--text-secondary)" }}>
            인구밀도 단계(㎢당):
          </span>
          {DENSITY_LEGEND.map((d) => (
            <span
              key={d.label}
              style={{ display: "inline-flex", alignItems: "center", gap: 4, fontSize: 11 }}
            >
              <span
                style={{
                  width: 12,
                  height: 12,
                  borderRadius: 2,
                  backgroundColor: d.color,
                  display: "inline-block"
                }}
              ></span>
              {d.label}
            </span>
          ))}
        </div>
      </Card>

      <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
        <Card title={`${currentRegion.name} 정책 변수 제어`} icon="fa-sliders">
          <p style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 16 }}>
            3대 맞춤형 정책 예산 가중치 슬라이더를 조정하면, AI 모델이 실시간으로 10년 후의
            출산율/인구수 변화를 즉시 시뮬레이션합니다.
          </p>

          <Slider
            label="1. 청년 영유아 복지 예산 가중치 (Welfare)"
            value={welfareWeight}
            onChange={setWelfareWeight}
          />
          <Slider
            label="2. 제조업 일자리 유치 세제 혜택 (Industry)"
            value={industryWeight}
            onChange={setIndustryWeight}
          />
          <Slider
            label="3. 청년 전세 자금 및 주택 공급 (Housing)"
            value={housingWeight}
            onChange={setHousingWeight}
          />

          <button
            className="btn btn-primary"
            style={{ width: "100%", marginTop: 8 }}
            onClick={handleRecommend}
          >
            <i className="fa-solid fa-lightbulb"></i> 정책 추천 도출
          </button>
          {recoStale && (
            <p style={{ fontSize: 11, color: "var(--accent-orange)", marginTop: 8, textAlign: "center" }}>
              <i className="fa-solid fa-triangle-exclamation"></i> 정책 변수가 변경되었습니다. [정책 추천
              도출]을 다시 실행하세요.
            </p>
          )}
        </Card>

        <Card title="10개년 인구 예측 시뮬레이션 결과" icon="fa-wand-magic-sparkles">
          <div style={{ position: "relative", height: 180, width: "100%" }}>
            <Line data={chartData} options={chartOpts} />
          </div>
          <div
            style={{
              display: "flex",
              justifyContent: "space-around",
              marginTop: 16,
              textAlign: "center"
            }}
          >
            <div>
              <div style={{ fontSize: 11, color: "var(--text-secondary)" }}>
                시뮬레이션 인구 (10년 후)
              </div>
              <div style={{ fontSize: 18, fontWeight: 700, color: "var(--accent-blue)" }}>
                {finalPop.toLocaleString()}명
              </div>
            </div>
            <div>
              <div style={{ fontSize: 11, color: "var(--text-secondary)" }}>인구 증가율 예측</div>
              <div className={growthClass} style={{ fontSize: 18, fontWeight: 700 }}>
                {parseFloat(growthPercent) > 0 ? "+" : ""}
                {growthPercent}%
              </div>
            </div>
          </div>
        </Card>
      </div>
    </div>

    {recommendation ? (
      <PolicyRecommendation region={currentRegion} ranked={recommendation} />
    ) : (
      <Card title="맞춤 정책 추천" icon="fa-lightbulb" style={{ marginTop: 24 }}>
        <div
          style={{
            textAlign: "center",
            padding: "32px 16px",
            color: "var(--text-secondary)",
            fontSize: 13
          }}
        >
          <i
            className="fa-solid fa-wand-magic-sparkles"
            style={{ fontSize: 28, color: "var(--accent-blue)", marginBottom: 12, display: "block" }}
          ></i>
          정책 변수(복지·산업·주거 가중치)를 조정한 뒤{" "}
          <strong>[정책 추천 도출]</strong> 버튼을 누르면, 시뮬레이션 결과에 맞는 맞춤 정책 추천이
          생성됩니다.
        </div>
      </Card>
    )}

    <Card title="전 지자체 위험지수 vs 인구 분포 (산점도)" icon="fa-braille">
      <p style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 12 }}>
        시범 분석 대상 지자체의 인구소멸 위험지수와 인구 규모를 산점도로 비교합니다. 파란 점이 현재
        선택된 지자체입니다.
      </p>
      <div style={{ position: "relative", height: 260, width: "100%" }}>
        <Scatter data={scatterData} options={scatterOpts} />
      </div>
    </Card>
    </>
  );
}

function Slider({ label, value, onChange }) {
  return (
    <div className="slider-container">
      <div className="slider-header">
        <span>{label}</span>
        <span className="slider-val">{value}%</span>
      </div>
      <input
        type="range"
        min="0"
        max="100"
        value={value}
        onChange={(e) => onChange(parseInt(e.target.value, 10))}
      />
    </div>
  );
}
