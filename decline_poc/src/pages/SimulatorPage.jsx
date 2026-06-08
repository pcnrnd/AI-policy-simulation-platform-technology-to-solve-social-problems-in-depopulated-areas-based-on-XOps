import { useEffect, useMemo, useRef } from "react";
import { Line } from "react-chartjs-2";
import L from "leaflet";
import Card from "../components/Card.jsx";
import { useAppState } from "../context/AppStateContext.jsx";

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
    setHousingWeight
  } = useAppState();

  const mapContainerRef = useRef(null);
  const mapRef = useRef(null);
  const markersRef = useRef([]);

  // Initialize map once
  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return undefined;

    const map = L.map(mapContainerRef.current).setView([35.8, 128.0], 7);

    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
      subdomains: "abcd",
      maxZoom: 20
    }).addTo(map);

    appData.regions.forEach((region) => {
      const color = region.riskIndex < 0.15 ? "#ef4444" : "#f59e0b";

      const marker = L.circleMarker([region.lat, region.lng], {
        radius: 10,
        fillColor: color,
        color: "#ffffff",
        weight: 1,
        opacity: 1,
        fillOpacity: 0.8
      }).addTo(map);

      marker.bindPopup(`
        <div style="font-family: 'Inter', sans-serif;">
          <h4 style="margin-bottom:6px; color: ${color};">${region.name}</h4>
          <p style="font-size:12px; margin: 2px 0;">인구수: <strong>${region.population.toLocaleString()}명</strong></p>
          <p style="font-size:12px; margin: 2px 0;">위험지수: <strong>${region.riskIndex}</strong> (소멸위기)</p>
          <p style="font-size:11px; margin-top:6px; color: #9ca3af;">클릭하여 정책 가이드를 설정하세요.</p>
        </div>
      `);

      marker.on("click", () => {
        setCurrentRegion(region);
      });

      markersRef.current.push(marker);
    });

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
      y: { grid: { color: "rgba(255, 255, 255, 0.05)" }, ticks: { color: "#9ca3af" } },
      x: { grid: { display: false }, ticks: { color: "#9ca3af" } }
    },
    plugins: { legend: { labels: { color: "#f3f4f6", boxWidth: 12 } } }
  };

  const finalPop = simTrend[9] ?? currentRegion.population;
  const growthPercent = (((finalPop - currentRegion.population) / currentRegion.population) * 100).toFixed(1);
  const growthClass = parseFloat(growthPercent) >= 0 ? "trend-up" : "trend-down";

  return (
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
          전국 89개 인구소멸 위험 지자체 중 실시간 파라미터가 공유되는 6개 연합 사일로 노드를 맵에
          핀으로 표시합니다. 노드를 클릭하면 해당 지역의 인구 현황 및{" "}
          <strong>정책 효과 예측 시뮬레이션</strong>이 연계 활성화됩니다.
        </p>
        <div id="map" ref={mapContainerRef}></div>
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
