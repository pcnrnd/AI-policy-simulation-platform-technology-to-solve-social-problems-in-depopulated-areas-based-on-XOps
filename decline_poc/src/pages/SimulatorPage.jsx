import { useEffect, useMemo, useRef, useState } from "react";
import { Line, Scatter } from "react-chartjs-2";
import L from "leaflet";
import Card from "../components/Card.jsx";
import PolicyRecommendation from "../components/PolicyRecommendation.jsx";
import PipelineStepper from "../components/PipelineStepper.jsx";
import CollapsibleStage from "../components/CollapsibleStage.jsx";
import FactorAnalysisStage from "../components/FactorAnalysisStage.jsx";
import FactorResultStage from "../components/FactorResultStage.jsx";
import ScenarioCompare from "../components/ScenarioCompare.jsx";
import { useAppState } from "../context/AppStateContext.jsx";
import { useChartTheme } from "../hooks/useChartTheme.js";
import { buildRegionGrid, densityColor, DENSITY_LEGEND } from "../lib/geo.js";
import { rankStrategies } from "../constants/policyStrategies.js";
import { YEAR_LABELS, computeTrends, budgetToFactor, controlBoostOf } from "../lib/simulation.js";

// STAGE① 요인분석 실행 단계 — 실행 버튼 클릭 시 순차 진행 후 결과 공개.
const ANALYSIS_STEP_MS = 900;
const ANALYSIS_STEPS = [
  { key: "collect", label: "연계 데이터 수집", icon: "fa-database", log: "주민등록·통신·카드 연계 데이터 소스 수집 및 정합성 검증" },
  { key: "dl", label: "딥러닝 모델 추론", icon: "fa-microchip", log: "딥러닝 예측 모델 추론 수행 (시계열 패턴 학습)" },
  { key: "xai", label: "XAI 기여도 분해", icon: "fa-magnifying-glass-chart", log: "SHAP 기반 주요 요인 기여도(Shapley value) 분해" },
  { key: "fusion", label: "통계 분석 융합", icon: "fa-layer-group", log: "통계적 모델 분석 결과와 융합 — 상관·문제진단 도출" }
];

// 소멸위험지수(낮을수록 위험) → 등급/색상.
function riskGrade(ri) {
  if (ri < 0.15) return { label: "고위험", color: "var(--accent-red)", rgb: "239, 68, 68" };
  if (ri < 0.18) return { label: "주의", color: "var(--accent-orange)", rgb: "245, 158, 11" };
  return { label: "관찰", color: "var(--accent-teal)", rgb: "16, 185, 129" };
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
    budgetTotal,
    setBudgetTotal,
    addConsoleLog,
    setActiveTab
  } = useAppState();
  const ct = useChartTheme();

  // 지도 레이어 토글 / 검색 (플로팅 패널)
  const [layerVis, setLayerVis] = useState({ markers: true, grid: true, facilities: true });
  const [searchText, setSearchText] = useState("");

  // 시설 조절변수(x) — 지자체 테마별 정의(case.simulation.controls). 지역 변경 시 기본값으로 초기화.
  const controls = currentRegion.case.simulation.controls ?? [];
  const [controlValues, setControlValues] = useState(() =>
    Object.fromEntries(controls.map((c) => [c.key, c.default]))
  );
  useEffect(() => {
    const cs = currentRegion.case.simulation.controls ?? [];
    setControlValues(Object.fromEntries(cs.map((c) => [c.key, c.default])));
  }, [currentRegion]);

  // 총 예산(제약요소) → 체감수익 곡선. 시설 조절변수 → 정규화×weight 합산 부스트.
  const budgetFactor = budgetToFactor(budgetTotal);
  const controlBoost = controlBoostOf(controls, controlValues);

  // 정책 추천(실행 버튼으로 도출) 상태
  const [recommendation, setRecommendation] = useState(null);
  const [recoSig, setRecoSig] = useState("");
  const controlSig = controls.map((c) => `${c.key}:${controlValues[c.key] ?? c.default}`).join(",");
  const currentSig = `${currentRegion.id}|${welfareWeight}|${industryWeight}|${housingWeight}|${budgetTotal}|${controlSig}`;
  const recoStale = recommendation !== null && recoSig !== currentSig;

  const handleRecommend = () => {
    const ranked = rankStrategies(
      currentRegion,
      { welfare: welfareWeight, industry: industryWeight, housing: housingWeight },
      budgetFactor
    );
    setRecommendation(ranked);
    setRecoSig(currentSig);
    addConsoleLog(
      `INFO: ${currentRegion.name} 정책 추천 도출 완료 — 1순위 '${ranked[0].name}' (점수 ${ranked[0].score.toLocaleString()}).`
    );
    // 결과는 STAGE ④에 렌더되므로 — 펼친 뒤 화면으로 이동해 즉시 보이게 한다
    setOpenStages((s) => ({ ...s, "stage-report": true }));
    setTimeout(() => {
      document.getElementById("stage-report")?.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 60);
  };

  // 단계 접기/펼치기 상태 (기본 전부 펼침)
  const [openStages, setOpenStages] = useState({
    "stage-factor": true,
    "stage-result": true,
    "stage-sim": true,
    "stage-report": true
  });

  // STAGE① 요인분석 실행 상태 — idle(미실행) → running(단계 진행) → done(결과 공개).
  // STAGE②(요인분석 결과)는 done 이후에만 공개된다.
  const [analysisStatus, setAnalysisStatus] = useState("idle");
  const [analysisStep, setAnalysisStep] = useState(0); // 1-based 진행 단계
  const analysisTimerRef = useRef(null);

  // 지역 변경 시 분석 상태 초기화 (지역별 요인분석은 별도 실행)
  useEffect(() => {
    if (analysisTimerRef.current) {
      clearTimeout(analysisTimerRef.current);
      analysisTimerRef.current = null;
    }
    setAnalysisStatus("idle");
    setAnalysisStep(0);
  }, [currentRegion]);

  // 분석 단계 진행 — 단계별 로그 출력 후 완료 시 STAGE② 자동 펼침
  useEffect(() => {
    if (analysisStatus !== "running" || analysisStep === 0) return undefined;

    if (analysisStep > ANALYSIS_STEPS.length) {
      setAnalysisStatus("done");
      addConsoleLog(
        `SUCCESS: ${currentRegion.name} 사회문제 요인분석 완료 — STAGE ② 요인분석 결과가 갱신되었습니다.`
      );
      addConsoleLog(
        "INFO: 도출 파라미터(조절변수·상관변수)가 STAGE ③ 시뮬레이션 입력으로 전달되었습니다."
      );
      setOpenStages((s) => ({ ...s, "stage-result": true, "stage-sim": true }));
      // 분석 결과가 갱신된 STAGE ②로 이동해 완료가 즉시 보이게 한다
      setTimeout(() => {
        document.getElementById("stage-result")?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 120);
      return undefined;
    }

    const step = ANALYSIS_STEPS[analysisStep - 1];
    addConsoleLog(`INFO: [요인분석 ${analysisStep}/${ANALYSIS_STEPS.length}] ${step.log}`);
    analysisTimerRef.current = setTimeout(() => setAnalysisStep((s) => s + 1), ANALYSIS_STEP_MS);

    return () => {
      if (analysisTimerRef.current) clearTimeout(analysisTimerRef.current);
    };
  }, [analysisStatus, analysisStep, currentRegion, addConsoleLog]);

  const handleRunAnalysis = () => {
    if (analysisStatus === "running") return;
    setOpenStages((s) => ({ ...s, "stage-factor": true }));
    setAnalysisStatus("running");
    setAnalysisStep(1);
    addConsoleLog(`INFO: ${currentRegion.name} 사회문제 요인분석 실행 — XAI·딥러닝·통계 융합 파이프라인 시작.`);
  };

  const toggleStage = (id) => setOpenStages((s) => ({ ...s, [id]: !s[id] }));
  const allOpen = Object.values(openStages).every(Boolean);
  const setAllStages = (open) =>
    setOpenStages({
      "stage-factor": open,
      "stage-result": open,
      "stage-sim": open,
      "stage-report": open
    });
  // 스텝퍼 클릭 → 대상 단계를 펼친 뒤 스크롤 이동
  const jumpToStage = (id) => {
    setOpenStages((s) => ({ ...s, [id]: true }));
    setTimeout(() => {
      document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 60);
  };

  // 스크롤 스파이 — 화면 상단 밴드에 걸린 단계를 스텝퍼에 하이라이트
  const [activeStageId, setActiveStageId] = useState("stage-factor");
  useEffect(() => {
    const ids = ["stage-factor", "stage-result", "stage-sim", "stage-report"];
    const visibility = new Map();
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => visibility.set(e.target.id, e.isIntersecting));
        const active = ids.find((id) => visibility.get(id));
        if (active) setActiveStageId(active);
      },
      { rootMargin: "-10% 0px -60% 0px" }
    );
    ids.forEach((id) => {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    });
    return () => observer.disconnect();
  }, []);

  const mapContainerRef = useRef(null);
  const mapRef = useRef(null);
  const markerLayerRef = useRef(null);
  const gridLayerRef = useRef(null);
  const facilityLayerRef = useRef(null);

  // 지도 1회 초기화 — 줌 컨트롤은 우하단, 레이어 토글은 커스텀 플로팅 패널이 담당
  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return undefined;

    const map = L.map(mapContainerRef.current, { zoomControl: false }).setView([35.1, 126.9], 8);
    L.control.zoom({ position: "bottomright" }).addTo(map);

    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
      subdomains: "abcd",
      maxZoom: 20
    }).addTo(map);

    const markerLayer = L.layerGroup();
    const gridLayer = L.layerGroup();
    const facilityLayer = L.layerGroup();

    // 시설물 공간정보 표시 — 정주여건·스마트팜 시설의 위치와 데이터를 표출 (레이어 적층)
    const FACILITY_COLORS = {
      스마트팜: "#10b981",
      임대주택: "#3b82f6",
      신재생: "#f59e0b",
      정원수: "#22d3ee"
    };
    (appData.facilities ?? []).forEach((f) => {
      const fColor = FACILITY_COLORS[f.type] ?? "#94a3b8";
      const fMarker = L.circleMarker([f.lat, f.lng], {
        radius: 6,
        fillColor: fColor,
        color: "#0b1220",
        weight: 1,
        opacity: 1,
        fillOpacity: 0.9
      });
      fMarker.bindPopup(`
        <div style="font-family: 'Inter', sans-serif;">
          <h4 style="margin-bottom:6px; color: ${fColor};">${f.name}</h4>
          <p style="font-size:12px; margin: 2px 0;">시설 유형: <strong>${f.type}</strong></p>
          <p style="font-size:12px; margin: 2px 0;">${f.metric}</p>
          <p style="font-size:12px; margin: 2px 0;">운영 예산: <strong>${f.operating_budget}억</strong></p>
        </div>
      `);
      fMarker.addTo(facilityLayer);
    });

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
          <p style="font-size:11px; margin-top:6px; color: var(--text-muted);">우측 패널에서 정책 변수를 설정하세요.</p>
        </div>
      `);
      marker.on("click", () => setCurrentRegion(region));
      marker.addTo(markerLayer);

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
    gridLayer.addTo(map);
    facilityLayer.addTo(map);
    markerLayerRef.current = markerLayer;
    gridLayerRef.current = gridLayer;
    facilityLayerRef.current = facilityLayer;
    mapRef.current = map;

    setTimeout(() => map.invalidateSize(), 100);

    return () => {
      map.remove();
      mapRef.current = null;
      markerLayerRef.current = null;
      gridLayerRef.current = null;
      facilityLayerRef.current = null;
    };
  }, [appData, setCurrentRegion]);

  // 레이어 토글 반영
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const sync = (layer, visible) => {
      if (!layer) return;
      if (visible) map.addLayer(layer);
      else map.removeLayer(layer);
    };
    sync(markerLayerRef.current, layerVis.markers);
    sync(gridLayerRef.current, layerVis.grid);
    sync(facilityLayerRef.current, layerVis.facilities);
  }, [layerVis]);

  // 선택 지자체로 지도 이동
  useEffect(() => {
    if (mapRef.current) mapRef.current.panTo([currentRegion.lat, currentRegion.lng]);
  }, [currentRegion]);

  // STAGE③ 재펼침/잠금해제 시 지도 크기 재계산 (display:none·blur 해제 후 타일 깨짐 방지)
  useEffect(() => {
    if (!openStages["stage-sim"] || !mapRef.current) return undefined;
    const t = setTimeout(() => mapRef.current?.invalidateSize(), 200);
    return () => clearTimeout(t);
  }, [openStages, analysisStatus]);

  const { baseTrend, simTrend } = useMemo(
    () => computeTrends(currentRegion, welfareWeight, industryWeight, housingWeight, budgetFactor, controlBoost),
    [currentRegion, welfareWeight, industryWeight, housingWeight, budgetFactor, controlBoost]
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

  const scatterData = useMemo(
    () => ({
      datasets: [
        {
          // 전국 인구감소 지역 분포 속에서 분석 대상의 상대 위치를 보여주는 배경 점
          label: "전국 비교 지자체",
          data: (appData.benchmark_regions ?? []).map((r) => ({
            x: r.riskIndex,
            y: r.population,
            name: r.name
          })),
          backgroundColor: "rgba(148, 163, 184, 0.55)",
          pointRadius: 5,
          pointHoverRadius: 8
        },
        {
          label: "분석 대상 지자체",
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
          label: (cx) => `${cx.raw.name}: 위험 ${cx.raw.x}, 인구 ${cx.raw.y.toLocaleString()}명`
        }
      }
    }
  };

  const filteredRegions = appData.regions.filter((r) =>
    r.name.toLowerCase().includes(searchText.trim().toLowerCase())
  );

  const sim = currentRegion.case.simulation;
  const analysisDone = analysisStatus === "done";
  const analysisRunning = analysisStatus === "running";

  // 스텝퍼 완료 표시: ①② 요인분석 완료, ③④ 정책 추천 도출 완료 기준
  const doneStages = [
    ...(analysisDone ? ["stage-factor", "stage-result"] : []),
    ...(recommendation && !recoStale ? ["stage-sim", "stage-report"] : [])
  ];

  return (
    <>
      <div className="pl-toolbar">
        <PipelineStepper activeId={activeStageId} doneIds={doneStages} onJump={jumpToStage} />
        <button type="button" className="pl-collapse-all" onClick={() => setAllStages(!allOpen)}>
          <i className={`fa-solid ${allOpen ? "fa-compress" : "fa-expand"}`} aria-hidden="true"></i>
          {allOpen ? "모두 접기" : "모두 펼치기"}
        </button>
      </div>

      {/* ── STAGE ① 사회문제 요인분석 ── */}
      <FactorAnalysisStage
        region={currentRegion}
        open={openStages["stage-factor"]}
        onToggle={() => toggleStage("stage-factor")}
        status={analysisStatus}
        stepIndex={analysisStep}
        steps={ANALYSIS_STEPS}
        onRun={handleRunAnalysis}
      />

      {/* ── STAGE ② 요인분석 결과 ── */}
      <FactorResultStage
        region={currentRegion}
        open={openStages["stage-result"]}
        onToggle={() => toggleStage("stage-result")}
        locked={analysisStatus !== "done"}
        onRun={handleRunAnalysis}
        running={analysisStatus === "running"}
      />

      {/* ── STAGE ③ 시뮬레이션: 공간정보 탐색 + 강화학습 변수 + 인구 예측 ── */}
      <CollapsibleStage
        id="stage-sim"
        no="STAGE ③"
        title="시뮬레이션"
        sub="복잡계 기반 강화학습 — 조절변수(x)·상관변수(y)·제약요소(예산)"
        open={openStages["stage-sim"]}
        onToggle={() => toggleStage("stage-sim")}
      >
        <div className="pl-result-zone">
        <div className={analysisDone ? "" : "pl-locked"}>
        {/* 강화학습 변수 프레임 (case.simulation) */}
        <div className="pl-sim-frame">
          <div className="pl-sim-cell pl-sim-obj">
            <span className="pl-sim-tag">목적함수</span>
            {sim.objective}
            {analysisDone && (
              <span className="pl-param-chip">
                <i className="fa-solid fa-circle-check" aria-hidden="true"></i> STAGE ② 도출 파라미터 적용
              </span>
            )}
          </div>
          <div className="pl-sim-vars">
            <div className="pl-sim-cell">
              <span className="pl-sim-tag">수요 (x)</span>
              {sim.factors.demand}
            </div>
            <div className="pl-sim-cell">
              <span className="pl-sim-tag">공급 (y)</span>
              {sim.factors.supply}
            </div>
            <div className="pl-sim-cell">
              <span className="pl-sim-tag">조절변수</span>
              {sim.factors.adjust}
            </div>
            <div className="pl-sim-cell pl-sim-constraint">
              <span className="pl-sim-tag">제약요소</span>
              {sim.constraint} · {budgetTotal.toLocaleString()}억
            </div>
          </div>
        </div>

      {/* 공간정보 탐색: 전체 지도 + 플로팅 패널 + 우측 지자체 독 */}
      <div className="sim-map-zone">
        <div className="sim-map-wrap">
          <div id="map" ref={mapContainerRef}></div>

          {/* 플로팅: 검색 + 레이어 + 범례 */}
          <div className="map-float map-float-tl">
            <div className="map-float-search">
              <i className="fa-solid fa-magnifying-glass" aria-hidden="true"></i>
              <input
                type="text"
                placeholder="지자체 검색"
                value={searchText}
                onChange={(e) => setSearchText(e.target.value)}
                aria-label="지자체 검색"
              />
            </div>
            <div className="map-float-section-label">표시 레이어</div>
            <label className="map-float-toggle">
              <input
                type="checkbox"
                checked={layerVis.markers}
                onChange={(e) => setLayerVis((v) => ({ ...v, markers: e.target.checked }))}
              />
              위험등급 마커
            </label>
            <label className="map-float-toggle">
              <input
                type="checkbox"
                checked={layerVis.grid}
                onChange={(e) => setLayerVis((v) => ({ ...v, grid: e.target.checked }))}
              />
              인구밀도 격자
            </label>
            <label className="map-float-toggle">
              <input
                type="checkbox"
                checked={layerVis.facilities}
                onChange={(e) => setLayerVis((v) => ({ ...v, facilities: e.target.checked }))}
              />
              시설물 (정주여건·스마트팜)
            </label>
            <div className="map-float-section-label">인구밀도(㎢당)</div>
            <div className="map-float-legend">
              {DENSITY_LEGEND.map((d) => (
                <span key={d.label} className="legend-item">
                  <span className="legend-swatch" style={{ backgroundColor: d.color }}></span>
                  {d.label}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* 우측 독: 분석 대상 지자체 목록 + 정책 변수 제어 */}
        <div className="sim-dock">
          <div className="sim-dock-section">
            <div className="sim-dock-title">
              <i className="fa-solid fa-list-ul"></i> 분석 대상 지자체
              <span className="sim-dock-count">{filteredRegions.length}</span>
            </div>
            <div className="sim-region-list">
              {filteredRegions.map((r) => {
                const g = riskGrade(r.riskIndex);
                const active = r.id === currentRegion.id;
                return (
                  <button
                    type="button"
                    key={r.id}
                    className={"sim-region-item" + (active ? " active" : "")}
                    onClick={() => setCurrentRegion(r)}
                  >
                    <div className="sim-region-head">
                      <strong>{r.name}</strong>
                      <span
                        className="grade-chip"
                        style={{
                          color: g.color,
                          backgroundColor: `rgba(${g.rgb}, 0.12)`,
                          borderColor: `rgba(${g.rgb}, 0.3)`
                        }}
                      >
                        {g.label}
                      </span>
                    </div>
                    <div className="sim-region-meta">
                      {r.theme} · {r.population.toLocaleString()}명 · 위험 {r.riskIndex}
                    </div>
                  </button>
                );
              })}
              {filteredRegions.length === 0 && (
                <div className="sim-region-empty">검색 결과가 없습니다.</div>
              )}
            </div>
          </div>

          <div className="sim-dock-section">
            <div className="sim-dock-title">
              <i className="fa-solid fa-sliders"></i> {currentRegion.name} 시뮬레이션 변수
            </div>
            <p className="sim-dock-hint">
              예산·배분·시설 변수를 조정하면 10년 인구 예측이 실시간 갱신됩니다.
            </p>

            <div className="sim-var-group-label">
              <i className="fa-solid fa-lock" aria-hidden="true"></i> 제약요소 · 총 예산
            </div>
            <Slider
              label="총 예산"
              value={budgetTotal}
              onChange={setBudgetTotal}
              min={100}
              max={2000}
              step={50}
              unit="억"
            />

            <div className="sim-var-group-label">예산 배분 (정책 변수)</div>
            <Slider label="복지 예산 (Welfare)" value={welfareWeight} onChange={setWelfareWeight} />
            <Slider label="산업 일자리 (Industry)" value={industryWeight} onChange={setIndustryWeight} />
            <Slider label="주거·주택 (Housing)" value={housingWeight} onChange={setHousingWeight} />

            {controls.length > 0 && (
              <div className="sim-var-group-label">
                <i className="fa-solid fa-arrows-up-down-left-right" aria-hidden="true"></i> 시설 조절변수 (x)
              </div>
            )}
            {controls.map((c) => (
              <Slider
                key={c.key}
                label={c.label}
                value={controlValues[c.key] ?? c.default}
                onChange={(v) => setControlValues((prev) => ({ ...prev, [c.key]: v }))}
                min={c.min}
                max={c.max}
                step={c.step ?? 1}
                unit={c.unit}
              />
            ))}

            <button className="btn btn-primary" style={{ width: "100%", marginTop: 8 }} onClick={handleRecommend}>
              <i className="fa-solid fa-lightbulb"></i> 정책 추천 도출
            </button>
            {recoStale && (
              <p style={{ fontSize: 11, color: "var(--accent-orange)", marginTop: 8, textAlign: "center" }}>
                <i className="fa-solid fa-triangle-exclamation"></i> 변수가 변경되었습니다. 추천을 다시 도출하세요.
              </p>
            )}
          </div>
        </div>
      </div>

      {/* ── 분석: 인구 예측 + 전 지자체 분포 ── */}
      <div className="grid-cols-2" style={{ marginTop: 24 }}>
        <Card title="10개년 인구 예측 시뮬레이션 결과" icon="fa-wand-magic-sparkles">
          <div style={{ position: "relative", height: 200, width: "100%" }}>
            <Line data={chartData} options={chartOpts} />
          </div>
          <div style={{ display: "flex", justifyContent: "space-around", marginTop: 16, textAlign: "center" }}>
            <div>
              <div style={{ fontSize: 11, color: "var(--text-secondary)" }}>시뮬레이션 인구 (10년 후)</div>
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

        <Card title="전 지자체 위험지수 vs 인구 분포" icon="fa-braille">
          <p style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 12 }}>
            전국 인구감소 지역(회색 점) 분포 속 시범 분석 대상 지자체의 상대 위치. 파란 점이 현재 선택
            지자체입니다.
          </p>
          <div style={{ position: "relative", height: 220, width: "100%" }}>
            <Scatter data={scatterData} options={scatterOpts} />
          </div>
        </Card>
      </div>

      {/* ── 시나리오 저장·비교 ── */}
      <ScenarioCompare
        region={currentRegion}
        snapshot={{
          budgetTotal,
          welfareWeight,
          industryWeight,
          housingWeight,
          controlValues
        }}
        onApply={(sc) => {
          setBudgetTotal(sc.budgetTotal);
          setWelfareWeight(sc.welfareWeight);
          setIndustryWeight(sc.industryWeight);
          setHousingWeight(sc.housingWeight);
          setControlValues((prev) => ({ ...prev, ...(sc.controlValues ?? {}) }));
          addConsoleLog(`INFO: 시나리오 '${sc.name}' 변수를 시뮬레이션에 적용했습니다.`);
        }}
        addConsoleLog={addConsoleLog}
      />
        </div>

        {!analysisDone && (
          <div className="pl-result-overlay" role="status">
            <i
              className={"fa-solid " + (analysisRunning ? "fa-spinner fa-spin" : "fa-lock")}
              aria-hidden="true"
            ></i>
            <strong>{analysisRunning ? "요인분석 진행 중..." : "시뮬레이션 대기"}</strong>
            <p>
              {analysisRunning
                ? "분석이 완료되면 도출 파라미터로 시뮬레이션이 활성화됩니다."
                : "STAGE ①에서 요인분석을 실행하면 도출된 파라미터로 시뮬레이션이 활성화됩니다."}
            </p>
            {!analysisRunning && (
              <button type="button" className="btn btn-primary" onClick={handleRunAnalysis}>
                <i className="fa-solid fa-play"></i> 요인분석 실행
              </button>
            )}
          </div>
        )}
        </div>
      </CollapsibleStage>

      {/* ── STAGE ④ 리포팅: 정책 추천 + AI 보고서 연계 ── */}
      <CollapsibleStage
        id="stage-report"
        no="STAGE ④"
        title="리포팅"
        sub={currentRegion.case.reportFocus}
        open={openStages["stage-report"]}
        onToggle={() => toggleStage("stage-report")}
      >
        <div className="pl-result-zone">
        <div className={analysisDone ? "" : "pl-locked"}>
        {recommendation ? (
          <PolicyRecommendation region={currentRegion} ranked={recommendation} />
        ) : (
          <Card title="맞춤 정책 추천" icon="fa-lightbulb">
            <div style={{ textAlign: "center", padding: "32px 16px", color: "var(--text-secondary)", fontSize: 13 }}>
              <i
                className="fa-solid fa-wand-magic-sparkles"
                style={{ fontSize: 28, color: "var(--accent-blue)", marginBottom: 12, display: "block" }}
              ></i>
              STAGE ③에서 정책 변수를 조정한 뒤 <strong>[정책 추천 도출]</strong> 버튼을 누르면, 시뮬레이션
              결과에 맞는 맞춤 정책 추천이 생성됩니다.
            </div>
          </Card>
        )}

        <div className="pl-report-handoff">
          <div className="pl-report-handoff-text">
            <i className="fa-solid fa-file-invoice" aria-hidden="true"></i>
            <div>
              <strong>AI 정책 보고서 생성 및 평가</strong>
              <p>{currentRegion.name} · {currentRegion.case.reportFocus}</p>
            </div>
          </div>
          <button className="btn btn-primary" onClick={() => setActiveTab("tab-reporter")}>
            보고서 생성으로 이동 <i className="fa-solid fa-arrow-right"></i>
          </button>
        </div>
        </div>

        {!analysisDone && (
          <div className="pl-result-overlay" role="status">
            <i
              className={"fa-solid " + (analysisRunning ? "fa-spinner fa-spin" : "fa-lock")}
              aria-hidden="true"
            ></i>
            <strong>{analysisRunning ? "요인분석 진행 중..." : "리포팅 대기"}</strong>
            <p>요인분석 → 시뮬레이션 → 정책 추천이 완료되면 보고서 연계가 활성화됩니다.</p>
          </div>
        )}
        </div>
      </CollapsibleStage>
    </>
  );
}

function Slider({ label, value, onChange, min = 0, max = 100, step = 1, unit = "%" }) {
  return (
    <div className="slider-container">
      <div className="slider-header">
        <span>{label}</span>
        <span className="slider-val">
          {value.toLocaleString()}
          {unit}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseInt(e.target.value, 10))}
      />
    </div>
  );
}
