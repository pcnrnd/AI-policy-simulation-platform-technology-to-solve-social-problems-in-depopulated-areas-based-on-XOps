import { useEffect, useMemo, useState } from "react";
import { Line } from "react-chartjs-2";
import Card from "./Card.jsx";
import { useChartTheme } from "../hooks/useChartTheme.js";
import { YEAR_LABELS, computeTrends, computeScenarioTrend, budgetToFactor, controlBoostOf } from "../lib/simulation.js";

const STORAGE_KEY = "decline_poc_scenarios";
const MAX_PER_REGION = 8;
const SCENARIO_COLORS = ["#a78bfa", "#f59e0b", "#10b981", "#ec4899", "#22d3ee", "#f97316", "#84cc16", "#e879f9"];

function loadScenarios() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.filter((s) => s && s.id && s.regionId) : [];
  } catch {
    return [];
  }
}

function persistScenarios(scenarios) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(scenarios));
  } catch {
    // 저장공간 초과 등 — 비교 기능은 메모리 상태로 계속 동작
  }
}

// 슬라이더 변수 조합을 "시나리오"로 저장하고, 10개년 인구 추이를 나란히 비교한다.
// 시나리오는 지자체(regionId) 단위로 관리되며 localStorage에 보존된다.
export default function ScenarioCompare({ region, snapshot, onApply, addConsoleLog }) {
  const ct = useChartTheme();
  const [scenarios, setScenarios] = useState(loadScenarios);
  const [name, setName] = useState("");
  const [compared, setCompared] = useState(() => new Set());

  const regionScenarios = scenarios.filter((s) => s.regionId === region.id);

  // 지역 변경 시 비교 선택을 해당 지역 시나리오 전체로 초기화
  useEffect(() => {
    setCompared(new Set(loadScenarios().filter((s) => s.regionId === region.id).map((s) => s.id)));
    setName("");
  }, [region.id]);

  const updateScenarios = (next) => {
    setScenarios(next);
    persistScenarios(next);
  };

  const handleSave = () => {
    if (regionScenarios.length >= MAX_PER_REGION) {
      addConsoleLog(`WARN: 시나리오는 지자체당 최대 ${MAX_PER_REGION}개까지 저장됩니다.`, false, true);
      return;
    }
    const label = name.trim() || `시나리오 ${regionScenarios.length + 1}`;
    const scenario = {
      id: `${region.id}-${Date.now()}`,
      regionId: region.id,
      name: label,
      createdAt: new Date().toISOString(),
      ...snapshot
    };
    updateScenarios([...scenarios, scenario]);
    setCompared((prev) => new Set(prev).add(scenario.id));
    setName("");
    addConsoleLog(
      `INFO: 시나리오 '${label}' 저장 — 예산 ${scenario.budgetTotal.toLocaleString()}억, ` +
        `복지 ${scenario.welfareWeight}/산업 ${scenario.industryWeight}/주거 ${scenario.housingWeight}.`
    );
  };

  const handleDelete = (id) => {
    updateScenarios(scenarios.filter((s) => s.id !== id));
    setCompared((prev) => {
      const next = new Set(prev);
      next.delete(id);
      return next;
    });
  };

  const toggleCompare = (id) =>
    setCompared((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  // 현재 설정 + 비교 대상 시나리오의 추이 계산
  const controls = region.case?.simulation?.controls ?? [];
  const currentTrend = useMemo(
    () =>
      computeTrends(
        region,
        snapshot.welfareWeight,
        snapshot.industryWeight,
        snapshot.housingWeight,
        budgetToFactor(snapshot.budgetTotal),
        controlBoostOf(controls, snapshot.controlValues)
      ),
    [region, snapshot, controls]
  );

  const comparedScenarios = regionScenarios.filter((s) => compared.has(s.id));
  const scenarioTrends = useMemo(
    () => comparedScenarios.map((s) => ({ scenario: s, trend: computeScenarioTrend(region, s) })),
    [comparedScenarios, region]
  );

  const chartData = useMemo(
    () => ({
      labels: YEAR_LABELS,
      datasets: [
        {
          label: "자연 감소 (Base)",
          data: currentTrend.baseTrend,
          borderColor: "rgba(239, 68, 68, 0.6)",
          borderDash: [5, 5],
          fill: false,
          tension: 0.2,
          pointRadius: 0
        },
        {
          label: "현재 설정",
          data: currentTrend.simTrend,
          borderColor: "rgba(59, 130, 246, 1)",
          fill: false,
          tension: 0.2,
          borderWidth: 2.5
        },
        ...scenarioTrends.map(({ scenario, trend }, i) => ({
          label: scenario.name,
          data: trend.simTrend,
          borderColor: SCENARIO_COLORS[i % SCENARIO_COLORS.length],
          fill: false,
          tension: 0.2,
          pointRadius: 2
        }))
      ]
    }),
    [currentTrend, scenarioTrends]
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

  const summaryRows = scenarioTrends.map(({ scenario, trend }) => {
    const finalPop = trend.simTrend[9];
    const growth = (((finalPop - region.population) / region.population) * 100).toFixed(1);
    return { scenario, finalPop, growth };
  });
  const currentFinal = currentTrend.simTrend[9];
  const currentGrowth = (((currentFinal - region.population) / region.population) * 100).toFixed(1);

  return (
    <div style={{ marginTop: 24 }}>
      <Card title={`시나리오 저장·비교 — ${region.name}`} icon="fa-code-compare">
        <p style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 14 }}>
          현재 예산·배분·시설 변수 조합을 시나리오로 저장하고, 10개년 인구 추이를 나란히 비교합니다.
          저장된 시나리오는 브라우저에 보존됩니다.
        </p>

        {/* 저장 입력 */}
        <div className="scenario-save-row">
          <input
            className="input-control"
            placeholder={`시나리오 이름 (기본: 시나리오 ${regionScenarios.length + 1})`}
            value={name}
            maxLength={30}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleSave();
            }}
            aria-label="시나리오 이름"
          />
          <button type="button" className="btn btn-primary" onClick={handleSave}>
            <i className="fa-solid fa-floppy-disk"></i> 현재 설정 저장
          </button>
        </div>

        {regionScenarios.length === 0 ? (
          <div className="scenario-empty">
            <i className="fa-solid fa-flask-vial" aria-hidden="true"></i>
            저장된 시나리오가 없습니다. 우측 변수 패널을 조정한 뒤 저장해 보세요.
          </div>
        ) : (
          <div className="scenario-body">
            {/* 비교 차트 */}
            <div style={{ position: "relative", height: 220, width: "100%" }}>
              <Line data={chartData} options={chartOpts} />
            </div>

            {/* 요약 테이블 */}
            <div className="table-container" style={{ marginTop: 14 }}>
              <table>
                <thead>
                  <tr>
                    <th>비교</th>
                    <th>시나리오</th>
                    <th style={{ textAlign: "right" }}>예산</th>
                    <th style={{ textAlign: "right" }}>복지/산업/주거</th>
                    <th style={{ textAlign: "right" }}>10년 후 인구</th>
                    <th style={{ textAlign: "right" }}>증감률</th>
                    <th style={{ textAlign: "right" }}>동작</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="scenario-current-row">
                    <td>
                      <span className="scenario-swatch" style={{ backgroundColor: "rgba(59,130,246,1)" }}></span>
                    </td>
                    <td>
                      <strong>현재 설정</strong>
                    </td>
                    <td style={{ textAlign: "right" }}>{snapshot.budgetTotal.toLocaleString()}억</td>
                    <td style={{ textAlign: "right" }}>
                      {snapshot.welfareWeight}/{snapshot.industryWeight}/{snapshot.housingWeight}
                    </td>
                    <td style={{ textAlign: "right" }}>{currentFinal.toLocaleString()}명</td>
                    <td
                      style={{ textAlign: "right" }}
                      className={parseFloat(currentGrowth) >= 0 ? "trend-up" : "trend-down"}
                    >
                      {parseFloat(currentGrowth) > 0 ? "+" : ""}
                      {currentGrowth}%
                    </td>
                    <td></td>
                  </tr>
                  {regionScenarios.map((s) => {
                    const row = summaryRows.find((r) => r.scenario.id === s.id);
                    const colorIdx = comparedScenarios.findIndex((cs) => cs.id === s.id);
                    return (
                      <tr key={s.id}>
                        <td>
                          <label className="scenario-compare-toggle" title="비교 차트에 표시">
                            <input
                              type="checkbox"
                              checked={compared.has(s.id)}
                              onChange={() => toggleCompare(s.id)}
                              aria-label={`${s.name} 비교 포함`}
                            />
                            <span
                              className="scenario-swatch"
                              style={{
                                backgroundColor:
                                  colorIdx >= 0
                                    ? SCENARIO_COLORS[colorIdx % SCENARIO_COLORS.length]
                                    : "var(--border-color)"
                              }}
                            ></span>
                          </label>
                        </td>
                        <td>{s.name}</td>
                        <td style={{ textAlign: "right" }}>{s.budgetTotal.toLocaleString()}억</td>
                        <td style={{ textAlign: "right" }}>
                          {s.welfareWeight}/{s.industryWeight}/{s.housingWeight}
                        </td>
                        <td style={{ textAlign: "right" }}>
                          {row ? `${row.finalPop.toLocaleString()}명` : "—"}
                        </td>
                        <td
                          style={{ textAlign: "right" }}
                          className={row && parseFloat(row.growth) >= 0 ? "trend-up" : "trend-down"}
                        >
                          {row ? `${parseFloat(row.growth) > 0 ? "+" : ""}${row.growth}%` : "—"}
                        </td>
                        <td style={{ textAlign: "right", whiteSpace: "nowrap" }}>
                          <button
                            type="button"
                            className="btn btn-secondary scenario-mini-btn"
                            onClick={() => onApply(s)}
                            title="이 시나리오 변수를 슬라이더에 적용"
                          >
                            <i className="fa-solid fa-arrow-rotate-left"></i> 적용
                          </button>
                          <button
                            type="button"
                            className="btn btn-secondary scenario-mini-btn scenario-del-btn"
                            onClick={() => handleDelete(s.id)}
                            title="시나리오 삭제"
                          >
                            <i className="fa-solid fa-trash-can"></i>
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}
