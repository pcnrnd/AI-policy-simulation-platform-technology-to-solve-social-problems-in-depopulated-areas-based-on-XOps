import { useEffect, useMemo, useState } from "react";
import { Line } from "react-chartjs-2";
import Card from "./Card.jsx";
import ConfirmDialog from "./ConfirmDialog.jsx";
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
    return true;
  } catch {
    // 저장공간 초과 등 — 비교 기능은 메모리 상태로 계속 동작
    return false;
  }
}

// 슬라이더 변수 조합을 "시나리오"로 저장하고, 10개년 인구 추이를 나란히 비교한다.
// 시나리오는 지자체(regionId) 단위로 관리되며 localStorage에 보존된다.
export default function ScenarioCompare({ region, snapshot, onApply, addConsoleLog }) {
  const ct = useChartTheme();
  const [scenarios, setScenarios] = useState(loadScenarios);
  const [name, setName] = useState("");
  const [compared, setCompared] = useState(() => new Set());
  const [deleteTarget, setDeleteTarget] = useState(null);
  const [saveFeedback, setSaveFeedback] = useState(null);

  const regionScenarios = scenarios.filter((s) => s.regionId === region.id);

  // 지역 변경 시 비교 선택을 해당 지역 시나리오 전체로 초기화
  useEffect(() => {
    setCompared(new Set(loadScenarios().filter((s) => s.regionId === region.id).map((s) => s.id)));
    setName("");
  }, [region.id]);

  const updateScenarios = (next) => {
    setScenarios(next);
    return persistScenarios(next);
  };

  const handleSave = (event) => {
    event?.preventDefault();
    if (regionScenarios.length >= MAX_PER_REGION) {
      const message = `시나리오는 지자체당 최대 ${MAX_PER_REGION}개까지 저장됩니다. 기존 시나리오를 삭제한 뒤 다시 시도하세요.`;
      setSaveFeedback({ tone: "error", message });
      addConsoleLog(`WARN: ${message}`);
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
    const persisted = updateScenarios([...scenarios, scenario]);
    setCompared((prev) => new Set(prev).add(scenario.id));
    setName("");
    setSaveFeedback({
      tone: persisted ? "success" : "error",
      message: persisted
        ? `‘${label}’ 시나리오를 저장했습니다.`
        : `‘${label}’ 시나리오는 현재 세션에만 유지됩니다. 브라우저 저장공간을 확인하세요.`
    });
    addConsoleLog(
      `INFO: 시나리오 '${label}' 저장 — 예산 ${scenario.budgetTotal.toLocaleString()}억, ` +
        `복지 ${scenario.welfareWeight}/산업 ${scenario.industryWeight}/주거 ${scenario.housingWeight}.`
    );
  };

  const handleDelete = (id) => {
    const persisted = updateScenarios(scenarios.filter((s) => s.id !== id));
    setCompared((prev) => {
      const next = new Set(prev);
      next.delete(id);
      return next;
    });
    const deleted = scenarios.find((scenario) => scenario.id === id);
    if (deleted) {
      addConsoleLog(`WARN: 시나리오 '${deleted.name}' 삭제.`);
      setSaveFeedback({
        tone: persisted ? "success" : "error",
        message: persisted
          ? `‘${deleted.name}’ 시나리오를 삭제했습니다.`
          : `‘${deleted.name}’은 현재 화면에서만 삭제됐습니다. 브라우저 저장소를 갱신하지 못해 새로고침하면 다시 나타날 수 있습니다.`
      });
    }
    setDeleteTarget(null);
  };

  const requestDelete = (scenario, rowIndex) => {
    const next = regionScenarios[rowIndex + 1] ?? regionScenarios[rowIndex - 1];
    setDeleteTarget({
      scenario,
      nextFocusSelector: next ? `[data-scenario-delete="${next.id}"]` : "#scenario-comparison-title",
      fallbackFocusSelector: "#scenario-comparison-title"
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
  const baseFinal = currentTrend.baseTrend[9];
  const currentVsBase = currentFinal - baseFinal;
  const currentGrowth = (((currentFinal - region.population) / region.population) * 100).toFixed(1);

  return (
    <div style={{ marginTop: 24 }}>
      <Card
        title={`시나리오 저장·비교 — ${region.name}`}
        titleId="scenario-comparison-title"
        titleTabIndex={-1}
        icon="fa-code-compare"
      >
        <p style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 14 }}>
          현재 예산·배분·시설 변수 조합을 시나리오로 저장하고, 10개년 인구 추이를 나란히 비교합니다.
          저장된 시나리오는 브라우저 저장소가 허용된 환경에서 보존됩니다.
        </p>

        {/* 저장 입력 */}
        <form className="scenario-save-row" onSubmit={handleSave}>
          <label className="scenario-name-field">
            <span>시나리오 이름 (선택)</span>
            <input
              className="input-control"
              placeholder={`비워두면 시나리오 ${regionScenarios.length + 1}`}
              value={name}
              maxLength={30}
              onChange={(e) => setName(e.target.value)}
            />
          </label>
          <button type="submit" className="btn btn-primary" disabled={regionScenarios.length >= MAX_PER_REGION}>
            <i className="fa-solid fa-floppy-disk" aria-hidden="true"></i> 현재 설정 저장
          </button>
        </form>
        {saveFeedback && (
          <p
            className={`async-feedback is-${saveFeedback.tone}`}
            role={saveFeedback.tone === "error" ? "alert" : "status"}
            aria-live="polite"
          >
            {saveFeedback.message}
          </p>
        )}
        {regionScenarios.length >= MAX_PER_REGION && (
          <p className="field-error">최대 {MAX_PER_REGION}개를 저장했습니다. 새로 저장하려면 기존 시나리오를 삭제하세요.</p>
        )}

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
            <p className="chart-summary">
              자연감소 기준은 10년 후 {baseFinal.toLocaleString()}명이며, 현재 설정은 {currentFinal.toLocaleString()}명
              ({parseFloat(currentGrowth) > 0 ? "+" : ""}{currentGrowth}%)으로 기준 대비 {currentVsBase >= 0 ? "+" : ""}{currentVsBase.toLocaleString()}명입니다.
              {scenarioTrends.length > 0
                ? ` 비교 중인 저장 시나리오 ${scenarioTrends.length}개의 연도별 추이는 차트와 아래 요약표에서 확인할 수 있습니다.`
                : " 비교할 저장 시나리오를 선택하면 추이가 함께 표시됩니다."}
            </p>

            {/* 요약 테이블 */}
            <div className="table-container" style={{ marginTop: 14 }}>
              <table id="scenario-comparison-table" tabIndex="-1">
                <caption className="sr-only">현재 설정과 저장한 인구 예측 시나리오 비교</caption>
                <thead>
                  <tr>
                    <th scope="col">비교</th>
                    <th scope="col">시나리오</th>
                    <th scope="col" className="cell-num">예산</th>
                    <th scope="col" className="cell-num">복지/산업/주거</th>
                    <th scope="col" className="cell-num">10년 후 인구</th>
                    <th scope="col" className="cell-num">증감률</th>
                    <th scope="col" className="cell-actions">동작</th>
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
                    <td className="cell-num">{snapshot.budgetTotal.toLocaleString()}억</td>
                    <td className="cell-num">
                      {snapshot.welfareWeight}/{snapshot.industryWeight}/{snapshot.housingWeight}
                    </td>
                    <td className="cell-num">{currentFinal.toLocaleString()}명</td>
                    <td
                      className={`cell-num ${parseFloat(currentGrowth) >= 0 ? "trend-up" : "trend-down"}`}
                    >
                      {parseFloat(currentGrowth) > 0 ? "+" : ""}
                      {currentGrowth}%
                    </td>
                    <td className="cell-actions">–</td>
                  </tr>
                  {regionScenarios.map((s, rowIndex) => {
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
                        <td className="cell-num">{s.budgetTotal.toLocaleString()}억</td>
                        <td className="cell-num">
                          {s.welfareWeight}/{s.industryWeight}/{s.housingWeight}
                        </td>
                        <td className="cell-num">
                          {row ? `${row.finalPop.toLocaleString()}명` : "–"}
                        </td>
                        <td
                          className={`cell-num ${row && parseFloat(row.growth) >= 0 ? "trend-up" : "trend-down"}`}
                        >
                          {row ? `${parseFloat(row.growth) > 0 ? "+" : ""}${row.growth}%` : "–"}
                        </td>
                        <td className="cell-actions">
                          <button
                            type="button"
                            className="btn btn-secondary scenario-mini-btn"
                            onClick={() => onApply(s)}
                            title="이 시나리오 변수를 슬라이더에 적용"
                            aria-label={`${s.name} 시나리오 적용`}
                          >
                            <i className="fa-solid fa-arrow-rotate-left"></i> 적용
                          </button>
                          <button
                            type="button"
                            className="btn btn-secondary scenario-mini-btn scenario-del-btn"
                            onClick={() => requestDelete(s, rowIndex)}
                            data-scenario-delete={s.id}
                            title="시나리오 삭제"
                            aria-label={`${s.name} 시나리오 삭제`}
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
      <ConfirmDialog
        open={Boolean(deleteTarget)}
        title="저장한 시나리오를 삭제할까요?"
        description={deleteTarget ? `${region.name}의 '${deleteTarget.scenario.name}' 시나리오를 브라우저 저장 목록에서 삭제합니다. 이 작업은 되돌릴 수 없습니다.` : ""}
        confirmLabel="시나리오 삭제"
        nextFocusSelector={deleteTarget?.nextFocusSelector}
        fallbackFocusSelector={deleteTarget?.fallbackFocusSelector}
        onCancel={() => setDeleteTarget(null)}
        onConfirm={() => handleDelete(deleteTarget.scenario.id)}
      />
    </div>
  );
}
