import { useEffect, useMemo, useState } from "react";
import Card from "./Card.jsx";
import ConfirmDialog from "./ConfirmDialog.jsx";
import TablePager, { usePager } from "./TablePager.jsx";
import { computeTrends, computeScenarioTrend, budgetToFactor, controlBoostOf } from "../lib/simulation.js";
import { scenarioStorageKey } from "../lib/dataMode.js";

const MAX_PER_REGION = 8;

function loadScenarios(allowSeed) {
  try {
    const raw = localStorage.getItem(scenarioStorageKey(allowSeed));
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.filter((s) => s && s.id && s.regionId) : [];
  } catch {
    return [];
  }
}

function persistScenarios(scenarios, allowSeed) {
  try {
    localStorage.setItem(scenarioStorageKey(allowSeed), JSON.stringify(scenarios));
    return true;
  } catch {
    // 저장공간 초과 등 — 비교 기능은 메모리 상태로 계속 동작
    return false;
  }
}

// 슬라이더 변수 조합을 "시나리오"로 저장하고, 저장 목록에서 다시 적용·삭제한다(비교 차트 칸은 두지 않는다).
// 시나리오는 지자체(regionId) 단위로 관리되며 localStorage에 보존된다.
//
// allowSeed=false(데모 표시 OFF): 저장·적용·삭제는 그대로 조작 가능하되, 10년 후 인구·증감률은
// 지자체 인구(시드 전용)에서 계산되므로 빈 값("–")으로 둔다.
export default function ScenarioCompare({ region, snapshot, onApply, addConsoleLog, allowSeed = true }) {
  const [scenarios, setScenarios] = useState(() => loadScenarios(allowSeed));
  const [name, setName] = useState("");
  const [deleteTarget, setDeleteTarget] = useState(null);
  const [saveFeedback, setSaveFeedback] = useState(null);

  const regionScenarios = scenarios.filter((s) => s.regionId === region.id);
  const list = usePager(regionScenarios);

  useEffect(() => {
    setName("");
  }, [region.id, allowSeed]);

  const updateScenarios = (next) => {
    setScenarios(next);
    return persistScenarios(next, allowSeed);
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

  // 현재 설정·저장 시나리오의 10년 후 인구(데모 계산) — 목록 표의 값 칸에만 쓴다.
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

  // 계산 자체는 그대로 두고(로직 보존), 표시 단계에서만 시드 파생 값을 비운다.
  const popCell = (text) => (allowSeed ? text : "–");
  const growthOf = (finalPop) => (((finalPop - region.population) / region.population) * 100).toFixed(1);
  const currentFinal = currentTrend.simTrend[9];
  const currentGrowth = growthOf(currentFinal);

  return (
    <div style={{ marginTop: 24 }}>
      <Card
        title={`시나리오 저장·비교 — ${region.name}`}
        titleId="scenario-comparison-title"
        titleTabIndex={-1}
        icon="fa-code-compare"
        dataSource={allowSeed ? "mock" : "api"}
        data-slot="sim.scenario"
      >
        <p style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 14 }}>
          {allowSeed
            ? "현재 예산·정책 강조·시설 변수 조합을 시나리오로 저장하고, 목록에서 다시 적용해 비교합니다."
            : "현재 관측 입력 조합을 시나리오로 저장하고, 목록에서 다시 적용한 뒤 [정책 추천 도출]로 예측을 비교합니다."}
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
            {/* 저장 목록 — 현재 설정 행 + 저장 시나리오 5개씩 */}
            <div className="table-container">
              <table id="scenario-comparison-table" tabIndex="-1">
                <caption className="sr-only">현재 설정과 저장한 시나리오 목록</caption>
                <thead>
                  <tr>
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
                      <strong>현재 설정</strong>
                    </td>
                    <td className="cell-num">{snapshot.budgetTotal.toLocaleString()}억</td>
                    <td className="cell-num">
                      {snapshot.welfareWeight}/{snapshot.industryWeight}/{snapshot.housingWeight}
                    </td>
                    <td className="cell-num">{popCell(`${currentFinal.toLocaleString()}명`)}</td>
                    <td
                      className={`cell-num ${!allowSeed ? "" : parseFloat(currentGrowth) >= 0 ? "trend-up" : "trend-down"}`}
                    >
                      {popCell(`${parseFloat(currentGrowth) > 0 ? "+" : ""}${currentGrowth}%`)}
                    </td>
                    <td className="cell-actions">–</td>
                  </tr>
                  {list.pageRows.map((s, pageIndex) => {
                    const rowIndex = list.offset + pageIndex;
                    const finalPop = computeScenarioTrend(region, s).simTrend[9];
                    const row = { finalPop, growth: growthOf(finalPop) };
                    return (
                      <tr key={s.id}>
                        <td>{s.name}</td>
                        <td className="cell-num">{s.budgetTotal.toLocaleString()}억</td>
                        <td className="cell-num">
                          {s.welfareWeight}/{s.industryWeight}/{s.housingWeight}
                        </td>
                        <td className="cell-num">{popCell(`${row.finalPop.toLocaleString()}명`)}</td>
                        <td
                          className={`cell-num ${!allowSeed ? "" : parseFloat(row.growth) >= 0 ? "trend-up" : "trend-down"}`}
                        >
                          {popCell(`${parseFloat(row.growth) > 0 ? "+" : ""}${row.growth}%`)}
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
            <TablePager {...list.pager} />
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
