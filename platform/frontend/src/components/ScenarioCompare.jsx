import { useEffect, useMemo, useState } from "react";
import Card from "./Card.jsx";
import ConfirmDialog from "./ConfirmDialog.jsx";
import TablePager, { usePager } from "./TablePager.jsx";
import { computeTrends, computeScenarioTrend, budgetToFactor, controlBoostOf } from "../lib/simulation.js";
import { scenarioStorageKey } from "../lib/dataMode.js";
import { formatUnitValue } from "../lib/realSimulation.js";

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
// allowSeed=false(데모 표시 OFF): 같은 열 자리에 10년 후 인구·증감률 대신 시뮬레이션 모델의 다음 달 예측과
// 기준 대비 변화율을 둔다(forecast: { ready, key, unit, current, predict(controlValues) }). 열 수·순서는 같다.
export default function ScenarioCompare({ region, snapshot, onApply, addConsoleLog, allowSeed = true, forecast = null }) {
  const [scenarios, setScenarios] = useState(() => loadScenarios(allowSeed));
  const [name, setName] = useState("");
  const [deleteTarget, setDeleteTarget] = useState(null);
  const [saveFeedback, setSaveFeedback] = useState(null);
  const [rowForecasts, setRowForecasts] = useState({}); // 시나리오 id → 추론 결과 | "failed"

  const regionScenarios = scenarios.filter((s) => s.regionId === region.id);
  const list = usePager(regionScenarios, undefined, region.id);
  const pageIds = list.pageRows.map((s) => s.id).join(",");
  const forecastReady = Boolean(forecast?.ready);

  // 실데이터: 보이는 쪽의 저장 시나리오마다 관측 입력으로 다음 달 예측을 계산한다(모델·쪽이 바뀌면 다시).
  useEffect(() => {
    setRowForecasts({});
    if (allowSeed || !forecastReady) return undefined;
    let alive = true;
    list.pageRows.forEach((s) => {
      forecast.predict(s.controlValues)
        .then((result) => alive && setRowForecasts((prev) => ({ ...prev, [s.id]: result ?? "failed" })))
        .catch((error) => {
          if (!alive) return;
          setRowForecasts((prev) => ({ ...prev, [s.id]: "failed" }));
          addConsoleLog(`WARN: 시나리오 '${s.name}' 예측 계산 실패 — ${error?.message ?? "알 수 없는 오류"}`);
        });
    });
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- predict 는 렌더마다 새 함수, 모델·입력 조건은 key 가 대표한다
  }, [allowSeed, forecastReady, forecast?.key, pageIds]);

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

  const growthOf = (finalPop) => (((finalPop - region.population) / region.population) * 100).toFixed(1);
  const trendClass = (growth) => (parseFloat(growth) >= 0 ? "trend-up" : "trend-down");
  // 한 행의 값 두 칸 [예측, 변화율, 색] — 데모는 10년 후 인구·증감률, 실데이터는 다음 달 예측·기준 대비 변화율.
  const demoCells = (finalPop) => {
    const growth = growthOf(finalPop);
    return [`${finalPop.toLocaleString()}명`, `${parseFloat(growth) > 0 ? "+" : ""}${growth}%`, trendClass(growth)];
  };
  const realCells = (result) => {
    if (!forecastReady) return ["모델 준비 후", "모델 준비 후", ""];
    if (result === undefined || result === null) return ["계산 중", "계산 중", ""];
    if (result === "failed") return ["계산 실패", "계산 실패", ""];
    const unit = forecast.unit;
    const rate = result.baseline_prediction ? ((result.delta / Math.abs(result.baseline_prediction)) * 100).toFixed(1) : "0.0";
    return [`${formatUnitValue(result.prediction, unit)}${unit}`, `${result.delta > 0 ? "+" : ""}${rate}%`, trendClass(result.delta)];
  };
  const currentCells = allowSeed ? demoCells(currentTrend.simTrend[9]) : realCells(forecast?.current);

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
                    <th scope="col" className="cell-num">{allowSeed ? "10년 후 인구" : "다음 달 예측"}</th>
                    <th scope="col" className="cell-num">{allowSeed ? "증감률" : "기준 대비 변화율"}</th>
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
                    <td className="cell-num">{currentCells[0]}</td>
                    <td className={`cell-num ${currentCells[2]}`}>{currentCells[1]}</td>
                    <td className="cell-actions">–</td>
                  </tr>
                  {list.pageRows.map((s, pageIndex) => {
                    const rowIndex = list.offset + pageIndex;
                    const cells = allowSeed ? demoCells(computeScenarioTrend(region, s).simTrend[9]) : realCells(rowForecasts[s.id]);
                    return (
                      <tr key={s.id}>
                        <td>{s.name}</td>
                        <td className="cell-num">{s.budgetTotal.toLocaleString()}억</td>
                        <td className="cell-num">
                          {s.welfareWeight}/{s.industryWeight}/{s.housingWeight}
                        </td>
                        <td className="cell-num">{cells[0]}</td>
                        <td className={`cell-num ${cells[2]}`}>{cells[1]}</td>
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
