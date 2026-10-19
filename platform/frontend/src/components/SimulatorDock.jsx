import { useId } from "react";
import { sliderValueText } from "../lib/realSimulation.js";

// 정책 시뮬레이터 STAGE ③ 우측 독 — 분석 대상 지자체 목록 + 시뮬레이션 변수.
// 두 모드가 같은 컨트롤을 쓴다. 실데이터 대응이 없는 예산·가중치는 표시는 유지하고 잠근다(lockText 로 안내).

/**
 * @param {{ regions: object[], currentRegion: object, onSelectRegion: Function, regionInfo: Function,
 *   searchText: string, onClearSearch: Function, hint: string,
 *   fixed: { budgetTotal: number, setBudgetTotal: Function, welfare: number, setWelfare: Function,
 *     industry: number, setIndustry: Function, housing: number, setHousing: Function, lockText: string|null },
 *   controls: object[], controlValues: object, onControlChange: Function, controlsLabel: string,
 *   controlsEmptyText: string|null, controlsLockTitle: string|undefined, onResetControls: Function, onRecommend: Function,
 *   recommendDisabled: boolean, notReadyText: string|null, stale: boolean }} props
 * regionInfo(region) → { chip: { label, style } | null, meta: string }
 */
export default function SimulatorDock({
  regions,
  currentRegion,
  onSelectRegion,
  regionInfo,
  searchText,
  onClearSearch,
  districtControl,
  hint,
  fixed,
  controls,
  controlValues,
  onControlChange,
  controlsLabel,
  controlsEmptyText,
  controlsLockTitle,
  onResetControls,
  onRecommend,
  recommendDisabled,
  notReadyText,
  stale
}) {
  const locked = Boolean(fixed.lockText);
  return (
    <div className="sim-dock">
      <div className="sim-dock-section" data-slot="sim.dock.regions">
        <div className="sim-dock-title">
          <i className="fa-solid fa-list-ul"></i> 분석 대상 지자체
          <span className="sim-dock-count" role="status" aria-live="polite">
            검색 결과 {regions.length}건
          </span>
        </div>
        <div className="sim-region-list">
          {regions.map((r) => {
            const active = r.id === currentRegion.id;
            const info = regionInfo(r);
            return (
              <button
                type="button"
                key={r.id}
                className={"sim-region-item" + (active ? " active" : "")}
                onClick={() => onSelectRegion(r)}
                aria-pressed={active}
              >
                {/* 지자체 이름·선택은 조작 컨트롤이라 모드와 무관하게 남기고, 등급·메타만 모드별 데이터로 채운다. */}
                <div className="sim-region-head">
                  <strong>{r.name}</strong>
                  <span className="grade-chip" style={info.chip?.style}>
                    {info.chip?.label ?? "–"}
                  </span>
                </div>
                <div className="sim-region-meta">{info.meta}</div>
              </button>
            );
          })}
          {regions.length === 0 && (
            <div className="sim-region-empty">
              <p>“{searchText}” 검색 결과가 없습니다.</p>
              <button type="button" className="btn btn-secondary" onClick={onClearSearch}>
                검색 해제
              </button>
            </div>
          )}
        </div>
        {districtControl}
      </div>

      <div className="sim-dock-section">
        <div className="sim-dock-title">
          <i className="fa-solid fa-sliders"></i> {currentRegion.name} 시뮬레이션 변수
        </div>
        <p className="sim-dock-hint">{hint}</p>

        <div data-slot="sim.dock.vars">
          <div className="sim-var-group-label">
            <i className="fa-solid fa-lock" aria-hidden="true"></i> 제약요소 · 총 예산
          </div>
          {locked && <p className="sim-dock-hint">{fixed.lockText}</p>}
          <Slider label="총 예산" value={fixed.budgetTotal} onChange={fixed.setBudgetTotal} min={100} max={2000} step={50} unit="억" locked={locked} lockTitle={fixed.lockText} />

          <div className="sim-var-group-label">정책 강조 가중치</div>
          <Slider label="복지 예산 (Welfare)" value={fixed.welfare} onChange={fixed.setWelfare} locked={locked} lockTitle={fixed.lockText} />
          <Slider label="산업 일자리 (Industry)" value={fixed.industry} onChange={fixed.setIndustry} locked={locked} lockTitle={fixed.lockText} />
          <Slider label="주거·주택 (Housing)" value={fixed.housing} onChange={fixed.setHousing} locked={locked} lockTitle={fixed.lockText} />
        </div>

        <div data-slot="sim.dock.controls">
          <div className="sim-var-group-label">
            <i className="fa-solid fa-arrows-up-down-left-right" aria-hidden="true"></i> {controlsLabel}
          </div>
          {controls.length === 0 && controlsEmptyText && <p className="sim-dock-hint">{controlsEmptyText}</p>}
          {controls.map((c) => (
            <Slider
              key={c.key}
              label={c.label}
              value={controlValues[c.key] ?? c.default}
              onChange={(v) => onControlChange(c.key, v)}
              min={c.min}
              max={c.max}
              step={c.step ?? 1}
              exactValue={c.exactValue}
              unit={c.unit}
              locked={c.locked}
            />
          ))}
        </div>

        <div data-slot="sim.dock.actions">
          <button
            type="button"
            className="btn btn-secondary"
            style={{ width: "100%", marginTop: 8 }}
            onClick={onResetControls}
            aria-disabled={controls.length === 0 || undefined}
            title={controls.length === 0 ? controlsLockTitle : undefined}
          >
            <i className="fa-solid fa-rotate-left" aria-hidden="true"></i> 조절변수 초기화
          </button>
          {/* 잠금은 aria-disabled + 사유 title 로 통일한다(포커스·낭독 유지, 클릭은 무시). */}
          <button
            className="btn btn-primary"
            style={{ width: "100%", marginTop: 8 }}
            onClick={() => { if (!recommendDisabled) onRecommend(); }}
            aria-disabled={recommendDisabled || undefined}
            title={recommendDisabled ? notReadyText ?? undefined : undefined}
          >
            <i className="fa-solid fa-lightbulb"></i> 정책 추천 도출
          </button>
          {notReadyText && (
            <p style={{ fontSize: 11, color: "var(--text-secondary)", marginTop: 8, textAlign: "center" }}>
              <i className="fa-solid fa-circle-info"></i> {notReadyText}
            </p>
          )}
          {stale && (
            <p style={{ fontSize: 11, color: "var(--accent-orange)", marginTop: 8, textAlign: "center" }}>
              <i className="fa-solid fa-triangle-exclamation"></i> 변수가 변경되었습니다. 추천을 다시 도출하세요.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

// locked: 값은 보이되 조정하지 않는다(aria-disabled — 포커스·낭독은 유지, 변경은 무시).
function Slider({ label, value, onChange, min = 0, max = 100, step = 1, unit = "%", locked = false, lockTitle, exactValue = false }) {
  const inputId = useId();
  const rangeDescriptionId = `${inputId}-range-description`;
  const shown = sliderValueText(value, unit, step);
  // 조절 범위 — 한 칸이 1 미만이면 원문(소수 칸 그대로), 아니면 값과 같은 표기.
  const rangeText = (n) => step < 1 ? n.toLocaleString() : sliderValueText(n, unit, step);

  return (
    <div className="slider-container">
      <div className="slider-header">
        <label htmlFor={inputId}>{label}</label>
        <output className="slider-val" htmlFor={inputId} aria-hidden="true">
          {shown}
          {unit}
        </output>
      </div>
      <span id={rangeDescriptionId} className="sr-only">
        조절 범위: {rangeText(min)}
        {unit}부터 {rangeText(max)}
        {unit}까지, {exactValue ? "키보드 조절 " : ""}{rangeText(step)}
        {unit} 단위
      </span>
      <input
        id={inputId}
        type="range"
        min={min}
        max={max}
        step={exactValue ? "any" : step}
        value={Number.isFinite(value) ? value : min}
        aria-valuetext={`${shown}${unit}`}
        aria-describedby={rangeDescriptionId}
        aria-disabled={locked || undefined}
        title={locked ? lockTitle : undefined}
        onKeyDown={(event) => {
          if (!exactValue) return;
          const direction = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1 }[event.key];
          if (!direction && event.key !== "Home" && event.key !== "End") return;
          event.preventDefault();
          if (locked) return;
          const next = event.key === "Home" ? min : event.key === "End" ? max : value + direction * step;
          onChange(Math.max(min, Math.min(max, next)));
        }}
        // 실데이터 입력은 소수 단위가 있어 Number 로 읽는다(정수 슬라이더도 그대로 정수).
        onChange={(e) => { if (!locked) onChange(Number(e.target.value)); }}
      />
    </div>
  );
}
