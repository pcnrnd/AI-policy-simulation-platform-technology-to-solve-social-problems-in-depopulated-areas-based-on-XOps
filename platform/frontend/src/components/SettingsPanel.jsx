import { useState, useRef, useEffect, useCallback } from "react";
import { useTheme } from "../context/ThemeContext.jsx";
import { useAppState } from "../context/AppStateContext.jsx";

export default function SettingsPanel() {
  const { isDark, toggleTheme } = useTheme();
  const { mockDataVisible, toggleMockDataVisible, driftInjected, driftChecking, injectDrift, navigateToTab } = useAppState();
  const [open, setOpen] = useState(false);
  const containerRef = useRef(null);
  const triggerRef = useRef(null);
  const firstControlRef = useRef(null);
  const driftBusyRef = useRef(false);
  const handleDrift = () => {
    if (!mockDataVisible || driftChecking || driftBusyRef.current) return;
    if (driftInjected) { close(); navigateToTab("tab-mlops-orch"); return; }
    driftBusyRef.current = true;
    Promise.resolve(injectDrift()).finally(() => { driftBusyRef.current = false; });
  };

  const close = useCallback(() => setOpen(false), []);

  // 바깥 클릭 / ESC 로 닫기
  useEffect(() => {
    if (!open) return undefined;
    const frame = requestAnimationFrame(() => firstControlRef.current?.focus());

    const onPointerDown = (event) => {
      if (containerRef.current && !containerRef.current.contains(event.target)) {
        close();
      }
    };
    const onKeyDown = (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        close();
        triggerRef.current?.focus();
      }
    };

    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      cancelAnimationFrame(frame);
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open, close]);

  return (
    <div className="settings-panel" ref={containerRef}>
      {open && (
        <div id="sidebar-settings" className="settings-popover" role="region" aria-label="설정">
          <div className="settings-popover-title">설정</div>

          <div className="settings-row">
            <div className="settings-row-label">
              <i className={"fa-solid " + (isDark ? "fa-moon" : "fa-sun")} aria-hidden="true"></i>
              <span>{isDark ? "다크 모드" : "라이트 모드"}</span>
            </div>
            <button
              ref={firstControlRef}
              type="button"
              role="switch"
              aria-checked={!isDark}
              aria-label="다크/라이트 모드 전환"
              className={"theme-switch" + (isDark ? "" : " is-light")}
              onClick={toggleTheme}
            >
              <span className="theme-switch-track">
                <i className="fa-solid fa-moon theme-switch-icon icon-moon" aria-hidden="true"></i>
                <i className="fa-solid fa-sun theme-switch-icon icon-sun" aria-hidden="true"></i>
                <span className="theme-switch-thumb"></span>
              </span>
            </button>
          </div>

          <div className="settings-row">
            <div className="settings-row-label">
              <i className="fa-solid fa-flask-vial" aria-hidden="true"></i>
              <span id="mock-data-switch-label">데모 데이터 표시</span>
            </div>
            <button
              type="button"
              role="switch"
              aria-checked={mockDataVisible}
              aria-labelledby="mock-data-switch-label"
              aria-describedby="mock-data-switch-desc"
              className={"theme-switch mock-switch" + (mockDataVisible ? " is-on" : "")}
              onClick={toggleMockDataVisible}
            >
              <span className="theme-switch-track">
                <span className="mock-switch-state">{mockDataVisible ? "ON" : "OFF"}</span>
                <span className="theme-switch-thumb"></span>
              </span>
            </button>
          </div>
          <p id="mock-data-switch-desc" className="settings-row-desc">
            켜면 예시 데이터로 모든 화면을 채웁니다. 끄면 실데이터만 표시합니다.
          </p>
          {mockDataVisible && <div className="settings-row">
            <button type="button" className="btn btn-secondary" onClick={handleDrift}
              aria-disabled={driftChecking} aria-busy={driftChecking}
              title={driftInjected ? "등록한 파이프라인을 확인하고 수동 실행합니다." : "인구이동 예측의 데모 예시 분포로 PSI/KL 판정과 경보를 확인합니다. 선택 모델의 실측 평가와는 별개입니다."}>
              <i className={`fa-solid ${driftChecking ? "fa-hourglass-half" : driftInjected ? "fa-diagram-project" : "fa-vial-circle-check"}`} aria-hidden="true"></i>{" "}
              {driftChecking ? "데모 분포 판정 중…" : driftInjected ? "등록 파이프라인 확인" : "이상 시나리오 재현"}
            </button>
          </div>}
        </div>
      )}

      <button
        ref={triggerRef}
        type="button"
        className={"settings-gear-btn" + (open ? " active" : "")}
        aria-expanded={open}
        aria-controls="sidebar-settings"
        aria-label={open ? "설정 닫기" : "설정 열기"}
        onClick={() => setOpen((prev) => !prev)}
      >
        <i className="fa-solid fa-gear" aria-hidden="true"></i>
        <span>설정</span>
      </button>
    </div>
  );
}
