import { useState, useRef, useEffect, useCallback } from "react";
import { useTheme } from "../context/ThemeContext.jsx";
import { useAppState } from "../context/AppStateContext.jsx";

export default function SettingsPanel() {
  const { isDark, toggleTheme } = useTheme();
  const { mockDataVisible, toggleMockDataVisible } = useAppState();
  const [open, setOpen] = useState(false);
  const containerRef = useRef(null);
  const triggerRef = useRef(null);
  const firstControlRef = useRef(null);

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
              <span id="mock-data-switch-label">목업 데이터 표시</span>
            </div>
            <button
              type="button"
              role="switch"
              aria-checked={mockDataVisible}
              aria-labelledby="mock-data-switch-label"
              className={"theme-switch mock-switch" + (mockDataVisible ? " is-on" : "")}
              onClick={toggleMockDataVisible}
            >
              <span className="theme-switch-track">
                <span className="mock-switch-state">{mockDataVisible ? "ON" : "OFF"}</span>
                <span className="theme-switch-thumb"></span>
              </span>
            </button>
          </div>

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
