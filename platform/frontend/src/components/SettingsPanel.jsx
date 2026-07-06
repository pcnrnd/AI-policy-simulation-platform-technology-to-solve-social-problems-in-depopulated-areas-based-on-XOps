import { useState, useRef, useEffect, useCallback } from "react";
import { useTheme } from "../context/ThemeContext.jsx";

export default function SettingsPanel() {
  const { isDark, toggleTheme, applyStandardMode } = useTheme();
  const [open, setOpen] = useState(false);
  const containerRef = useRef(null);

  const close = useCallback(() => setOpen(false), []);

  // 바깥 클릭 / ESC 로 닫기
  useEffect(() => {
    if (!open) return undefined;

    const onPointerDown = (event) => {
      if (containerRef.current && !containerRef.current.contains(event.target)) {
        close();
      }
    };
    const onKeyDown = (event) => {
      if (event.key === "Escape") close();
    };

    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open, close]);

  return (
    <div className="settings-panel" ref={containerRef}>
      {open && (
        <div className="settings-popover" role="menu" aria-label="설정">
          <div className="settings-popover-title">설정</div>

          <div className="settings-row">
            <div className="settings-row-label">
              <i className={"fa-solid " + (isDark ? "fa-moon" : "fa-sun")} aria-hidden="true"></i>
              <span>{isDark ? "다크 모드" : "라이트 모드"}</span>
            </div>
            <button
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

          <div className="settings-row settings-row-stack">
            <button
              type="button"
              className="btn btn-secondary"
              style={{ width: "100%", justifyContent: "center" }}
              onClick={applyStandardMode}
              disabled={!isDark}
            >
              <i className="fa-solid fa-universal-access" aria-hidden="true"></i>
              전자정부 표준 모드 적용
            </button>
            <p className="settings-hint">
              밝은 고대비 테마로 전환하여 전자정부 UI·UX 가이드라인(KWCAG 명암비·키보드 포커스·모션
              축소)에 맞춥니다.
            </p>
          </div>
        </div>
      )}

      <button
        type="button"
        className={"settings-gear-btn" + (open ? " active" : "")}
        aria-haspopup="true"
        aria-expanded={open}
        aria-label="설정 열기"
        onClick={() => setOpen((prev) => !prev)}
      >
        <i className="fa-solid fa-gear" aria-hidden="true"></i>
        <span>설정</span>
      </button>
    </div>
  );
}
