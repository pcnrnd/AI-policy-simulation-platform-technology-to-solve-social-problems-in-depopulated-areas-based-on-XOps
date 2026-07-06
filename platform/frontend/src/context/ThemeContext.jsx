import { createContext, useContext, useState, useCallback, useEffect } from "react";

const ThemeContext = createContext(null);

const STORAGE_KEY = "decline-poc-theme";
const DARK = "dark";
const LIGHT = "light";

function readInitialTheme() {
  if (typeof window === "undefined") return DARK;
  const saved = window.localStorage.getItem(STORAGE_KEY);
  if (saved === DARK || saved === LIGHT) return saved;
  // 저장된 값이 없으면 OS 환경설정을 존중 (기본은 다크)
  const prefersLight = window.matchMedia?.("(prefers-color-scheme: light)").matches;
  return prefersLight ? LIGHT : DARK;
}

export function ThemeProvider({ children }) {
  const [theme, setTheme] = useState(readInitialTheme);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    window.localStorage.setItem(STORAGE_KEY, theme);
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme((prev) => (prev === DARK ? LIGHT : DARK));
  }, []);

  // 전자정부 표준(밝은 고대비) 모드 직접 적용 — 라이트 테마로 고정.
  const applyStandardMode = useCallback(() => setTheme(LIGHT), []);

  const value = {
    theme,
    isDark: theme === DARK,
    toggleTheme,
    applyStandardMode
  };

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used within ThemeProvider");
  return ctx;
}
