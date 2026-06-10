import { useMemo } from "react";
import { useTheme } from "../context/ThemeContext.jsx";

/**
 * 현재 테마(다크/라이트)에 맞는 Chart.js 색상 팔레트를 반환한다.
 * 라이트 모드에서 흰색 계열 축/범례 라벨이 흰 배경에 묻히는 문제를 방지.
 * @returns {{ tick: string, tickStrong: string, grid: string, angleLines: string, legend: string }}
 */
export function useChartTheme() {
  const { isDark } = useTheme();

  return useMemo(
    () => ({
      tick: isDark ? "#9ca3af" : "#475569",
      tickStrong: isDark ? "#f3f4f6" : "#1e293b",
      grid: isDark ? "rgba(255, 255, 255, 0.05)" : "rgba(15, 23, 42, 0.08)",
      angleLines: isDark ? "rgba(255, 255, 255, 0.08)" : "rgba(15, 23, 42, 0.12)",
      legend: isDark ? "#f3f4f6" : "#1e293b"
    }),
    [isDark]
  );
}
