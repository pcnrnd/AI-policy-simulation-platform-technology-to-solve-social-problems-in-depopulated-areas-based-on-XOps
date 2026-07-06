// 시각화/API 응답속도 계측 유틸 — 공인인증 정량목표(처리 응답속도 ≤ 2초) 대응.
import { useEffect, useRef, useState } from "react";

// 공인인증 기준: 데이터 시각화 처리 응답속도 2 sec.
export const PERF_BUDGET_MS = 2000;

/**
 * 의존성이 바뀔 때 렌더→첫 페인트까지 경과 시간을 측정한다.
 * 시각화 컴포넌트의 "처리 응답속도" 프록시로 사용.
 * @param {any[]} deps
 * @returns {number|null} 경과 ms (페인트 후 갱신)
 */
export function useRenderTiming(deps) {
  const [ms, setMs] = useState(null);
  const startRef = useRef(0);
  startRef.current = performance.now();

  useEffect(() => {
    const raf = requestAnimationFrame(() => {
      setMs(performance.now() - startRef.current);
    });
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return ms;
}

/**
 * 비동기 작업의 실제 경과 시간을 측정한다.
 * @param {() => Promise<T> | T} fn
 * @returns {Promise<{ result: T, ms: number }>}
 */
export async function measureAsync(fn) {
  const start = performance.now();
  const result = await fn();
  return { result, ms: performance.now() - start };
}
