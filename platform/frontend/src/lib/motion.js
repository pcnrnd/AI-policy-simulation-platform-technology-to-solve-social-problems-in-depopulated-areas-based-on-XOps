/**
 * 운영체제의 동작 줄이기 설정을 확인한다.
 * SSR·테스트처럼 window가 없는 환경에서는 기본 동작을 유지한다.
 */
export function prefersReducedMotion() {
  return typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/** scrollIntoView 등 DOM 스크롤 API에 전달할 동작 값. */
export function getScrollBehavior() {
  return prefersReducedMotion() ? "auto" : "smooth";
}
