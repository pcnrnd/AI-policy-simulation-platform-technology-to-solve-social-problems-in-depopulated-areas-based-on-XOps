import { useState, useCallback, useEffect } from "react";

const STORAGE_KEY = "decline-poc-sidebar-width";
const MIN_WIDTH = 200;
const MAX_WIDTH = 420;
const DEFAULT_WIDTH = 260;
const KEYBOARD_STEP = 16;

function clampWidth(value) {
  return Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, value));
}

function readInitialWidth() {
  if (typeof window === "undefined") return DEFAULT_WIDTH;
  const saved = Number(window.localStorage.getItem(STORAGE_KEY));
  if (Number.isFinite(saved) && saved >= MIN_WIDTH && saved <= MAX_WIDTH) {
    return saved;
  }
  return DEFAULT_WIDTH;
}

/**
 * 사이드바 너비를 마우스 드래그로 조절하는 훅.
 * @returns {{ width: number, resizing: boolean, startResize: (e: PointerEvent) => void,
 *   resizeBy: (delta: number) => void, resetWidth: () => void }}
 */
export function useResizableSidebar() {
  const [width, setWidth] = useState(readInitialWidth);
  const [resizing, setResizing] = useState(false);

  useEffect(() => {
    window.localStorage.setItem(STORAGE_KEY, String(width));
  }, [width]);

  const startResize = useCallback((event) => {
    event.preventDefault();
    setResizing(true);
  }, []);

  const resizeBy = useCallback((delta) => {
    setWidth((prev) => clampWidth(prev + delta));
  }, []);

  const resetWidth = useCallback(() => setWidth(DEFAULT_WIDTH), []);

  // 드래그 중 전역 포인터 이벤트 추적 (사이드바는 화면 좌측이므로 clientX = 너비)
  useEffect(() => {
    if (!resizing) return undefined;

    const onMove = (event) => setWidth(clampWidth(event.clientX));
    const onUp = () => setResizing(false);

    document.addEventListener("pointermove", onMove);
    document.addEventListener("pointerup", onUp);
    const prevCursor = document.body.style.cursor;
    const prevSelect = document.body.style.userSelect;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";

    return () => {
      document.removeEventListener("pointermove", onMove);
      document.removeEventListener("pointerup", onUp);
      document.body.style.cursor = prevCursor;
      document.body.style.userSelect = prevSelect;
    };
  }, [resizing]);

  return { width, resizing, startResize, resizeBy, resetWidth };
}
