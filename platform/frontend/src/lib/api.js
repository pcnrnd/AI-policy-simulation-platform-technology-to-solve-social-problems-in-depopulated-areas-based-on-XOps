// xops-service 백엔드 호출 클라이언트 — native fetch 래퍼.
// base가 빈값이면 상대경로(/api/v3/...)로 요청 → dev는 vite proxy(8000), prod는 VITE_API_BASE_URL.

const BASE = import.meta.env.VITE_API_BASE_URL || "";

// 응답이 오지 않는 백엔드에 await가 영구히 매달리지 않게 하는 기본 상한(ms).
// 호출부가 opts.timeoutMs로 늘리거나 줄일 수 있다.
const DEFAULT_TIMEOUT_MS = 30000;

/** 쿼리 파라미터 객체를 ?a=b&c=d 문자열로 (null/undefined 제외). */
function toQuery(params) {
  if (!params) return "";
  const usp = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== null && v !== undefined && v !== "") usp.append(k, String(v));
  });
  const s = usp.toString();
  return s ? `?${s}` : "";
}

/**
 * 백엔드 요청 공통 경로. 비-2xx면 파싱한 본문을 담아 Error를 throw.
 * @param {string} method
 * @param {string} path  예: "/api/v3/dataops/ds_01"
 * @param {{ token?: string, body?: unknown, params?: Record<string, unknown>, timeoutMs?: number }} [opts]
 * @returns {Promise<any>}
 */
export async function apiRequest(method, path, opts = {}) {
  const { token, body, params, timeoutMs = DEFAULT_TIMEOUT_MS } = opts;
  const headers = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await fetch(`${BASE}${path}${toQuery(params)}`, {
    method,
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
    // 타임아웃은 reject로 나간다 — 호출부의 catch가 진행 중 상태를 정리한다.
    signal: AbortSignal.timeout(timeoutMs)
  });

  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (!res.ok) {
    // 백엔드 에러 본문(status/error/message 또는 buildUnauthorized)을 그대로 노출
    const err = new Error(data?.message || data?.detail || `HTTP ${res.status}`);
    err.status = res.status;
    err.body = data;
    throw err;
  }
  return data;
}

export const apiGet = (path, opts) => apiRequest("GET", path, opts);
export const apiSend = (method, path, opts) => apiRequest(method, path, opts);
