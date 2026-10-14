// xops-service 백엔드 호출 클라이언트 — native fetch 래퍼.
// base가 빈값이면 상대경로(/api/v3/...)로 요청 → dev는 vite proxy(8000), prod는 VITE_API_BASE_URL.

import { isGateResponse, loginUrl, markSessionExpired } from "./sessionGate.js";
import { beginDataRequest } from "./dataMode.js";

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

// 동시에 여러 요청이 게이트에 막혀도 로그인 화면 이동은 한 번만 한다.
let redirectingToLogin = false;

/** 게이트가 막은 응답 → 표식을 남기고 로그인 포털로 보낸 뒤, 호출부 catch 가 정리하도록 throw. */
function failSessionExpired() {
  if (typeof window !== "undefined" && !redirectingToLogin) {
    redirectingToLogin = true;
    markSessionExpired(window.sessionStorage);
    window.location.assign(loginUrl(window.location.href));
  }
  const err = new Error("세션이 만료되어 로그인 화면으로 이동합니다.");
  err.status = 401;
  err.sessionExpired = true;
  throw err;
}

/**
 * 백엔드 요청 공통 경로. 비-2xx면 파싱한 본문을 담아 Error를 throw.
 * 로그인 게이트가 막은 응답(세션 만료)이면 로그인 화면으로 보내고 err.sessionExpired=true로 throw.
 * @param {string} method
 * @param {string} path  예: "/api/v3/dataops/ds_01"
 * @param {{ token?: string, body?: unknown, params?: Record<string, unknown>, timeoutMs?: number }} [opts]
 * @returns {Promise<any>}
 */
export async function apiRequest(method, path, opts = {}) {
  const { token, body, params, timeoutMs = DEFAULT_TIMEOUT_MS } = opts;
  const modeRequest = beginDataRequest();
  // Accept·X-Requested-With: 세션이 끊겼을 때 게이트가 포털 HTML(302) 대신 401을 주게 한다.
  const headers = {
    "Content-Type": "application/json",
    Accept: "application/json",
    "X-Requested-With": "XMLHttpRequest"
  };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  headers["X-Data-Mode"] = modeRequest.mode;

  try {
  const res = await fetch(`${BASE}${path}${toQuery(params)}`, {
    method,
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
    // 타임아웃은 reject로 나간다 — 호출부의 catch가 진행 중 상태를 정리한다.
    signal: AbortSignal.any([AbortSignal.timeout(timeoutMs), modeRequest.signal])
  });

  const contentType = res.headers.get("content-type") || "";
  if (isGateResponse({ status: res.status, contentType, redirected: res.redirected, url: res.url })) {
    failSessionExpired();
  }

  const text = await res.text();
  if (!modeRequest.isCurrent()) throw new Error("데이터 모드가 변경됐습니다.");
  // JSON이 아닌 본문(nginx 오류 HTML 등)은 파싱하지 않는다 — "Unexpected token '<'" 대신 HTTP 상태로 알린다.
  const data = text && contentType.includes("application/json") ? JSON.parse(text) : null;
  if (!res.ok) {
    // 백엔드 에러 본문(status/error/message 또는 buildUnauthorized)을 그대로 노출
    const err = new Error(data?.message || data?.detail || `HTTP ${res.status}`);
    err.status = res.status;
    err.body = data;
    throw err;
  }
  return data;
  } finally { modeRequest.release(); }
}

export const apiGet = (path, opts) => apiRequest("GET", path, opts);
export const apiSend = (method, path, opts) => apiRequest(method, path, opts);
