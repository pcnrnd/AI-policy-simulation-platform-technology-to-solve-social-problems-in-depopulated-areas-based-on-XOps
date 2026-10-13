// 로그인 게이트(Caddy forward_auth → Authelia)가 막은 응답 판별과 세션 만료 표식.
// JSX·import.meta 와 분리해 node 로 바로 검사한다(sessionGate.test.mjs).
//
// 세션이 끊긴 뒤 /api 요청은 백엔드에 닿지 않고 게이트에서 막힌다.
// - JSON 을 원한다고 알린 요청(Accept·X-Requested-With) → 본문 없는 401
// - 그렇지 않으면 → 로그인 포털(/authelia/)로 302, fetch 가 따라가 HTML 을 받는다
// 백엔드 자체 401 은 JSON 본문(status/error/message)이라 게이트 401 과 구분된다.

export const SESSION_EXPIRED_STORAGE_KEY = "population-session-expired";

const PORTAL_PATH = "/authelia/";

/**
 * 게이트가 막은 응답이면 true. 백엔드 응답(200 JSON, 401 JSON)·nginx 502 HTML 등은 false.
 * @param {{ status: number, contentType?: string | null, redirected?: boolean, url?: string, bodyText?: string }} res
 * @returns {boolean}
 */
export function isGateResponse({ status, contentType, redirected, url }) {
  if (status === 401 && !String(contentType || "").includes("application/json")) return true;
  if (!redirected || !url) return false;
  try {
    return new URL(url).pathname.startsWith(PORTAL_PATH);
  } catch {
    return false;
  }
}

/**
 * 로그인 뒤 지금 화면으로 돌아오는 포털 주소.
 * @param {string} href 현재 화면 주소
 * @returns {string}
 */
export function loginUrl(href) {
  return `/authelia/?rd=${encodeURIComponent(href)}`;
}

/**
 * "세션 만료로 로그인을 거침" 표식을 남긴다. 스토리지를 못 쓰면 false.
 * @param {{ setItem(key: string, value: string): void } | undefined} storage
 * @returns {boolean}
 */
export function markSessionExpired(storage) {
  try {
    storage.setItem(SESSION_EXPIRED_STORAGE_KEY, "1");
    return true;
  } catch {
    return false;
  }
}

/**
 * 표식이 있으면 지우고 true(1회 소비). 없거나 스토리지를 못 쓰면 false.
 * @param {{ getItem(key: string): string | null, removeItem(key: string): void } | undefined} storage
 * @returns {boolean}
 */
export function consumeSessionExpired(storage) {
  try {
    if (storage.getItem(SESSION_EXPIRED_STORAGE_KEY) !== "1") return false;
    storage.removeItem(SESSION_EXPIRED_STORAGE_KEY);
    return true;
  } catch {
    return false;
  }
}
