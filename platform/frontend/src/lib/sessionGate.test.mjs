// 로그인 게이트 응답 판별·세션 만료 표식 회귀 검사 (vitest 미도입 → node 직접 실행).
//
// 세션이 끊기면 /api 요청이 게이트에서 401(본문 없음) 또는 포털 HTML 로 막힌다.
// 이를 백엔드 응답(200 JSON, 401 JSON)·nginx 502 HTML 과 구분해야 화면이 로그인으로 보낼지 정한다.
import assert from "node:assert/strict";
import {
  SESSION_EXPIRED_STORAGE_KEY,
  consumeSessionExpired,
  isGateResponse,
  loginUrl,
  markSessionExpired
} from "./sessionGate.js";

let passed = 0;
const check = (name, fn) => {
  fn();
  passed += 1;
  console.log(`  ok  ${name}`);
};

/** 메모리 스토리지(sessionStorage 대역). */
const memoryStorage = () => {
  const map = new Map();
  return {
    getItem: (key) => (map.has(key) ? map.get(key) : null),
    setItem: (key, value) => map.set(key, String(value)),
    removeItem: (key) => map.delete(key)
  };
};

/** 접근 자체가 던지는 스토리지(프라이빗 모드·권한 거부). */
const throwingStorage = {
  getItem() {
    throw new DOMException("denied", "SecurityError");
  },
  setItem() {
    throw new DOMException("denied", "SecurityError");
  },
  removeItem() {
    throw new DOMException("denied", "SecurityError");
  }
};

check("게이트 401(본문 없음·비JSON) → 게이트 응답", () => {
  assert.equal(isGateResponse({ status: 401, contentType: "", redirected: false, url: "https://h/api/v3/x" }), true);
  assert.equal(isGateResponse({ status: 401, contentType: null }), true);
  assert.equal(isGateResponse({ status: 401, contentType: "text/html; charset=utf-8" }), true);
});

check("포털로 리다이렉트된 HTML → 게이트 응답", () => {
  assert.equal(
    isGateResponse({
      status: 200,
      contentType: "text/html; charset=utf-8",
      redirected: true,
      url: "https://h/authelia/?rd=https%3A%2F%2Fh%2Fapi%2Fv3%2Fx",
      bodyText: "<!-- SPDX-..."
    }),
    true
  );
});

check("200 JSON → 게이트 응답 아님", () => {
  assert.equal(isGateResponse({ status: 200, contentType: "application/json", redirected: false, url: "https://h/api/v3/x" }), false);
});

check("백엔드 401 JSON → 게이트 응답 아님", () => {
  assert.equal(isGateResponse({ status: 401, contentType: "application/json", redirected: false, url: "https://h/api/v3/x" }), false);
});

check("502 HTML(리다이렉트 아님)·백엔드 정상 리다이렉트 → 게이트 응답 아님", () => {
  assert.equal(isGateResponse({ status: 502, contentType: "text/html", redirected: false, url: "https://h/api/v3/x" }), false);
  assert.equal(isGateResponse({ status: 200, contentType: "application/json", redirected: true, url: "https://h/api/v3/x/" }), false);
  assert.equal(isGateResponse({ status: 200, contentType: "text/html", redirected: true, url: "not a url" }), false);
});

check("loginUrl 은 현재 주소를 rd 로 인코딩한다", () => {
  assert.equal(
    loginUrl("https://h/realdata?tab=a&b=1#x"),
    "/authelia/?rd=https%3A%2F%2Fh%2Frealdata%3Ftab%3Da%26b%3D1%23x"
  );
});

check("표식은 저장 뒤 1회만 소비된다", () => {
  const s = memoryStorage();
  assert.equal(consumeSessionExpired(s), false);
  assert.equal(markSessionExpired(s), true);
  assert.equal(s.getItem(SESSION_EXPIRED_STORAGE_KEY), "1");
  assert.equal(consumeSessionExpired(s), true);
  assert.equal(consumeSessionExpired(s), false);
});

check("스토리지가 없거나 던지면 조용히 false", () => {
  assert.equal(markSessionExpired(undefined), false);
  assert.equal(consumeSessionExpired(undefined), false);
  assert.equal(markSessionExpired(throwingStorage), false);
  assert.equal(consumeSessionExpired(throwingStorage), false);
});

console.log(`\n${passed} checks passed`);
