// 알림(role=alert) 문구 — 서버·브라우저 원문(HTTP 상태·네트워크 오류·영문 예외·5xx 본문)은 보이지 않고 fallback 으로 바꾼다.
// 백엔드·화면 코드가 사용자용으로 쓴 한글 안내(4xx 본문, 화면 코드의 throw)만 그대로 보인다.
// 원문은 호출부가 UI 활동 로그(WARN)에 남긴다.
export function alertText(error, fallback) {
  const message = typeof error?.message === "string" ? error.message : "";
  const userWritten = /[가-힣]/.test(message) && !/HTTP\s*\d{3}|fetch|Traceback|Exception|[A-Za-z]*Error\b/i.test(message);
  return userWritten && !(error?.status >= 500) ? message : fallback;
}
