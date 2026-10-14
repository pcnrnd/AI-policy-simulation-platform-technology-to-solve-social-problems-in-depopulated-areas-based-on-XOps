// No application cookies/storage are read or written by the dummy module.
const serverMode = document.body.dataset.mode;
const queryMode = new URLSearchParams(location.search).get("mode");
const mode = ["mock", "real"].includes(serverMode) ? serverMode : ["mock", "real"].includes(queryMode) ? queryMode : "real";
document.getElementById("mode").textContent = `요청 모드: ${mode === "mock" ? "데모(mock)" : "실데이터(real)"}`;
document.getElementById("asset").textContent = "JS 자산 로드 완료 (모듈 base path)";
const deep = document.querySelector('a[href="deep/refresh-check/"]');
deep.href += `?mode=${encodeURIComponent(mode)}&probe=deep-link`;
const apiPath = document.body.dataset.api;
fetch(`${apiPath}probe?mode=${encodeURIComponent(mode)}&probe=frontend&module=${document.body.dataset.module}`, {
  headers: { "X-Data-Mode": mode, Accept: "application/json", "X-Requested-With": "XMLHttpRequest" }, credentials: "same-origin", cache: "no-store"
}).then(async (response) => {
  if (response.status === 401) {
    location.assign(`/authelia/?rd=${encodeURIComponent(location.href)}`);
    return;
  }
  document.getElementById("echo").textContent = `HTTP ${response.status}\n${await response.text()}`;
}).catch(() => { document.getElementById("echo").textContent = "모듈 API 요청을 확인하지 못했습니다."; });
