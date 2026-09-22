// 실데이터 연계(R3·R4) 백엔드 호출 헬퍼 — DataOps 발급 흐름(POST /dataops/token/{source_id})을
// 그대로 재사용해 단일 토큰(scope: data:read data:write)으로 조회·쓰기를 모두 게이팅한다.
// JSX는 없지만 컴포넌트 소유 경계(components/realdata/*.jsx)를 지키기 위해 .jsx로 둔다.
import { apiGet, apiSend } from "../../lib/api.js";

const BASE = "/api/v3/realdata";

export async function issueRealdataToken() {
  // 소스 무관 발급 경로. `/token/{source_id}` 는 카탈로그 존재를 검사하므로 카탈로그에 없는
  // 주체("realdata-namwon")로는 404가 났다 — dts 로그에서 실제로 404로 확인.
  const res = await apiSend("POST", "/api/v3/dataops/token", {});
  return res.access_token;
}

// 발급 토큰 공유 캐시 — 패널 3개와 모니터·오케스트레이터 화면이 각자 발급하면 한 화면에서
// 같은 토큰을 다섯 번 만든다. 만료(서버 기본 3600초) 전에 새로 받도록 여유를 두고 다시 발급한다.
const TOKEN_TTL_MS = 50 * 60 * 1000;
let cachedToken = null;

/** 유효한 실데이터 토큰을 돌려준다(진행 중 요청은 공유, 실패하면 캐시를 비워 재시도 가능). */
export function getRealdataToken() {
  if (cachedToken && Date.now() - cachedToken.issuedAt < TOKEN_TTL_MS) return cachedToken.promise;
  const entry = {
    issuedAt: Date.now(),
    promise: issueRealdataToken().catch((err) => {
      if (cachedToken === entry) cachedToken = null;
      throw err;
    })
  };
  cachedToken = entry;
  return entry.promise;
}

export const getModels = (token) => apiGet(`${BASE}/models`, { token });

export const getDatasets = (token) => apiGet(`${BASE}/datasets`, { token });

export const createDataset = (token, modelId) =>
  apiSend("POST", `${BASE}/datasets`, { token, body: { model_id: modelId } });

export const startTrainingRun = (token, modelId, datasetId) =>
  apiSend("POST", `${BASE}/training-runs`, { token, body: { model_id: modelId, dataset_id: datasetId } });

export const getTrainingRun = (token, jobId) => apiGet(`${BASE}/training-runs/${jobId}`, { token });

export const listTrainingRuns = (token, modelId) =>
  apiGet(`${BASE}/training-runs`, { token, params: { model_id: modelId } });

export const getCandidates = (token, modelId) => apiGet(`${BASE}/models/${modelId}/candidates`, { token });

export const applyCandidate = (token, modelId, version) =>
  apiSend("POST", `${BASE}/models/${modelId}/candidates/${version}/apply`, { token });

export const restoreCandidate = (token, modelId, version, note) =>
  apiSend("POST", `${BASE}/models/${modelId}/restore/${version}`, { token, body: note ? { note } : {} });

export const getEvaluation = (token, modelId, version) =>
  apiGet(`${BASE}/models/${modelId}/evaluation`, { token, params: { version } });

export const getExplain = (token, modelId, version, baseYm, dongCode) =>
  apiGet(`${BASE}/models/${modelId}/explain`, { token, params: { version, base_ym: baseYm, dong_code: dongCode } });

export const getDrift = (token, modelId, version) =>
  apiGet(`${BASE}/models/${modelId}/drift`, { token, params: { version } });
