// 실데이터 연계(R3·R4) 백엔드 호출 헬퍼 — DataOps 발급 흐름(POST /dataops/token/{source_id})을
// 그대로 재사용해 단일 토큰(scope: data:read data:write)으로 조회·쓰기를 모두 게이팅한다.
// JSX는 없지만 컴포넌트 소유 경계(components/realdata/*.jsx)를 지키기 위해 .jsx로 둔다.
import { apiGet, apiSend } from "../../lib/api.js";

const TOKEN_SOURCE = "realdata-namwon";
const BASE = "/api/v3/realdata";

export async function issueRealdataToken() {
  const res = await apiSend("POST", `/api/v3/dataops/token/${TOKEN_SOURCE}`, {});
  return res.access_token;
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
