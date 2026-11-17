import { useEffect, useState } from "react";
import { getRealdataToken } from "../components/realdata/realdataClient.jsx";
import { apiGet } from "../lib/api.js";
import { dataModeEpoch } from "../lib/dataMode.js";

/**
 * ③ 릿지·Chronos-2 비교 — 활성 모델 버전의 forecast-compare(같은 학습데이터로 원격 예측까지 서버가 계산)를 읽는다.
 * model이 없으면 요청하지 않는다. 실패는 결과 없음(failed)으로만 두고 화면은 비교 줄을 그리지 않는다.
 */
export function useForecastCompare({ mode, model, dong }) {
  const key = model ? [mode, model.id, model.active_version, dong ?? ""].join("|") : null;
  const [state, setState] = useState({ key: null, result: null, failed: false });
  useEffect(() => {
    if (!key) return undefined;
    let alive = true;
    const epoch = dataModeEpoch();
    const current = () => alive && epoch === dataModeEpoch();
    const path = `/api/v3/modelops/instances/${encodeURIComponent(model.id)}/versions/${encodeURIComponent(model.active_version)}/forecast-compare`;
    getRealdataToken()
      .then((token) => apiGet(path, { token, params: { dong_code: dong || undefined } }))
      .then((result) => { if (current()) setState({ key, result, failed: false }); })
      .catch(() => { if (current()) setState({ key, result: null, failed: true }); });
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- key가 모드·모델·버전·행정동을 모두 담는다
  }, [key]);
  if (!key) return { result: null, loading: false, failed: false };
  return state.key === key ? { result: state.result, loading: false, failed: state.failed } : { result: null, loading: true, failed: false };
}
