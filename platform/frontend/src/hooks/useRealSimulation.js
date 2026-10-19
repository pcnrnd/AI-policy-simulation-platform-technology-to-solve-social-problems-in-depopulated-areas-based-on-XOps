import { useCallback, useEffect, useRef, useState } from "react";
import { getRealdataToken } from "../components/realdata/realdataClient.jsx";
import { apiGet, apiSend } from "../lib/api.js";
import { dataModeEpoch } from "../lib/dataMode.js";
import {
  activeModelPair, inferenceRequest, validateInference, createRequestGate,
  REAL_LOAD_FAILED_TEXT, REAL_PREDICT_FAILED_TEXT, REAL_VERSION_CHANGED_TEXT
} from "../lib/realSimulation.js";

const EMPTY = { mode: "real", source: "real", instances: [], datasets: [] };
const endpoint = (instance) =>
  `/api/v3/modelops/instances/${encodeURIComponent(instance.id)}/versions/${encodeURIComponent(instance.active_version)}`;
const rawText = (error) => error?.message || String(error);

/**
 * 정책 시뮬레이터 실데이터 표시의 데이터 층 — 같은 지역·학습 대상의 활성 시뮬레이션 버전을 찾아
 * 관측 입력(inputs)을 읽고, 시뮬레이션 추론(predict)을 돌려 준다. 요인분석 모델 유무와 무관하다.
 * 화면 구조는 페이지가 갖는다. enabled=false(데모 표시)면 요청을 보내지 않는다.
 * 알림 문구(feedback)는 고정 한글 문구이고, 서버·브라우저 원문은 log(WARN 한 줄)로만 남긴다.
 * revision: 활성 버전이 바뀌거나(409) 다시 확인(reload)할 때마다 오른다 — 페이지는 이 값으로 결과를 비운다.
 */
export function useRealSimulation({ enabled, region, target, dong, log }) {
  const [catalog, setCatalog] = useState(EMPTY);
  const [context, setContext] = useState(null);
  const [loading, setLoading] = useState(enabled);
  const [loadFailed, setLoadFailed] = useState(false); // 목록·입력 조회 실패(모델 없음과 구분)
  const [feedback, setFeedback] = useState(null); // 추론 실패·활성 버전 변경 안내
  const [revision, setRevision] = useState(0);
  const logRef = useRef(log);
  logRef.current = log;
  const mounted = useRef(true);
  const loadGate = useRef(createRequestGate(dataModeEpoch));
  const predictGate = useRef(createRequestGate(dataModeEpoch));
  const conflictKey = useRef(null); // 409 로 다시 읽은 모델 버전 — 같은 버전의 409 반복이면 다시 읽지 않는다
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; loadGate.current.cancel(); predictGate.current.cancel(); };
  }, []);
  const guard = (gate) => { const current = gate.current.begin(); return () => mounted.current && current(); };

  const model = enabled ? activeModelPair(catalog, region, target).simulation : undefined;
  const modelKey = `${model?.id}|${model?.active_version}`;

  // 페이지 진입·지역·학습 대상·행정동 변경·활성 버전 변경(409)·다시 확인 때 목록과 관측 입력을 다시 읽는다.
  useEffect(() => {
    if (!enabled) return undefined;
    const current = guard(loadGate);
    setContext(null); setLoading(true); setLoadFailed(false);
    (async () => {
      try {
        const token = await getRealdataToken();
        if (!current()) return;
        const data = await apiGet("/api/v3/modelops/catalog", { token });
        if (!current()) return;
        if (data.mode !== "real" || data.source !== "real") throw new Error(`catalog mode=${data.mode} source=${data.source}`);
        setCatalog(data);
        const active = activeModelPair(data, region, target).simulation;
        if (!active) return;
        const inputs = await apiGet(endpoint(active) + "/inputs", { token, params: { dong_code: dong || undefined } });
        if (current()) setContext(inputs);
      } catch (error) {
        if (!current()) return;
        setLoadFailed(true);
        logRef.current?.(`WARN: 실데이터 모델 정보 조회 실패 — ${rawText(error)}`);
      } finally {
        if (current()) setLoading(false);
      }
    })();
    return () => { loadGate.current.cancel(); predictGate.current.cancel(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, region, target, dong, revision]);
  // 지역·학습 대상·행정동이 바뀌면 이전 조건의 안내를 지운다(409 재조회는 revision 만 바뀌어 안내를 유지).
  useEffect(() => { setFeedback(null); }, [region, target, dong]);

  const reload = useCallback(() => setRevision((value) => value + 1), []);

  // 시뮬레이션 추론(읽기 전용 계산) — 늦게 온 이전 응답은 버린다.
  const simulate = useCallback(async (changes) => {
    if (!context || !model) return null;
    const current = guard(predictGate);
    // 다시 계산하는 동안 이전 실패 문구 대신 계산 중으로 보인다(활성 버전 변경 안내는 남긴다).
    setFeedback((value) => (value === REAL_PREDICT_FAILED_TEXT ? null : value));
    try {
      const token = await getRealdataToken();
      const result = await apiSend("POST", endpoint(model) + "/predict", { token, body: inferenceRequest(context, changes) });
      if (!current()) return null;
      const checked = validateInference(result, context);
      conflictKey.current = null;
      // 새 버전으로 다시 계산했으면 "활성 버전이 바뀌었습니다" 안내를 거둔다(기록은 활동 로그에 남아 있다).
      setFeedback((value) => (value === REAL_VERSION_CHANGED_TEXT ? null : value));
      return checked;
    } catch (error) {
      if (!current()) return null;
      logRef.current?.(`WARN: 다음 달 예측 계산 실패 — ${rawText(error)}`);
      // 409 는 같은 버전에서 한 번만 다시 읽는다 — 서버가 계속 409 를 주면(버전 변경이 아닌 근거 불일치)
      // 다시 읽기 → 자동 재계산 → 409 가 되풀이되지 않게 실패 문구로 멈춘다.
      if (error.status === 409 && conflictKey.current !== modelKey) {
        conflictKey.current = modelKey;
        setFeedback(REAL_VERSION_CHANGED_TEXT);
        reload();
      } else setFeedback(REAL_PREDICT_FAILED_TEXT);
      return null;
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [context, modelKey, reload]);

  const datasetName = (id) => catalog.datasets?.find((row) => row.id === id || row.dataset_id === id)?.name ?? id;
  const ready = Boolean(enabled && context && model);
  // 모델이 실제로 없을 때만 "준비" 안내를 띄운다(조회 실패는 고정 실패 문구로 알린다).
  const missing = Boolean(enabled && !loading && !ready && !loadFailed);
  return {
    catalog, context, loading, ready, missing, loadFailed: Boolean(enabled && loadFailed),
    predictFailed: feedback === REAL_PREDICT_FAILED_TEXT,
    feedback: enabled && loadFailed ? REAL_LOAD_FAILED_TEXT : feedback,
    revision, datasetName, simulate, reload
  };
}
