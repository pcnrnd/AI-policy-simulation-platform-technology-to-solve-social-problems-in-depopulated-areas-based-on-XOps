import { useCallback, useEffect, useRef, useState } from "react";
import { getRealdataToken } from "../components/realdata/realdataClient.jsx";
import { apiGet, apiSend } from "../lib/api.js";
import { dataModeEpoch } from "../lib/dataMode.js";
import { activeModelPair, matchingFlowBasis, inferenceRequest, validateInference, createRequestGate } from "../lib/realSimulation.js";

const EMPTY = { mode: "real", source: "real", instances: [], datasets: [] };
// 요청이 더 새 요청에 밀려 취소됨 — "모델 없음"(null)과 구분해 조용히 끝낸다.
const CANCELLED = Symbol("cancelled");
const VERSION_CHANGED = "활성 버전이 바뀌었습니다. 모델과 관측 입력을 다시 확인했으니 요인분석을 다시 실행하세요.";
const endpoint = (instance) =>
  `/api/v3/modelops/instances/${encodeURIComponent(instance.id)}/versions/${encodeURIComponent(instance.active_version)}`;

/**
 * 정책 시뮬레이터 실데이터 표시의 데이터 층 — 같은 지역·학습 대상의 활성 요인분석·시뮬레이션 버전을 찾아
 * 관측 입력(inputs)을 읽고, 요인분석·시뮬레이션 추론(predict)을 돌려 준다. 화면 구조는 페이지가 갖는다.
 * enabled=false(데모 표시)면 요청을 보내지 않는다.
 * revision: 활성 버전이 바뀌어(409) 입력을 다시 읽을 때마다 오른다 — 페이지는 이 값으로 분석 결과를 비운다.
 */
export function useRealSimulation({ enabled, region, target, dong }) {
  const [catalog, setCatalog] = useState(EMPTY);
  const [contexts, setContexts] = useState(null);
  const [loading, setLoading] = useState(enabled);
  const [loadError, setLoadError] = useState(null); // 목록·입력 조회 실패(모델 없음과 구분)
  const [feedback, setFeedback] = useState(null); // 실행(추론) 실패
  const [revision, setRevision] = useState(0);
  const mounted = useRef(true);
  const catalogGate = useRef(createRequestGate(dataModeEpoch));
  const inputGate = useRef(createRequestGate(dataModeEpoch));
  const predictGate = useRef(createRequestGate(dataModeEpoch));
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      catalogGate.current.cancel(); inputGate.current.cancel(); predictGate.current.cancel();
    };
  }, []);
  const guard = (gate) => { const current = gate.current.begin(); return () => mounted.current && current(); };

  const pair = enabled ? activeModelPair(catalog, region, target) : {};
  const pairKey = [pair.factor?.id, pair.factor?.active_version, pair.simulation?.id, pair.simulation?.active_version].join("|");

  // 같은 활성 두 버전의 관측 입력. 두 모델이 같은 자료·좌표로 학습됐을 때만 연결한다.
  // 새 값이 올 때까지 이전 값을 지우지 않는다(지역·대상 변경 때는 effect 가 먼저 비운다).
  const loadInputs = useCallback(async (models) => {
    const current = guard(inputGate);
    if (!models.factor || !models.simulation) { setContexts(null); return null; }
    const token = await getRealdataToken();
    if (!current()) return CANCELLED;
    const [factor, simulation] = await Promise.all([models.factor, models.simulation].map((model) =>
      apiGet(endpoint(model) + "/inputs", { token, params: { dong_code: dong || undefined } })));
    if (!current()) return CANCELLED;
    if (!matchingFlowBasis(factor, simulation)) {
      setContexts(null);
      throw new Error("두 활성 모델의 지역·학습 대상·데이터 내용이 다릅니다. 같은 자료로 학습한 버전을 확인하세요.");
    }
    const next = { factor, simulation };
    setContexts(next);
    return next;
  }, [dong]);

  const loadCatalog = useCallback(async () => {
    const current = guard(catalogGate);
    const token = await getRealdataToken();
    if (!current()) return CANCELLED;
    const data = await apiGet("/api/v3/modelops/catalog", { token });
    if (!current()) return CANCELLED;
    if (data.mode !== "real" || data.source !== "real") throw new Error("실데이터 모델 목록을 확인하세요.");
    setCatalog(data);
    return data;
  }, []);

  // 페이지 진입·지역·학습 대상·행정동 변경·활성 버전 변경(409) 때 다시 읽는다(따로 새로고침 버튼을 두지 않는다).
  useEffect(() => {
    if (!enabled) return undefined;
    let cancelled = false;
    setContexts(null); setLoading(true); setLoadError(null);
    (async () => {
      try {
        const data = await loadCatalog();
        if (data === CANCELLED || cancelled) return;
        await loadInputs(activeModelPair(data, region, target));
      } catch (error) {
        if (!cancelled && mounted.current) setLoadError(error.message || "모델 목록을 다시 확인하세요.");
      } finally {
        if (!cancelled && mounted.current) setLoading(false);
      }
    })();
    return () => { cancelled = true; inputGate.current.cancel(); predictGate.current.cancel(); };
  }, [enabled, region, target, revision, loadCatalog, loadInputs]);

  const onConflict = () => { setFeedback(VERSION_CHANGED); setRevision((value) => value + 1); };

  // 요인분석 실행 — 1 목록·입력 재확인 → 2 추론 → 3 근거·합계 검증 → 4 완료. onStep(n)으로 단계 칩을 움직인다.
  const runAnalysis = useCallback(async (onStep) => {
    setFeedback(null); setLoadError(null);
    let phase = "load"; // 목록·입력 단계 실패는 loadError(모델 없음 안내와 구분), 추론 단계 실패는 feedback
    try {
      onStep(1);
      const data = await loadCatalog();
      if (data === CANCELLED) return null;
      const models = activeModelPair(data, region, target);
      const fresh = await loadInputs(models);
      if (fresh === CANCELLED) return null;
      if (!fresh) throw new Error("이 지역·학습 대상의 활성 모델을 확인하세요.");
      phase = "predict";
      const current = guard(predictGate);
      onStep(2);
      const token = await getRealdataToken();
      const result = await apiSend("POST", endpoint(models.factor) + "/predict", { token, body: inferenceRequest(fresh.factor) });
      if (!current()) return null;
      onStep(3);
      const analysis = validateInference(result, fresh.factor);
      onStep(4);
      return analysis;
    } catch (error) {
      if (!mounted.current) return null;
      if (error.status === 409) onConflict();
      else if (phase === "load") setLoadError(error.message || "모델 목록을 다시 확인하세요.");
      else setFeedback(error.message || "요인분석을 다시 실행하세요.");
      return null;
    }
  }, [region, target, loadCatalog, loadInputs]);

  // 시뮬레이션 추론(읽기 전용 계산) — 늦게 온 이전 응답은 버린다.
  const simulate = useCallback(async (changes) => {
    if (!contexts || !pair.simulation) return null;
    const current = guard(predictGate);
    try {
      const token = await getRealdataToken();
      const result = await apiSend("POST", endpoint(pair.simulation) + "/predict", { token, body: inferenceRequest(contexts.simulation, changes) });
      if (!current()) return null;
      setFeedback(null);
      return validateInference(result, contexts.simulation);
    } catch (error) {
      if (!current()) return null;
      if (error.status === 409) onConflict();
      else setFeedback(error.message || "시뮬레이션을 다시 확인하세요.");
      return null;
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [contexts, pairKey]);

  const datasetName = (id) => catalog.datasets?.find((row) => row.id === id || row.dataset_id === id)?.name ?? id;
  const ready = Boolean(enabled && contexts && pair.factor && pair.simulation);
  // 모델이 실제로 없을 때만 "준비" 안내를 띄운다(조회 실패·근거 불일치는 loadError 문구로 알린다).
  const missing = Boolean(enabled && !loading && !ready && !loadError);
  return { catalog, contexts, loading, ready, missing, feedback: loadError ?? feedback, revision, datasetName, runAnalysis, simulate };
}
