import { useCallback, useEffect, useRef, useState } from "react";
import { getRealdataToken } from "../components/realdata/realdataClient.jsx";
import { apiGet, apiSend } from "../lib/api.js";
import { dataModeEpoch } from "../lib/dataMode.js";
import { activeModelPair, matchingFlowBasis, inferenceRequest, validateInference, createRequestGate } from "../lib/realSimulation.js";

const EMPTY = { mode: "real", source: "real", instances: [], datasets: [] };
const endpoint = (instance) =>
  `/api/v3/modelops/instances/${encodeURIComponent(instance.id)}/versions/${encodeURIComponent(instance.active_version)}`;

/**
 * 정책 시뮬레이터 실데이터 표시의 데이터 층 — 같은 지역·학습 대상의 활성 요인분석·시뮬레이션 버전을 찾아
 * 관측 입력(inputs)을 읽고, 요인분석·시뮬레이션 추론(predict)을 돌려 준다. 화면 구조는 페이지가 갖는다.
 * enabled=false(데모 표시)면 요청을 보내지 않는다.
 */
export function useRealSimulation({ enabled, region, target, dong }) {
  const [catalog, setCatalog] = useState(EMPTY);
  const [contexts, setContexts] = useState(null);
  const [loading, setLoading] = useState(enabled);
  const [feedback, setFeedback] = useState(null);
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
  const loadInputs = useCallback(async (models) => {
    const current = guard(inputGate);
    setContexts(null);
    if (!models.factor || !models.simulation) return null;
    const token = await getRealdataToken();
    if (!current()) return null;
    const [factor, simulation] = await Promise.all([models.factor, models.simulation].map((model) =>
      apiGet(endpoint(model) + "/inputs", { token, params: { dong_code: dong || undefined } })));
    if (!current()) return null;
    if (!matchingFlowBasis(factor, simulation)) throw new Error("두 활성 모델의 지역·학습 대상·데이터 내용이 다릅니다. 같은 자료로 학습한 버전을 확인하세요.");
    const next = { factor, simulation };
    setContexts(next);
    return next;
  }, [dong]);

  const loadCatalog = useCallback(async () => {
    const current = guard(catalogGate);
    const token = await getRealdataToken();
    if (!current()) return null;
    const data = await apiGet("/api/v3/modelops/catalog", { token });
    if (!current()) return null;
    if (data.mode !== "real" || data.source !== "real") throw new Error("실데이터 모델 목록을 확인하세요.");
    setCatalog(data);
    return data;
  }, []);

  // 페이지 진입·지역·학습 대상·행정동 변경 때 다시 읽는다(따로 새로고침 버튼을 두지 않는다).
  useEffect(() => {
    if (!enabled) return undefined;
    let cancelled = false;
    setLoading(true); setFeedback(null);
    (async () => {
      try {
        const data = await loadCatalog();
        if (!data || cancelled) return;
        await loadInputs(activeModelPair(data, region, target));
      } catch (error) {
        if (!cancelled && mounted.current) setFeedback(error.message || "모델 목록을 다시 확인하세요.");
      } finally {
        if (!cancelled && mounted.current) setLoading(false);
      }
    })();
    return () => { cancelled = true; inputGate.current.cancel(); predictGate.current.cancel(); };
  }, [enabled, region, target, loadCatalog, loadInputs]);

  // 요인분석 실행 — 1 목록·입력 재확인 → 2 추론 → 3 근거·합계 검증 → 4 완료. onStep(n)으로 단계 칩을 움직인다.
  const runAnalysis = useCallback(async (onStep) => {
    setFeedback(null);
    try {
      onStep(1);
      const data = await loadCatalog();
      if (!data) return null;
      const fresh = await loadInputs(activeModelPair(data, region, target));
      if (!fresh) throw new Error("이 지역·학습 대상의 활성 모델을 확인하세요.");
      const current = guard(predictGate);
      onStep(2);
      const token = await getRealdataToken();
      const result = await apiSend("POST", endpoint(activeModelPair(data, region, target).factor) + "/predict", { token, body: inferenceRequest(fresh.factor) });
      if (!current()) return null;
      onStep(3);
      const analysis = validateInference(result, fresh.factor);
      onStep(4);
      return analysis;
    } catch (error) {
      if (mounted.current) setFeedback(error.status === 409 ? "활성 버전이 바뀌었습니다. 다시 실행하세요." : error.message || "요인분석을 다시 실행하세요.");
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
      if (current()) setFeedback(error.status === 409 ? "활성 버전이 바뀌었습니다. 요인분석을 다시 실행하세요." : error.message || "시뮬레이션을 다시 확인하세요.");
      return null;
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [contexts, pairKey]);

  const datasetName = (id) => catalog.datasets?.find((row) => row.id === id || row.dataset_id === id)?.name ?? id;
  const ready = Boolean(enabled && contexts && pair.factor && pair.simulation);
  const missing = Boolean(enabled && !loading && !ready);
  return { catalog, contexts, loading, ready, missing, feedback, datasetName, runAnalysis, simulate };
}
