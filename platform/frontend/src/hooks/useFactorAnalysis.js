import { useCallback, useEffect, useRef, useState } from "react";
import { getRealdataToken } from "../components/realdata/realdataClient.jsx";
import { apiGet, apiSend } from "../lib/api.js";
import { dataModeEpoch } from "../lib/dataMode.js";
import { displayId } from "../lib/modelops.js";
import {
  selectedFactorModel, inferenceRequest, validateInference, createRequestGate, formatUnitValue, formatMonth, targetUnit,
  FACTOR_FAILED_TEXT, FACTOR_VERSION_CHANGED_TEXT
} from "../lib/realSimulation.js";

const BASE = "/api/v3/modelops";
// 요청이 더 새 실행·조건 변경·모드 전환에 밀려 취소됨 — 화면에 아무것도 남기지 않는다.
const CANCELLED = Symbol("cancelled");
// 실데이터에 요인분석 활성 모델이 없음 — 알림 대신 안내 상태로 둔다.
const MISSING = Symbol("missing");
const endpoint = (instance) =>
  `${BASE}/instances/${encodeURIComponent(instance.id)}/versions/${encodeURIComponent(instance.active_version)}`;
const rawText = (error) => error?.message || String(error);
/**
 * 정책 시뮬레이터 STAGE ① 요인분석 — 두 모드가 같은 흐름을 쓴다(모드는 요청 헤더로 데이터 출처만 바뀐다).
 * [요인분석 실행] → 1 선택한 활성 모델·관측 입력 → 2 추론 → 3 기여도 합계 확인 → 4 완료.
 * 두 모드 모두 기존 활성 모델만 사용하며 이 화면에서 등록·학습·활성화하지 않는다.
 * 알림(feedback)은 고정 문구이고 서버·브라우저 원문은 log(WARN 한 줄)로만 남긴다.
 * 지역·학습 대상·행정동이 바뀌면 진행 중 실행을 버리고 처음 상태로 돌아간다.
 */
export function useFactorAnalysis({ demo, region, target, dong, modelId, versionId, log }) {
  const [catalog, setCatalog] = useState(null);
  const [loading, setLoading] = useState(true);
  const [loadFailed, setLoadFailed] = useState(false);
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    let alive = true;
    const epoch = dataModeEpoch();
    const current = () => alive && epoch === dataModeEpoch();
    setCatalog(null); setLoading(true); setLoadFailed(false);
    getRealdataToken().then((token) => apiGet(`${BASE}/catalog`, { token })).then((data) => {
      if (!current()) return;
      const mode = demo ? "mock" : "real";
      if (data.mode !== mode || data.source !== mode) throw new Error("모델 목록의 데이터 모드를 확인하세요.");
      setCatalog(data);
    }).catch(() => { if (current()) setLoadFailed(true); })
      .finally(() => { if (current()) setLoading(false); });
    return () => { alive = false; };
  }, [demo, revision]);
  const [status, setStatus] = useState("idle"); // idle | running | done
  const [step, setStep] = useState(0); // 1-based 진행 단계
  const [result, setResult] = useState(null); // { analysis(추론 결과), dataset(카탈로그 행) }
  const [feedback, setFeedback] = useState(null);
  const [missing, setMissing] = useState(false);
  const logRef = useRef(log);
  logRef.current = log;
  const mounted = useRef(true);
  const gate = useRef(createRequestGate(dataModeEpoch));
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; gate.current.cancel(); };
  }, []);

  useEffect(() => {
    gate.current.cancel();
    setStatus("idle"); setStep(0); setResult(null); setFeedback(null); setMissing(false);
  }, [demo, region, target, dong, modelId, versionId]);

  const run = useCallback(async () => {
    const begun = gate.current.begin();
    const current = () => mounted.current && begun();
    const at = async (promise) => { const value = await promise; if (!current()) throw CANCELLED; return value; };
    const say = (message) => logRef.current?.(message);
    const mode = demo ? "mock" : "real";
    setStatus("running"); setStep(1); setResult(null); setFeedback(null); setMissing(false);
    let phase = "prepare"; // 409 는 추론 단계에서만 "활성 버전 변경"이다(준비 단계 409 는 일반 실패 문구)
    try {
      const token = await at(getRealdataToken());
      const catalog = await at(apiGet(`${BASE}/catalog`, { token }));
      setCatalog(catalog);
      const model = selectedFactorModel(catalog, modelId, region, target, mode);
      if (!model) throw MISSING;
      if (model.active_version !== versionId) {
        setFeedback(FACTOR_VERSION_CHANGED_TEXT); setStatus("idle"); setStep(0);
        return null;
      }
      phase = "predict";
      const inputs = await at(apiGet(endpoint(model) + "/inputs", { token, params: { dong_code: dong || undefined } }));
      const dataset = catalog.datasets.find((row) => row.id === inputs.provenance.dataset_id) ?? null;
      const proof = inputs.provenance;
      say(`INFO: [요인분석 1/4] 학습데이터 ${dataset?.name ?? displayId(proof.dataset_id)} · 관측 ${formatMonth(proof.observed_from)}~${formatMonth(proof.observed_to)} 확인`);
      setStep(2);
      say(`INFO: [요인분석 2/4] 요인분석 활성 버전 ${displayId(model.active_version)} 추론`);
      const raw = await at(apiSend("POST", endpoint(model) + "/predict", { token, body: inferenceRequest(inputs) }));
      setStep(3);
      const analysis = validateInference(raw, inputs);
      say(`INFO: [요인분석 3/4] 요인 ${analysis.contributions.length}개 기여도 분해 — 합계 확인`);
      setStep(4);
      const unit = targetUnit(proof.target);
      say(`INFO: [요인분석 4/4] 검증 평균 절대 오차 ${formatUnitValue(proof.metrics?.mae, unit)} / 기준선 ${formatUnitValue(proof.baseline?.mae, unit)} ${unit}`);
      setResult({ analysis, dataset });
      setStatus("done");
      return analysis;
    } catch (error) {
      if (error === CANCELLED || !current()) return null;
      setStatus("idle"); setStep(0);
      if (error === MISSING) { setMissing(true); return null; }
      say(`WARN: 요인분석 실행 실패 — ${rawText(error)}`);
      const conflict = phase === "predict" && error?.status === 409;
      setFeedback(conflict ? FACTOR_VERSION_CHANGED_TEXT : FACTOR_FAILED_TEXT);
      return null;
    }
  }, [demo, region, target, dong, modelId, versionId]);

  return { catalog, loading, loadFailed, reload: () => setRevision((value) => value + 1), status, step, result, feedback, missing, run };
}
