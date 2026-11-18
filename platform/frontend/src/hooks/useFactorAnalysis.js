import { useCallback, useEffect, useRef, useState } from "react";
import { getRealdataToken } from "../components/realdata/realdataClient.jsx";
import { apiGet, apiSend } from "../lib/api.js";
import { dataModeEpoch } from "../lib/dataMode.js";
import { displayId } from "../lib/modelops.js";
import {
  selectedFactorModel, inferenceRequest, factorAnalysisView, yoyChange, rankTargets, REAL_TARGETS, validateInference, createRequestGate, formatUnitValue, formatMonth, targetUnit,
  periodBounds, resolvePeriod, periodRequest, validatePeriodAnalysis, FACTOR_FAILED_TEXT, FACTOR_VERSION_CHANGED_TEXT, FACTOR_PERIOD_EMPTY_TEXT
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
 * 지역·학습 대상·행정동·분석 기간·비교 기준이 바뀌면 진행 중 실행을 버리고 처음 상태로 돌아간다.
 * period: ① 분석 기간 선택({ from, to } YYYYMM, null이면 최근 12개월) — 실행 때 결과 모델의 관측 범위로 맞춘다. baseline: 비교 기준 id.
 */
export function useFactorAnalysis({ demo, region, target, dong, modelId, versionId, period = null, baseline = "train_mean", log }) {
  const periodFrom = period?.from ?? null;
  const periodTo = period?.to ?? null;
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
  const [result, setResult] = useState(null); // { analysis(추론 결과), dataset(카탈로그 행), period(기간 분석 결과 | null) }
  const [feedback, setFeedback] = useState(null);
  const [missing, setMissing] = useState(false);
  const [ranking, setRanking] = useState(null); // 자동 실행의 지표별 순위 [{ target, change }]
  const adopt = useRef(null); // 자동 실행이 고른 지표로 조건이 바뀐 뒤 결과를 이어 받는다
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
    const a = adopt.current;
    adopt.current = null;
    if (a && a.region === region && a.target === target && a.dong === dong && a.modelId === modelId && a.versionId === versionId) {
      setResult(a.result); setRanking(a.ranking); setStatus("done"); setStep(4); setFeedback(null); setMissing(false);
      return;
    }
    setStatus("idle"); setStep(0); setResult(null); setRanking(null); setFeedback(null); setMissing(false);
  }, [demo, region, target, dong, modelId, versionId, periodFrom, periodTo, baseline]);

  // ① 분석 기간 기여도 — 예측월 추론(analysis) 뒤에 같은 모델·입력으로. 기간을 고를 수 없는 옛 지역 원천 모델은 null.
  const analyzePeriod = async (at, model, inputs, token) => {
    const range = resolvePeriod(periodBounds(inputs.provenance, inputs.provenance.mode === "mock"), periodFrom && periodTo ? { from: periodFrom, to: periodTo } : null);
    if (!range) return null;
    const request = periodRequest(inputs, range, baseline);
    const raw = await at(apiSend("POST", endpoint(model) + "/factor-period", { token, body: request }));
    return validatePeriodAnalysis(raw, inputs, request);
  };
  // 알림은 고정 문구 — 상태 코드로만 고른다(원문은 로그 한 줄).
  const failText = (phase, status) => (phase === "period" && status === 422 ? FACTOR_PERIOD_EMPTY_TEXT
    : phase !== "prepare" && status === 409 ? FACTOR_VERSION_CHANGED_TEXT : FACTOR_FAILED_TEXT);

  // accept: 결과를 게시하기 직전에 부른다 — false면 결과를 버리고 처음 상태로 둔다(자동 계산 중 입력이 바뀐 경우).
  const run = useCallback(async ({ accept } = {}) => {
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
      const analysis = validateInference(raw, inputs);
      phase = "period";
      const period = await analyzePeriod(at, model, inputs, token);
      setStep(3);
      say(`INFO: [요인분석 3/4] 요인 ${analysis.contributions.length}개 기여도 분해 — 합계 확인`);
      setStep(4);
      const unit = targetUnit(proof.target);
      say(`INFO: [요인분석 4/4] 검증 평균 절대 오차 ${formatUnitValue(proof.metrics?.mae, unit)} / 기준선 ${formatUnitValue(proof.baseline?.mae, unit)} ${unit}`);
      if (accept && !accept()) { setStatus("idle"); setStep(0); return null; }
      setResult({ analysis, dataset, period });
      setStatus("done");
      return analysis;
    } catch (error) {
      if (error === CANCELLED || !current()) return null;
      setStatus("idle"); setStep(0);
      if (error === MISSING) { setMissing(true); return null; }
      say(`WARN: 요인분석 실행 실패 — ${rawText(error)}`);
      const code = error?.status;
      setFeedback(failText(phase, code));
      return null;
    }
  }, [demo, region, target, dong, modelId, versionId, periodFrom, periodTo, baseline]);

  // 자동(가장 나빠진 지표) — 이 지역에 활성 요인분석 모델이 있는 학습 대상마다 같은 추론을 돌려, 다음 달 추정이
  // 전년 같은 달보다 가장 많이 줄어든 지표를 고른다(전년 비교 불가는 맨 뒤). 고른 지표로 조건을 옮기고(onPick) 그 결과를 보인다.
  const runAuto = useCallback(async ({ onPick }) => {
    const begun = gate.current.begin();
    const current = () => mounted.current && begun();
    const at = async (promise) => { const value = await promise; if (!current()) throw CANCELLED; return value; };
    const say = (message) => logRef.current?.(message);
    setStatus("running"); setStep(1); setResult(null); setRanking(null); setFeedback(null); setMissing(false);
    let phase = "prepare";
    try {
      const token = await at(getRealdataToken());
      const catalog = await at(apiGet(`${BASE}/catalog`, { token }));
      setCatalog(catalog);
      const found = [];
      for (const t of Object.keys(REAL_TARGETS)) {
        const inst = catalog.instances?.find((row) => row.region_id === region && row.target === t && row.model_type === "factor_analysis" && row.active_version);
        const model = inst && selectedFactorModel(catalog, inst.id, region, t, "real");
        if (!model) continue;
        const inputs = await at(apiGet(endpoint(model) + "/inputs", { token, params: { dong_code: dong || undefined } }));
        setStep(2);
        const raw = await at(apiSend("POST", endpoint(model) + "/predict", { token, body: inferenceRequest(inputs) }));
        const analysis = validateInference(raw, inputs);
        const dataset = catalog.datasets.find((row) => row.id === inputs.provenance.dataset_id) ?? null;
        const change = yoyChange(factorAnalysisView(analysis, { demo: false, dataset, label: (f) => f }));
        found.push({ target: t, change, model, analysis, dataset, inputs });
        say(`INFO: [요인분석 자동] ${REAL_TARGETS[t]} 추론 완료`);
      }
      if (!found.length) throw MISSING;
      const ranked = rankTargets(found);
      const best = ranked[0];
      // 지표 순위는 예측월 추론으로 고르고, 고른 지표 하나만 기간 분석을 이어서 한다.
      phase = "period";
      const period = await analyzePeriod(at, best.model, best.inputs, token);
      setStep(4);
      const summary = ranked.map(({ target: t, change }) => ({ target: t, change }));
      const next = { result: { analysis: best.analysis, dataset: best.dataset, period }, ranking: summary };
      if (best.target === target && best.model.id === modelId && best.model.active_version === versionId) {
        setResult(next.result); setRanking(summary); setStatus("done");
      } else {
        adopt.current = { ...next, region, target: best.target, dong, modelId: best.model.id, versionId: best.model.active_version };
        onPick(best.model);
      }
      return best.analysis;
    } catch (error) {
      if (error === CANCELLED || !current()) return null;
      setStatus("idle"); setStep(0);
      if (error === MISSING) { setMissing(true); return null; }
      say(`WARN: 요인분석 실행 실패 — ${rawText(error)}`);
      const code = error?.status;
      setFeedback(failText(phase, code));
      return null;
    }
  }, [region, target, dong, modelId, versionId, periodFrom, periodTo, baseline]);

  // 같은 조건에서 이미 끝난 결과(앱 상태에 보존)를 다시 계산하지 않고 되살린다 — 진행 중 실행은 버린다.
  const restore = useCallback((saved) => {
    gate.current.cancel();
    setResult(saved); setStatus("done"); setStep(4); setFeedback(null); setMissing(false);
  }, []);

  return { catalog, loading, loadFailed, reload: () => setRevision((value) => value + 1), status, step, result, feedback, missing, ranking, run, runAuto, restore };
}
