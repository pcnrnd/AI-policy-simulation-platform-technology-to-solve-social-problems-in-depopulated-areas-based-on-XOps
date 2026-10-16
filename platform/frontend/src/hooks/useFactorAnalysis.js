import { useCallback, useEffect, useRef, useState } from "react";
import { getRealdataToken } from "../components/realdata/realdataClient.jsx";
import { apiGet, apiSend } from "../lib/api.js";
import { dataModeEpoch } from "../lib/dataMode.js";
import { belowBaseline, cardPipelineId, displayId } from "../lib/modelops.js";
import {
  activeModelPair, inferenceRequest, validateInference, createRequestGate, formatUnitValue, formatMonth, targetUnit,
  DEMO_FACTOR_SOURCE_PREFIX, FACTOR_FAILED_TEXT, FACTOR_VERSION_CHANGED_TEXT
} from "../lib/realSimulation.js";

const BASE = "/api/v3/modelops";
const JOB_POLL_MS = 500;
const JOB_POLL_MAX = 60; // 30초 — 넘으면 실패 문구로 멈춘다(서버 작업은 계속되고 다음 실행이 이어 쓴다)
// 요청이 더 새 실행·조건 변경·모드 전환에 밀려 취소됨 — 화면에 아무것도 남기지 않는다.
const CANCELLED = Symbol("cancelled");
// 실데이터에 요인분석 활성 모델이 없음 — 알림 대신 안내 상태로 둔다.
const MISSING = Symbol("missing");
const endpoint = (instance) =>
  `${BASE}/instances/${encodeURIComponent(instance.id)}/versions/${encodeURIComponent(instance.active_version)}`;
const rawText = (error) => error?.message || String(error);
const wait = (ms) => new Promise((resolve) => { setTimeout(resolve, ms); });

/**
 * 데모 표시에서 이 지역의 요인분석 활성 모델을 준비한다 — 학습 카드와 같은 경로(등록 학습데이터 → 결정적 ID 파이프라인
 * → 학습 → 활성화)를 그대로 쓰고, 이미 있는 등록본·파이프라인·후보 버전·진행 중 작업은 다시 만들지 않는다(멱등).
 * at(promise): 응답이 낡았으면(취소) CANCELLED 를 던진다. 반환: 활성화 뒤 다시 읽은 카탈로그.
 */
async function ensureDemoFactorModel({ token, catalog, region, at, log }) {
  const sources = (await at(apiGet(`${BASE}/available-datasets`, { token }))).data ?? [];
  const source = sources.find((row) => row.id === DEMO_FACTOR_SOURCE_PREFIX + region);
  if (!source) throw new Error(`demo factor source missing: ${region}`);
  let dataset = catalog.datasets.find((row) => row.source_dataset_id === source.id && row.content_hash === source.content_hash);
  if (!dataset) {
    dataset = await at(apiSend("POST", "/api/v3/modelops/datasets", { token, body: {
      name: `${source.name} (정책 시뮬레이터 자동 등록)`, source_dataset_id: source.id, source_hash: source.content_hash, id_prefix: "demo_factor_"
    } }));
    log(`INFO: 데모 요인분석 모델 준비 — 학습데이터 등록 ${displayId(dataset.id)}`);
  }
  const pipelineId = cardPipelineId("factor_analysis", dataset.id);
  let pipeline = catalog.pipelines.find((row) => row.id === pipelineId);
  if (!pipeline) {
    pipeline = await at(apiSend("POST", "/api/v3/modelops/pipelines", { token, body: {
      id: pipelineId, name: "요인분석 학습 (정책 시뮬레이터 데모)", model_type: "factor_analysis",
      dataset_id: dataset.id, dataset_hash: dataset.content_hash, trigger_policy: "자동(정책 시뮬레이터)"
    } }));
    log(`INFO: 데모 요인분석 모델 준비 — 파이프라인 등록 ${displayId(pipeline.id)}`);
  }
  let fresh = await at(apiGet(`${BASE}/catalog`, { token }));
  const instanceId = pipeline.instance_id;
  let version = fresh.versions.find((row) => row.instance_id === instanceId && row.dataset_id === dataset.id && !belowBaseline(row));
  if (!version) {
    const runningJob = (rows) => rows.jobs.find((row) => row.instance_id === instanceId && (row.state === "queued" || row.state === "running"));
    let job = runningJob(fresh);
    if (!job) {
      try {
        job = await at(apiSend("POST", `/api/v3/modelops/pipelines/${encodeURIComponent(pipeline.id)}/run`, { token }));
      } catch (error) {
        // 409: 목록을 읽은 뒤 다른 화면이 같은 모델 학습을 시작했다 — 그 작업을 이어 폴링한다.
        if (error?.status !== 409) throw error;
        job = runningJob(await at(apiGet(`${BASE}/catalog`, { token })));
        if (!job) throw error;
      }
    }
    log(`INFO: 데모 요인분석 모델 준비 — 학습 시작 ${displayId(job.id)}`);
    for (let tries = 0; job.state !== "saved" && job.state !== "failed"; tries += 1) {
      if (tries >= JOB_POLL_MAX) throw new Error(`training poll timeout: ${job.id}`);
      await at(wait(JOB_POLL_MS));
      job = await at(apiGet(`${BASE}/jobs/${encodeURIComponent(job.id)}`, { token }));
    }
    if (job.state !== "saved") throw new Error(`training ${job.state}: ${job.error ?? ""}`);
    fresh = await at(apiGet(`${BASE}/catalog`, { token }));
    version = fresh.versions.find((row) => row.id === job.version_id);
    if (!version || belowBaseline(version)) throw new Error(`version not activatable: ${job.version_id}`);
  }
  await at(apiSend("POST", `/api/v3/modelops/instances/${encodeURIComponent(instanceId)}/versions/${encodeURIComponent(version.id)}/activate`, { token }));
  log(`INFO: 데모 요인분석 모델 준비 — 활성화 ${displayId(version.id)}`);
  return at(apiGet(`${BASE}/catalog`, { token }));
}

/**
 * 정책 시뮬레이터 STAGE ① 요인분석 — 두 모드가 같은 흐름을 쓴다(모드는 요청 헤더로 데이터 출처만 바뀐다).
 * [요인분석 실행] → 1 목록·(데모: 모델 준비)·관측 입력 → 2 추론 → 3 기여도 합계 확인 → 4 완료.
 * 데모 표시만 활성 모델이 없을 때 합성 요인 패널로 자동 준비하고, 실데이터는 활성 모델이 있어야 실행된다.
 * 알림(feedback)은 고정 문구이고 서버·브라우저 원문은 log(WARN 한 줄)로만 남긴다.
 * 지역·학습 대상·행정동이 바뀌면 진행 중 실행을 버리고 처음 상태로 돌아간다.
 */
export function useFactorAnalysis({ demo, region, target, dong, log }) {
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
  }, [demo, region, target, dong]);

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
      let catalog = await at(apiGet(`${BASE}/catalog`, { token }));
      let model = activeModelPair(catalog, region, target, mode).factor;
      if (!model && demo) {
        catalog = await ensureDemoFactorModel({ token, catalog, region, at, log: say });
        model = activeModelPair(catalog, region, target, mode).factor;
      }
      if (!model) throw MISSING;
      phase = "predict";
      const inputs = await at(apiGet(endpoint(model) + "/inputs", { token, params: { dong_code: dong || undefined } }));
      const dataset = catalog.datasets.find((row) => row.id === inputs.provenance.dataset_id) ?? null;
      const proof = inputs.provenance;
      say(`INFO: [요인분석 1/4] 학습데이터 ${dataset?.name ?? proof.dataset_id} · 관측 ${formatMonth(proof.observed_from)}~${formatMonth(proof.observed_to)} 확인`);
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
  }, [demo, region, target, dong]);

  return { status, step, result, feedback, missing, run };
}
