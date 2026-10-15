import { useCallback, useEffect, useRef, useState } from "react";
import Card from "./Card.jsx";
import { getRealdataToken } from "./realdata/realdataClient.jsx";
import { useAppState } from "../context/AppStateContext.jsx";
import { apiGet, apiSend } from "../lib/api.js";
import { dataModeEpoch } from "../lib/dataMode.js";
import { REAL_TARGETS, activeModelPair, matchingFlowBasis, inferenceRequest, validateInference, rankSensitivity, observedFeatureLabel, createRequestGate } from "../lib/realSimulation.js";

const EMPTY = { mode:"real", source:"real", instances:[] };
const number = (value) => Number.isFinite(value) ? value.toLocaleString("ko-KR",{maximumFractionDigits:3}) : "—";
const unit = (target) => target === "observed_sales_krw" ? "원" : "명";
const month = (value) => value ? `${String(value).slice(0,4)}-${String(value).slice(4)}` : "—";
const endpoint = (instance) => `/api/v3/modelops/instances/${encodeURIComponent(instance.id)}/versions/${encodeURIComponent(instance.active_version)}`;

function Evidence({ result, regions }) {
  const p = result.provenance;
  const evaluation = p.evaluation_period;
  return <details className="real-flow-evidence"><summary>모델·학습데이터와 계산 근거</summary>
    <dl>
      <dt>지역 / 학습 대상</dt><dd>{regions.find((r) => r.id === p.region_id)?.name ?? p.region_id} / {REAL_TARGETS[p.target] ?? p.target}</dd>
      <dt>모델 종류 / 활성 버전</dt><dd>{p.model_type === "factor_analysis" ? "요인분석" : "시뮬레이션"} / {p.version_id}</dd>
      <dt>등록 모델 / 학습 작업</dt><dd>{p.instance_id} / {p.job_id}</dd>
      <dt>등록 데이터셋 / 원본</dt><dd>{p.dataset_id} / {p.source_dataset_id}</dd>
      <dt>데이터 내용 해시</dt><dd>{p.dataset_hash}</dd>
      <dt>모델 파일 해시</dt><dd>{p.artifact_hash}</dd>
      <dt>관측 기간 / 예측월</dt><dd>{month(p.observed_from)}~{month(p.observed_to)} / {month(p.prediction_period)}</dd>
      <dt>검증 기간 / 평균 절대 오차</dt><dd>{evaluation ? `${month(evaluation.from)}~${month(evaluation.to)}` : "—"} / {number(p.metrics.mae)} {unit(p.target)} (기준선 {number(p.baseline.mae)} {unit(p.target)})</dd>
      <dt>입력 항목</dt><dd>{p.features.map((feature) => observedFeatureLabel(feature,p.target,result.dong_options)).join(", ")}</dd>
      {result.inference_id && <><dt>추론 / 예측 출력</dt><dd>{result.inference_id} / {number(result.prediction)} {unit(p.target)}</dd></>}
    </dl>
  </details>;
}

export default function RealSimulationFlow() {
  const { appData, currentRegion, setCurrentRegion, navigateToTab } = useAppState();
  const regions = appData.regions.map(({id,name}) => ({id,name}));
  const [catalog,setCatalog] = useState(EMPTY);
  const [catalogRevision,setCatalogRevision] = useState(0);
  const region=currentRegion.id;
  const [target,setTarget] = useState("observed_sales_krw");
  const [dong,setDong] = useState("");
  const [contexts,setContexts] = useState(null);
  const [fields,setFields] = useState({});
  const [analysis,setAnalysis] = useState(null);
  const [simulation,setSimulation] = useState(null);
  const [ranking,setRanking] = useState(null);
  const [busy,setBusy] = useState("");
  const [feedback,setFeedback] = useState(null);
  const mounted = useRef(true);
  const requestRef = useRef(createRequestGate(dataModeEpoch));
  const catalogRef = useRef(createRequestGate(dataModeEpoch));
  useEffect(() => { mounted.current=true; return () => { mounted.current=false; requestRef.current.cancel();catalogRef.current.cancel(); }; },[]);
  const guard = () => {
    const current=requestRef.current.begin();
    return () => mounted.current && current();
  };
  const reset = (inputs=true) => {
    requestRef.current.cancel();
    setAnalysis(null);setSimulation(null);setRanking(null);setFeedback(null);setBusy("");
    if (inputs) { setContexts(null);setFields({}); }
  };
  const load = useCallback(async () => {
    const isCurrent=catalogRef.current.begin();
    const current=() => mounted.current && isCurrent();
    requestRef.current.cancel();setCatalog(EMPTY);setDong("");
    setBusy("catalog");setContexts(null);setAnalysis(null);setSimulation(null);setRanking(null);setFeedback(null);
    try {
      const token=await getRealdataToken();
      if (!current()) return;
      const data=await apiGet("/api/v3/modelops/catalog",{token});
      if (!current()) return;
      if (data.mode!=="real" || data.source!=="real") throw new Error("실데이터 모델 목록을 확인하세요.");
      setCatalog(data);
      setCatalogRevision((revision) => revision+1);
    } catch (error) { if (current()) { setCatalog(EMPTY);setFeedback(error.message || "모델 목록을 다시 확인하세요."); } }
    finally { if (current()) setBusy(""); }
  },[]);
  useEffect(() => { load(); },[load]);
  const pair=activeModelPair(catalog,region,target);
  const pairKey=[pair.factor?.id,pair.factor?.active_version,pair.simulation?.id,pair.simulation?.active_version].join("|");
  useEffect(() => {
    setContexts(null);setAnalysis(null);setSimulation(null);setRanking(null);setFields({});
    if (!pair.factor || !pair.simulation) return;
    const current=guard();
    setBusy("inputs");setFeedback(null);
    (async () => {
      try {
        const token=await getRealdataToken();
        if (!current()) return;
        const [factor,sim]=await Promise.all([pair.factor,pair.simulation].map((model) => apiGet(endpoint(model)+"/inputs",{token,params:{dong_code:dong||undefined}})));
        if (!current()) return;
        if (!matchingFlowBasis(factor,sim)) throw new Error("두 활성 모델의 지역·학습 대상·데이터 내용이 다릅니다. 같은 자료로 학습한 버전을 확인하세요.");
        setContexts({factor,simulation:sim});
        setFields(Object.fromEntries(sim.controls.map((row) => [row.feature,String(sim.inputs[row.feature])])));
      } catch (error) { if (current()) setFeedback(error.message || "관측 입력을 다시 확인하세요."); }
      finally { if (current()) setBusy(""); }
    })();
    return () => { requestRef.current.cancel(); };
  },[pairKey,dong,catalogRevision]);
  const runAnalysis = async () => {
    if (!contexts || busy) return;
    const current=guard();setBusy("analysis");setFeedback(null);setAnalysis(null);setSimulation(null);setRanking(null);
    try {
      const token=await getRealdataToken();if (!current()) return;
      const result=await apiSend("POST",endpoint(pair.factor)+"/predict",{token,body:inferenceRequest(contexts.factor)});
      if (current()) setAnalysis(validateInference(result,contexts.factor));
    } catch (error) { if (current()) { if (error.status===409) setContexts(null);setFeedback(error.message || "요인분석을 다시 실행하세요."); } }
    finally { if (current()) setBusy(""); }
  };
  const runSimulation = async (event) => {
    event.preventDefault();
    if (!contexts || !analysis || busy) return;
    const changes=Object.fromEntries(Object.entries(fields).map(([name,value]) => [name,Number(value)]));
    if (Object.values(fields).some((value) => !value.trim()) || !Object.values(changes).every(Number.isFinite)) { setFeedback("모든 조정 입력에 유효한 숫자를 입력하세요.");return; }
    const current=guard();setBusy("simulation");setFeedback(null);setSimulation(null);setRanking(null);
    try {
      const token=await getRealdataToken();if (!current()) return;
      const result=await apiSend("POST",endpoint(pair.simulation)+"/predict",{token,body:inferenceRequest(contexts.simulation,changes)});
      if (current()) setSimulation(validateInference(result,contexts.simulation));
    } catch (error) { if (current()) { if (error.status===409) { setContexts(null);setAnalysis(null); } setFeedback(error.message || "시뮬레이션을 다시 실행하세요."); } }
    finally { if (current()) setBusy(""); }
  };
  const label=(feature) => observedFeatureLabel(feature,target,contexts?.simulation.dong_options);
  const missing=!pair.factor || !pair.simulation;
  return <div className="real-simulation-flow">
    <Card title="실데이터 요인분석·시뮬레이션" icon="fa-chart-line" dataSource="api">
      <div className="real-flow-controls">
        <label>지역<select value={region} disabled={busy==="catalog"} onChange={(e) => { reset();setDong("");setCurrentRegion(appData.regions.find((row) => row.id===e.target.value)); }}>{regions.map((row) => <option key={row.id} value={row.id}>{row.name}</option>)}</select></label>
        <label>학습 대상<select value={target} disabled={busy==="catalog"} onChange={(e) => { reset();setDong("");setTarget(e.target.value); }}>{Object.entries(REAL_TARGETS).map(([id,name]) => <option key={id} value={id}>{name}</option>)}</select></label>
        <label>행정동<select value={dong || contexts?.simulation.provenance.dong_code || ""} disabled={!contexts || Boolean(busy)} onChange={(e) => { reset();setDong(e.target.value); }}>
          {!contexts && <option value="">모델과 관측 입력 확인 후 선택</option>}
          {contexts?.simulation.dong_options.map((row) => <option key={row.code} value={row.code}>{row.name}</option>)}
        </select></label>
        <button className="btn btn-secondary" disabled={Boolean(busy)} onClick={load}>모델 목록 새로고침</button>
      </div>
      {busy && <p role="status">{busy==="catalog" ? "모델 목록 확인 중…" : busy==="inputs" ? "관측 입력 확인 중…" : "활성 모델로 계산 중…"}</p>}
      {feedback && <p className="async-feedback is-error" role="alert">{feedback}</p>}
      {!busy && (missing || !contexts) && <div className="real-flow-guide">
        <p>선택한 지역·학습 대상의 요인분석과 시뮬레이션 활성 버전이 필요합니다. 학습데이터 등록 → 파이프라인 학습 → 버전 활성화 순서로 준비하세요.</p>
        <button className="btn btn-secondary" onClick={() => navigateToTab("tab-dataops")}>DataOps에서 학습데이터 등록</button>{" "}
        <button className="btn btn-primary" onClick={() => navigateToTab("tab-mlops-orch")}>오케스트레이터에서 학습·활성화</button>
      </div>}
    </Card>
    {contexts && <>
      <Card title="① 활성 요인분석의 입력 기여도" icon="fa-magnifying-glass-chart" dataSource="api">
        <button className="btn btn-primary" disabled={Boolean(busy)} onClick={runAnalysis}>{analysis ? "요인분석 재실행" : "요인분석 실행"}</button>
        {analysis ? <>
          <p>기준 입력의 모델 예측: <strong>{number(analysis.prediction)} {unit(target)}</strong> · 예측월 {month(analysis.provenance.prediction_period)}</p>
          <p>각 기여도의 합에 기준값 {number(analysis.intercept)} {unit(target)}을 더하면 예측값이 됩니다.</p>
          <div className="table-container"><table><caption className="sr-only">활성 요인분석 버전의 실제 입력과 예측 기여도</caption>
            <thead><tr><th>입력 지표</th><th>관측 입력</th><th>예측 기여도 ({unit(target)})</th></tr></thead>
            <tbody>{[...analysis.contributions].sort((a,b) => Math.abs(b.value)-Math.abs(a.value)).map((row) => <tr key={row.feature}><td>{label(row.feature)}</td><td>{number(row.input)}</td><td>{number(row.value)}</td></tr>)}</tbody>
          </table></div>
        </> : <p>활성 요인분석 버전의 관측 입력으로 기여도를 계산하세요.</p>}
        <Evidence result={analysis ?? contexts.factor} regions={regions}/>
      </Card>
      <Card title="② 활성 시뮬레이션의 입력 가정 비교" icon="fa-sliders" dataSource="api">
        <p>과거 관측 지표의 입력 가정을 바꿔 모델 예측을 비교합니다. 입력 변화에 비례해 예측하며, 실제 정책 효과를 확인한 결과가 아닙니다.</p>
        <p>범위: 선택한 행정동의 학습 구간 {month(contexts.simulation.provenance.range_provenance.from)}~{month(contexts.simulation.provenance.range_provenance.to)} · 예측은 등록 관측 마지막 달 다음 달({month(contexts.simulation.provenance.prediction_period)}) 기준입니다.</p>
        <form onSubmit={runSimulation}>
          <div className="real-flow-inputs">{contexts.simulation.controls.map((control) => <label key={control.feature}>
            <span>{label(control.feature)}</span><input type="number" step="any" value={fields[control.feature] ?? ""} disabled={Boolean(busy)} required aria-label={label(control.feature)} onChange={(e) => { requestRef.current.cancel();setFields((prev) => ({...prev,[control.feature]:e.target.value}));setSimulation(null);setRanking(null);setFeedback(null); }}/>
            <span>관측 범위 {number(control.min)}~{number(control.max)} {unit(target)}</span>
          </label>)}</div>
          <button className="btn btn-primary" type="submit" disabled={Boolean(busy) || !analysis}>시뮬레이션 실행</button>{" "}
          <button className="btn btn-secondary" type="button" disabled={Boolean(busy)} onClick={() => { setFields(Object.fromEntries(contexts.simulation.controls.map((row) => [row.feature,String(row.default)])));setSimulation(null);setRanking(null); }}>관측 입력으로 초기화</button>
        </form>
        {!analysis && <p>먼저 요인분석을 실행하세요.</p>}
        {simulation && <>
          <p>기준 예측 {number(simulation.baseline_prediction)} {unit(target)} → 조정 예측 <strong>{number(simulation.prediction)} {unit(target)}</strong> · 모델 예측상 변화 {number(simulation.delta)} {unit(target)}</p>
          {simulation.outside_observed_range.length>0 && <p className="async-feedback is-error" role="status">관측 범위 밖 입력: {simulation.outside_observed_range.map(label).join(", ")}. 해당 구간의 예측 정확도는 확인되지 않았습니다.</p>}
          <p>검증 평균 절대 오차는 {number(simulation.provenance.metrics.mae)} {unit(target)}입니다. 예측 변화와 오차를 함께 확인하세요.</p>
        </>}
        <Evidence result={simulation ?? contexts.simulation} regions={regions}/>
      </Card>
      <Card title="③ 정책 추천" icon="fa-list-ol" dataSource="api">
        <p>관측 지표의 예측 민감도를 근거로 추천합니다. 각 관측 지표를 해당 학습 범위의 1%만큼 증가·감소한 결과 중, {REAL_TARGETS[target]} 증가 예측이 큰 방향을 지표별로 비교합니다. 변수끼리의 관계와 실제 실현 가능성을 함께 확인하세요.</p>
        <button className="btn btn-primary" disabled={Boolean(busy) || !simulation} onClick={() => setRanking(rankSensitivity(simulation))}>정책 추천 보기</button>
        {ranking && <>
          {!ranking.length ? <p>관측 범위 안에서 증가 예측을 만든 조정 항목이 없습니다. 모델과 입력 범위를 확인하세요.</p> : <div className="table-container"><table><caption className="sr-only">시뮬레이션 예측 변화량에 근거한 관측 지표 정책 추천 순위</caption>
            <thead><tr><th>순위</th><th>관측 지표 입력 가정</th><th>입력 변화 ({unit(target)})</th><th>모델 예측상 변화 ({unit(target)})</th></tr></thead>
            <tbody>{ranking.map((row) => <tr key={row.feature}><td>{row.rank}</td><td>{label(row.feature)} {row.to>row.from ? "증가" : "감소"}</td><td>{number(row.from)} → {number(row.to)}</td><td>{number(row.delta)}</td></tr>)}</tbody>
          </table></div>}
          <p>근거 추론: {simulation.inference_id} · 기준 예측 {number(simulation.prediction)} {unit(target)} · 시뮬레이션 활성 버전 {simulation.provenance.version_id}</p>
          <p>추천 순위는 모델 예측의 민감도이며, 정책의 실제 인과 효과를 검증한 결과가 아닙니다.</p>
          <Evidence result={simulation} regions={regions}/>
        </>}
      </Card>
    </>}
  </div>;
}
