import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { instanceLabel } from "./modelops.js";
import { ANALYSIS_STEPS } from "./simulation.js";
import { activeModelPair, selectedFactorModel, inferenceMatches, inferenceContextKey, inferenceRequest, validateInference, rankSensitivity, observedFeatureLabel, createRequestGate,
  contributionFactors, contributionSplit, realSliderControls, forecastSeries, sensitivityRecommendation, regionRealStatus,
  REAL_GUIDE_TEXT, FACTOR_GUIDE_TEXT, FACTOR_PENDING_TEXT, FACTOR_RUN_LOCK_TEXT, REAL_LOAD_FAILED_TEXT, REAL_PREDICT_FAILED_TEXT, REAL_VERSION_CHANGED_TEXT,
  FACTOR_IDLE_TEXT, FACTOR_FAILED_TEXT, FACTOR_VERSION_CHANGED_TEXT, FACTOR_EXPLAIN_TEXT,
  DEMO_FACTOR_TARGET, FACTOR_PANEL_FEATURES, FACTOR_LIST_LIMIT, factorAnalysisView, diagnosisView, simFrameView, monthText, scopeFactors, isNonPolicyFeature, FACTOR_SCOPES, signedAmount, quarterYoyChange, yoyChange, rankTargets, AUTO_TARGET_LABEL, featureUnit, targetUnit, formatUnitValue, sliderValueText,
  forecastCompareView, REMOTE_FORECAST_GUIDE_TEXT } from "./realSimulation.js";
const proof={instance_id:"factor",model_type:"factor_analysis",version_id:"v1",job_id:"job1",dataset_id:"registered1",dataset_hash:"a".repeat(64),region_id:"namwon",target:"observed_sales_krw",dong_code:"45190250",prediction_period:202401,features:["y_lag1"],mode:"real",source:"real"};
const sim={provenance:{...proof,instance_id:"sim",model_type:"simulation",dataset_id:"registered2",version_id:"v2",job_id:"job2"}};
const request=inferenceRequest(sim,{y_lag1:100});
assert.equal(request.dataset_id,"registered2");assert.equal(request.dataset_hash,proof.dataset_hash);assert.deepEqual(request.adjustments,{y_lag1:100});
const result={provenance:sim.provenance,inference_id:"inference_new",intercept:10,prediction:30,baseline_prediction:25,delta:5,contributions:[{feature:"y_lag1",value:20}],recommendations:[
  {feature:"y_lag1",from:10,to:11,prediction:33,delta:3},{feature:"y_lag1",from:10,to:9,prediction:27,delta:-3},
  {feature:"y_lag2",from:20,to:19,prediction:35,delta:5},{feature:"invalid",from:1,to:2,prediction:NaN,delta:9}]};
assert.equal(validateInference(result,sim),result);
assert.throws(() => validateInference({...result,provenance:{...sim.provenance,job_id:"wrong"}},sim));
assert.throws(() => validateInference({...result,prediction:99},sim));
const ranked=rankSensitivity(result);
assert.deepEqual(ranked.map((row) => row.feature),["y_lag2","y_lag1"]);
assert.equal(ranked[0].inference_id,result.inference_id);assert.equal(ranked[0].provenance.dataset_id,"registered2");assert.equal(ranked[0].reference_prediction,30);
assert.throws(() => rankSensitivity({...result,provenance:proof}));
const pair=activeModelPair({mode:"real",source:"real",instances:[{id:"f",mode:"real",region_id:"namwon",target:"observed_sales_krw",model_type:"factor_analysis",active_version:"v1"},{id:"s",mode:"real",region_id:"namwon",target:"observed_sales_krw",model_type:"simulation",active_version:null}]},"namwon","observed_sales_krw");
assert.equal(pair.factor.id,"f");assert.equal(pair.simulation,undefined);
assert.throws(() => activeModelPair({mode:"mock",source:"mock",instances:[]},"namwon","observed_sales_krw"));
assert.equal(observedFeatureLabel("y_lag1","observed_sales_krw"),"전월 관측 소비");
assert.equal(observedFeatureLabel("dong_45190250","observed_sales_krw",[{code:"45190250",name:"운봉읍"}]),"행정동: 운봉읍");
let epoch=1;
const catalogGate=createRequestGate(() => epoch), inputGate=createRequestGate(() => epoch);
const loadingCatalog=catalogGate.begin(), previousInput=inputGate.begin();
inputGate.cancel();
assert.equal(previousInput(),false);assert.equal(loadingCatalog(),true);
const selectedInput=inputGate.begin();
epoch+=2;
assert.equal(selectedInput(),false);assert.equal(loadingCatalog(),false);
// ── 시뮬레이터 4단계 화면 매핑 ──
const label=(feature) => observedFeatureLabel(feature,"observed_sales_krw",[{code:"45190250",name:"운봉읍"}]);
const analysis={contributions:[{feature:"y_lag1",input:5,value:2},{feature:"y_lag2",input:7,value:-9},{feature:"dong_45190250",input:1,value:0}]};
const factors=contributionFactors(analysis,label);
assert.deepEqual(factors.map((row) => row.feature),["y_lag2","y_lag1","dong_45190250"]);
assert.deepEqual(factors.map((row) => row.direction),["-","+","+"]);
assert.equal(factors[2].label,"행정동: 운봉읍");
const split=contributionSplit(analysis,label);
assert.deepEqual([split.positive.map((row) => row.label),split.negative.map((row) => row.label)],[["전월 관측 소비"],["2개월 전 관측 소비"]]);
assert.deepEqual([split.positive[0].value,split.negative[0].value],[2,-9]);
assert.deepEqual(contributionSplit(null,label),{positive:[],negative:[]});
const sliders=realSliderControls({controls:[{feature:"y_lag1",min:1000,max:3730,default:2000},{feature:"y_yoy",min:5,max:5,default:5}],inputs:{}},label,"원");
assert.deepEqual([sliders[0].min,sliders[0].max,sliders[0].default,sliders[0].step,sliders[0].unit,sliders[0].locked],[1000,3730,2000,27,"원",false]);
assert.equal(sliders[1].step,1);assert.equal(sliders[1].locked,true);
assert.deepEqual(realSliderControls(null,label,"원"),[]);
const series=forecastSeries({inputs:{y_lag1:30,y_lag2:20,y_lag3:10},provenance:{prediction_period:202410}},{baseline_inputs:{y_lag1:30,y_lag2:20,y_lag3:10},baseline_prediction:40,prediction:44,provenance:{prediction_period:202410}});
assert.deepEqual(series.labels,["3개월 전","2개월 전","전월","예측월 2024-10"]);
assert.deepEqual(series.baseline,[10,20,30,40]);assert.deepEqual(series.adjusted,[10,20,30,44]);
const view=sensitivityRecommendation(ranked,label,"원","관측 소비");
assert.deepEqual(view.items.map((row) => row.name),["2개월 전 관측 소비 감소","전월 관측 소비 증가"]);
assert.equal(view.items[0].score,5);assert.equal(view.columns.length,view.items[0].cells.length+1);
assert.doesNotMatch(JSON.stringify(view),/RICE|점수|"점"/);
assert.ok(!("footnote" in view));
const statusCatalog={instances:[{mode:"real",region_id:"namwon",target:"observed_sales_krw",model_type:"factor_analysis",active_version:"v1"},{mode:"real",region_id:"namwon",target:"observed_sales_krw",model_type:"simulation",active_version:"v2"},{mode:"mock",region_id:"shinan",target:"observed_sales_krw",model_type:"factor_analysis",active_version:"v3"},{mode:"mock",region_id:"shinan",target:"observed_sales_krw",model_type:"simulation",active_version:"v4"}]};
assert.match(regionRealStatus(statusCatalog,"namwon","observed_sales_krw"),/준비됨/);
assert.equal(regionRealStatus(statusCatalog,"namwon","nonlocal_visitors"),"실데이터 모델 없음");
assert.equal(regionRealStatus(statusCatalog,"shinan","observed_sales_krw"),"실데이터 모델 없음");
// ③·④는 요인분석 모델과 무관 — 시뮬레이션 활성 버전만 있어도 준비됨, 요인분석만 있으면 없음.
assert.match(regionRealStatus({instances:[statusCatalog.instances[1]]},"namwon","observed_sales_krw"),/준비됨/);
assert.equal(regionRealStatus({instances:[statusCatalog.instances[0]]},"namwon","observed_sales_krw"),"실데이터 모델 없음");
// 안내·알림 문구에 내부 사유·구현명·서버/브라우저 원문 금지
const INTERNAL=/세종대|미수신|릿지|ridge|lag|XGBoost|SHAP|HTTP|fetch|Failed|Error|mock|seed|시드|modelops|\/api\//i;
const guideTexts=[REAL_GUIDE_TEXT,FACTOR_GUIDE_TEXT,FACTOR_PENDING_TEXT,FACTOR_RUN_LOCK_TEXT,REAL_LOAD_FAILED_TEXT,REAL_PREDICT_FAILED_TEXT,REAL_VERSION_CHANGED_TEXT,
  FACTOR_IDLE_TEXT,FACTOR_FAILED_TEXT,FACTOR_VERSION_CHANGED_TEXT,FACTOR_EXPLAIN_TEXT];
// 인과로 과장하는 표현 금지.
const CAUSAL=/효과가 있|때문에|유발|원인|덕분/;
for (const text of guideTexts) { assert.doesNotMatch(text,INTERNAL,text); assert.doesNotMatch(text,CAUSAL,text); }
assert.match(REAL_GUIDE_TEXT,/시뮬레이션 활성 모델이 없습니다/);
assert.match(REAL_VERSION_CHANGED_TEXT,/^활성 버전이 바뀌었습니다/);
// 구조 가드 — 시뮬레이터는 모드별 다른 컴포넌트를 띄우지 않는다(한 구조, 모드는 데이터만 바꾼다).
const here=dirname(fileURLToPath(import.meta.url));
const page=readFileSync(join(here,"..","pages","SimulatorPage.jsx"),"utf-8");
assert.doesNotMatch(page,/mockDataVisible\s*\?\s*</);
assert.doesNotMatch(page,/RealSimulationFlow/);
assert.equal(existsSync(join(here,"..","components","RealSimulationFlow.jsx")),false);
// 별칭(const demo = mockDataVisible, { mockDataVisible: isMock } 등)으로 모드를 나눠도 잡는다.
// 모드 분기(? · && · ? x :) 바로 뒤에 올 수 있는 컴포넌트는 두 모드가 함께 쓰는 부품뿐이다.
const SHARED_BRANCH_COMPONENTS=new Set(["PendingData","PolicyRecommendation"]);
const modeBranchComponents=(src) => {
  const names=new Set(["mockDataVisible","demo","mock","isDemo","isMock",
    ...[...src.matchAll(/(?:const|let)\s+(\w+)\s*=\s*!?\s*mockDataVisible\b/g)].map((m) => m[1]),
    ...[...src.matchAll(/mockDataVisible\s*:\s*(\w+)/g)].map((m) => m[1])]);
  return [...names].flatMap((name) => [...src.matchAll(new RegExp(String.raw`\b${name}\s*(?:\?|&&|\?[^:?]{0,40}:)\s*\(?\s*<([A-Z]\w*)`,"g"))]
    .map((m) => m[1]).filter((component) => !SHARED_BRANCH_COMPONENTS.has(component)));
};
assert.deepEqual(modeBranchComponents(page),[]);
assert.deepEqual(modeBranchComponents("const demo = mockDataVisible; return demo ? <MockPage /> : <RealFlow />;"),["MockPage"]);
assert.deepEqual(modeBranchComponents("const { mockDataVisible: isReal } = s; return isReal ? null : <RealFlow />;"),["RealFlow"]);
assert.deepEqual(modeBranchComponents("const mock = mockDataVisible; return <>{mock && <MockPanel />}</>;"),["MockPanel"]);
// 결정 26: ①은 설정·실행만, 결과 카드는 모두 ②. 카드 제목은 두 모드가 같다(view 가 제목을 바꾸지 않는다).
const read=(...path) => readFileSync(join(here,"..",...path),"utf-8");
const factorStage=read("components","FactorAnalysisStage.jsx"), resultStage=read("components","FactorResultStage.jsx");
for (const title of ["예측을 높이고 낮춘 요인","문제 유형 진단 결과","요인분석 모델","통계적 모델 분석 결과"]) assert.match(resultStage,new RegExp(`<Card title="${title.replace(/[+.]/g,"\\$&")}"`));
assert.doesNotMatch(factorStage,/<Card|usePager|TablePager|view\./); // ①에 결과 카드·목록 없음(결정 26)
assert.doesNotMatch(factorStage+resultStage,/XAI · 딥러닝|상관관계 분석|양의 상관|음의 상관/); // 실제 모델(릿지 회귀)·기여도와 맞지 않는 제목 제거
for (const src of [factorStage,resultStage]) assert.doesNotMatch(src,/title=\{view|view\.cards\[index\]\.title|corrTitle/);
for (const src of [page,factorStage,resultStage]) assert.doesNotMatch(src,/입력 지표 기여도 분석|"기여 방향"|"검증 결과"|title: "예측 모델"/);
// 같은 컨트롤: 단계 칩·조건 select·실행 버튼은 두 모드 공용(실데이터는 실행만 aria-disabled + 사유 title).
assert.match(page,/steps=\{ANALYSIS_STEPS\}/);
assert.match(page,/conditions=\{conditions\}/);
assert.match(page,/runLockTitle=\{factorLockTitle\}/);
assert.match(page,/const factorLockTitle = factorModel || factorRetry || autoReady ? null/); // 자동 대상: 지역에 활성 요인분석 모델이 하나라도 있으면 실행 가능
assert.match(factorStage,/aria-disabled=\{runLockTitle \? true : undefined\}/);
// 실데이터 조회 실패면 [요인분석 실행]이 잠기지 않고 재시도(목록·입력 다시 읽기)가 된다 — title "다시 확인".
assert.match(page,/const factorRetry = factor\.loadFailed && !factorModel;/);
assert.match(page,/if \(factorRetry\) \{ factor\.reload\(\); return; \}/);
assert.match(page,/runTitle=\{factorRetry \? "다시 확인" : null\}/);
assert.match(factorStage,/title=\{runLockTitle \?\? runTitle \?\? undefined\}/);
// ② 요인 기여도: 전체 요인(상위 10 + 나머지)을 요인 범위로 거른 뒤 +/− 구역에 이름·기여도 값, 구역마다 5개 먼저·[모두 보기]로 전체.
assert.doesNotMatch(factorStage,/계산 근거에서 확인|<details/);
assert.match(resultStage,/scopeFactors\(\[\.\.\.view\.factors, \.\.\.view\.rest\], scope\)/);
assert.match(resultStage,/const ZONE_LIMIT = 5;[\s\S]*signedAmount\(r\.value, view\.unit\)[\s\S]*모두 보기/);
assert.match(resultStage,/rows\.length === 0 \? <PendingData text=\{FACTOR_SCOPE_EMPTY_TEXT\} \/>/);
// ① 조건 줄: 요인 범위·사람이 읽는 모델 이름 — 두 모드 같음. 학습 조건 연결은 오케스트레이터에 설정이 없어 두지 않는다.
assert.match(page,/aria-label="요인 범위"[\s\S]*FACTOR_SCOPES\.map/);
assert.doesNotMatch(page,/학습 조건 바꾸기|tab-mlops-orch/);
// ① 실행 단계 이름은 실제 모델(릿지)과 맞게, ② 기여도 카드의 기준 줄·검증 카드의 기준선 이름 괄호는 화면에 두지 않는다.
assert.deepEqual(ANALYSIS_STEPS.map((s)=>s.label),["연계 데이터 수집","모델 추론","XAI 기여도 분해","통계 분석 융합"]);
assert.doesNotMatch(resultStage,/view\.basis|pl-source-note|validation\.baselineName/);
// 화면에 특정 알고리즘·모델 이름을 쓰지 않는다(②·③, 사용자 지시 11-17) — JSX 글자·안내 문구 기준(주석 제외).
{
  const visible = (src) => src.replace(/\/\*[\s\S]*?\*\/|\/\/.*$/gm, "");
  const compare = readFileSync(join(here, "..", "components", "ForecastCompare.jsx"), "utf-8");
  assert.doesNotMatch(visible(resultStage + compare) + REMOTE_FORECAST_GUIDE_TEXT, /릿지|Ridge|Chronos|딥러닝|인구학적/);
}
assert.match(page,/instanceLabel\(row, regionNameOf\)/);
assert.doesNotMatch(page,/\{row\.name\} ·/); // 내부 등록 이름(phaseb_… 등)을 모델 선택 칸에 보이지 않는다
// 데모 진단은 사례 서사(출처 mock), 실데이터 진단은 요인분석 결과(출처 api)로 표기한다.
assert.match(page,/diagnosisSource=\{cardSource\}/);assert.match(resultStage,/dataSource=\{diagnosisSource\}/);
// 단위별 반올림(공용 포맷) — 카드 수치는 formatUnitValue 로만, 활성 버전은 짧은 ID + title 전체 ID.
assert.doesNotMatch(factorStage,/formatValue/);
assert.match(resultStage,/const versionTitle = view\?\.model\.versionId;/);assert.equal(resultStage.match(/title=\{versionTitle\}/g)?.length,1);
assert.equal(factorStage.match(/title=\{versionTitle\}/g)?.length,1);assert.match(page,/versionTitle=\{factorView\?\.model\.versionId\}/);
// 활성 버전 변경(추론 409)은 다시 실행 안내로 처리하며 자동 학습으로 우회하지 않는다.
const hookSource=read("hooks","useFactorAnalysis.js");
assert.match(hookSource,/phase === "predict" && error\?\.status === 409/);
assert.doesNotMatch(hookSource,/shortId|slice\(0, 12\)/);
// ③ 계산 근거는 검증 오차·기준선·버전·데이터셋만 — 기여도 3행은 ① 카드로 옮겼다. "요인" 표현도 쓰지 않는다.
const evidence=read("components","ModelEvidence.jsx");
assert.doesNotMatch(evidence,/예측 입력 기여도|예측을 높인 입력|예측을 낮춘 입력|contributionSplit/);assert.doesNotMatch(evidence,/요인/);
assert.match(evidence,/typeName\(p\.model_type\)/);
// ①은 두 모드 같은 렌더 — 데모 사례 서사(region.case)를 ①에서 그리지 않고, 데모 타이머 진행이 남지 않는다.
assert.doesNotMatch(factorStage,/region\.case|c\.factorAnalysis|aiModel|analysisTargets|dataSources/);
assert.doesNotMatch(page,/ANALYSIS_STEP_MS|setAnalysisStep|DEMO_ANALYSIS_STEPS/);
assert.equal(page.match(/useFactorAnalysis\(/g)?.length,1);
assert.doesNotMatch(page,/useFactorAnalysis\(\{[^}]*enabled/);
// 선택한 기존 활성 버전만 읽기 추론한다. 등록·학습·활성화 자동 실행은 금지한다.
const factorHook=read("hooks","useFactorAnalysis.js");
assert.doesNotMatch(factorHook,/ensureDemoFactorModel|\/activate|\/pipelines|available-datasets/);
assert.match(factorHook,/selectedFactorModel\(catalog, modelId, region, target, mode\)/);
assert.doesNotMatch(factorHook,/set(?:Feedback|Missing)\([^)]*(?:error|err)\b/);
// ② 기여도 카드는 같은 요인분석 view 와 ① 요인 범위를 받는다.
assert.match(page,/view=\{factorView\}\s*scope=\{factorScope\}/);
// 실데이터 진단은 제약 문구 대신 ① 결과에서 읽은 신호 — 화면에 "등록 후 제공" 같은 문구가 없다.
assert.match(page,/diagnosis=\{diagnosisView\(factorView, autoRanking, demo \? currentRegion\.case\.problemDiagnosis : \[\], diagHead\)\}/); // 두 모드 같은 진단 함수·구조(데모 머리 칸은 공표 통계 전년 같은 분기 대비)
assert.doesNotMatch(page+resultStage+read("lib","realSimulation.js"),/등록 후 제공|diagnosisNote/);
// ④ 추천 전 안내는 실데이터에서 "관측 입력", 데모 시설 추천에서 "시설 변수"
assert.match(page,/\{demo \? "시설 변수를" : "관측 입력을"\}/);
// ── 요인분석 매핑(두 모드 같은 함수) ──
const panelProof={instance_id:"fa",model_type:"factor_analysis",version_id:"demo_factor_version_123456",job_id:"j",dataset_id:"demo_factor_ds",dataset_hash:"b".repeat(64),
  region_id:"namwon",target:DEMO_FACTOR_TARGET,dong_code:"all",prediction_period:202412,observed_from:201501,observed_to:202412,features:FACTOR_PANEL_FEATURES,
  evaluation_period:{from:202401,to:202412,n:12},metrics:{mae:1224},baseline:{name:"same_month_last_year",mae:6020},mode:"mock",source:"mock"};
const panelValues=[-3040,-5080,-503,-5203,-7650,2616,1432,1750,4098];
const panelResult={provenance:panelProof,intercept:230000,prediction:230000+panelValues.reduce((a,b)=>a+b,0),
  contributions:FACTOR_PANEL_FEATURES.map((feature,i)=>({feature,input:i+1,value:panelValues[i]}))};
const panelLabel=(feature) => observedFeatureLabel(feature,DEMO_FACTOR_TARGET);
assert.equal(new Set(FACTOR_PANEL_FEATURES.map(panelLabel)).size,9);
for (const feature of FACTOR_PANEL_FEATURES) assert.notEqual(panelLabel(feature),feature,feature);
assert.deepEqual(["youth_ratio","aging_index","facility_capacity"].map(panelLabel),["유소년 비율","노령화지수","시설 수용력(인구 대비)"]);
assert.equal(targetUnit(DEMO_FACTOR_TARGET),"명");assert.equal(targetUnit("population_change_pct"),"%");assert.equal(formatUnitValue(1224.215,targetUnit("living_population")),"1,224");assert.equal(formatUnitValue(0.1234,targetUnit("population_change_pct")),"0.12");assert.equal(featureUnit("edu_facility_density",DEMO_FACTOR_TARGET),"개소/만 명");
assert.equal(featureUnit("y_lag1","observed_sales_krw"),"원");assert.equal(featureUnit("dong_1","observed_sales_krw"),"");
assert.equal(featureUnit("aging_index",DEMO_FACTOR_TARGET),"지수");
for (const feature of FACTOR_PANEL_FEATURES) assert.ok(featureUnit(feature,DEMO_FACTOR_TARGET),feature);
// 단위별 반올림: 명·원·개소(명/월 포함)는 정수 + 천 단위 구분, 비율·지수·밀도는 소수 1~2자리, 단위 없음은 최대 2자리.
assert.equal(formatUnitValue(-7650.421,"명"),"-7,650");assert.equal(formatUnitValue(228393.436,"원"),"228,393");
assert.equal(formatUnitValue(-0.3,"명"),"0");assert.equal(formatUnitValue(81.6,"명/월"),"82");assert.equal(formatUnitValue(12,"개소"),"12");
assert.equal(formatUnitValue(8.934,"%"),"8.93");assert.equal(formatUnitValue(352,"지수"),"352.0");assert.equal(formatUnitValue(3.256,"개소/만 명"),"3.26");
assert.equal(formatUnitValue(1,""),"1");assert.equal(formatUnitValue(0.5123,""),"0.51");assert.equal(formatUnitValue(NaN,"명"),"—");
const demoView=factorAnalysisView(panelResult,{demo:true,dataset:{name:"남원 요인 패널"},label:panelLabel});
assert.deepEqual(demoView.factors.map((row)=>row.label).slice(0,3),["사망말소","고령인구","노령화지수"]);
assert.equal(demoView.factors.length,9);assert.equal(demoView.more,0);assert.equal(demoView.unit,"명");
assert.deepEqual([demoView.split.positive.length,demoView.split.negative.length],[4,5]);
assert.deepEqual([...demoView.split.positive,...demoView.split.negative].map((r)=>r.feature).sort(),[...FACTOR_PANEL_FEATURES].sort());
assert.equal(demoView.basis,undefined);assert.equal(demoView.validation.baselineName,undefined);
assert.equal(demoView.model.version,"…n_123456");assert.equal(demoView.model.versionId,panelProof.version_id);
assert.match(demoView.runInfo,/활성 버전 …n_123456$/);assert.deepEqual(demoView.rest,[]);
assert.match(demoView.validation.targets.join("|"),/생활인구 · 요인 9개\|관측 기간 2015-01~2024-12\|검증 기간 2024-01~2024-12/);
const realView=factorAnalysisView({...panelResult,provenance:{...panelProof,mode:"real",source:"real",target:"observed_sales_krw"}},{demo:false,dataset:{name:"남원 관측 소비 스냅샷"},label:panelLabel});
assert.match(realView.runInfo,/^학습데이터 남원 관측 소비 스냅샷 · /);
assert.equal(realView.unit,"원");
// 요인 목록 쪽 번호 기준: 같은 활성 버전이라도 행정동이 바뀌면 다른 키(→ 1쪽부터).
const dongView=factorAnalysisView({...panelResult,provenance:{...panelProof,dong_code:"45190250"}},{demo:false,dataset:null,label:panelLabel});
assert.equal(demoView.listKey,`${panelProof.version_id}|all`);assert.notEqual(dongView.listKey,demoView.listKey);
// 문제 유형 진단(실데이터 ②) — 추정 대 전년 같은 달로 유형, 행정동 제외 최대 하방·상방 요인, 계절 비중.
const salesLabel=(feature) => observedFeatureLabel(feature,"observed_sales_krw",[{code:"1",name:"운봉읍"}]);
const diagOf=(contributions,prediction,ranking=null,items=[],head={}) => diagnosisView(factorAnalysisView({provenance:{...proof,metrics:{},baseline:{}},intercept:0,prediction,contributions},{demo:false,dataset:null,label:salesLabel}),ranking,items,head);
const salesRows=[{feature:"dong_1",input:0,value:-90},{feature:"y_yoy",input:100,value:40},{feature:"month_cos",input:0.5,value:-30},{feature:"has_yoy",input:1,value:0},{feature:"y_lag1",input:80,value:-20}];
// 머리: 유형·변화율(전년 같은 달 대비), 행: 하방·상방 요인(행정동 제외, 부호 붙은 기여도), 계절 비중 15% 이상이면 계절 행.
assert.deepEqual(diagOf(salesRows,90),{kind:"감소형",change:"−10.0%",basis:"전년 같은 달 대비",items:[],rows:[
  {dir:"down",name:"하방 요인",label:"월별 계절성 2",value:"−30원"},{dir:"up",name:"상방 요인",label:"전년 같은 달 관측 소비",value:"+40원"},
  {dir:"season",name:"계절 영향",label:"월별 계절성",value:"17%"}],indicators:[]});
assert.deepEqual([diagOf(salesRows,103).kind,diagOf(salesRows,103).change],["정체형","+3.0%"]);
// 전년 관측이 없으면 유형을 단정하지 않는다(요인 행만), 결과가 없으면 null.
const noYoy=diagOf(salesRows.map((r)=>r.feature==="has_yoy"?{...r,input:0}:r),90);
assert.deepEqual([noYoy.kind,noYoy.change,noYoy.rows.length],[null,"전년 비교 불가",3]); // 머리 자리는 그대로, 사실 표시
// 점검 항목: 넘긴 목록 그대로(데모 사례 정의), 지표별 이름은 데모 대상(생활인구)도 같은 함수로.
assert.deepEqual(diagOf(salesRows,90,[{target:DEMO_FACTOR_TARGET,change:null}],["가","나"]).items,["가","나"]);
assert.deepEqual(diagOf(salesRows,90,[{target:DEMO_FACTOR_TARGET,change:null}]).indicators,[{target:DEMO_FACTOR_TARGET,name:"생활인구",kind:null,change:"전년 비교 불가"}]);
assert.equal(diagnosisView(null),null);
// 지표별 상태(자동 실행 순위 순) — 비교 불가는 유형 없이 "전년 비교 불가".
assert.deepEqual(diagOf(salesRows,90,[{target:"observed_sales_krw",change:-8.24},{target:"nonlocal_visitors",change:null}]).indicators,[
  {target:"observed_sales_krw",name:"관측 소비",kind:"감소형",change:"−8.2%"},{target:"nonlocal_visitors",name:"타지역 방문객",kind:null,change:"전년 비교 불가"}]);
for (const text of JSON.stringify(diagOf(salesRows,90)).split('"')) { assert.doesNotMatch(text,INTERNAL,text); assert.doesNotMatch(text,CAUSAL,text); }
const many={...panelResult,contributions:Array.from({length:14},(_,i)=>({feature:`dong_${i}`,input:1,value:i-7}))};
const manyView=factorAnalysisView(many,{demo:false,dataset:null,label:(f)=>f});
assert.equal(manyView.factors.length,FACTOR_LIST_LIMIT);assert.equal(manyView.more,4);
assert.equal(manyView.rest.length,4);assert.deepEqual([...manyView.factors,...manyView.rest].map((r)=>r.feature).sort(),many.contributions.map((r)=>r.feature).sort());
assert.match(manyView.runInfo,/demo_factor_ds/);
for (const text of [realView.runInfo,demoView.runInfo,...demoView.validation.targets]) assert.doesNotMatch(text,INTERNAL,text);
// 데모 카탈로그(mock)에서 요인분석 활성 모델을 고른다 — 모드가 다르면 거부, 기본은 실데이터.
const mockCatalog={mode:"mock",source:"mock",instances:[{id:"m",mode:"mock",region_id:"namwon",target:DEMO_FACTOR_TARGET,model_type:"factor_analysis",active_version:"v"}]};
assert.equal(activeModelPair(mockCatalog,"namwon",DEMO_FACTOR_TARGET,"mock").factor.id,"m");
assert.throws(() => activeModelPair(mockCatalog,"namwon",DEMO_FACTOR_TARGET));
assert.throws(() => activeModelPair({mode:"real",source:"real",instances:[]},"namwon",DEMO_FACTOR_TARGET,"mock"));
// 알림(role=alert) 문구는 고정 문구만 — 훅이 오류 원문을 화면 상태에 넣지 않는다(원문은 log 로만).
const hook=read("hooks","useRealSimulation.js");
assert.doesNotMatch(hook,/set(?:Feedback|LoadFailed)\([^)]*(?:error|err)\b/);
assert.doesNotMatch(hook,/runAnalysis|factor_analysis/);
// ③·④ 수치도 ①과 같은 단위 규칙 — 원·명은 정수 천 단위(소수 3자리 원문 금지), 데모 ③·④도 같은 함수를 지난다.
const fracView=sensitivityRecommendation([{feature:"y_lag1",from:410000000.289,to:410600000.6,delta:600000.311}],label,"원","관측 소비");
assert.equal(fracView.items[0].scoreText,"+600,000");assert.deepEqual(fracView.items[0].cells,["410,000,000 → 410,600,001원","600,000원"]);
assert.equal(formatUnitValue(430000000.194,"원"),"430,000,000");assert.equal(formatUnitValue(1234.6,"점"),"1,235");
assert.doesNotMatch(read("lib","realSimulation.js"),/formatValue|maximumFractionDigits: 3/);
const forecastCard=read("components","RealForecastResult.jsx"),recoPanel=read("components","PolicyRecommendation.jsx");
assert.match(forecastCard,/const value = \(number\) => formatUnitValue\(number, unit\);/);assert.equal(forecastCard.match(/\{value\(/g)?.length,6);
assert.equal(evidence.match(/formatUnitValue\([^)]*, unit\)/g)?.length,3);
// 데모 추천은 시설 × 활용 방안 — RICE 구성 열·점수를 패널·도출 로그에 두지 않는다.
assert.doesNotMatch(recoPanel,/s\.reach|RICE|Impact|Confidence|Effort/);assert.match(recoPanel,/s\.scoreText \?\? formatUnitValue\(s\.score, scoreUnit\)/);
assert.match(page,/모델 예측상 변화 \+\$\{formatUnitValue\(ranking\[0\]\.delta, unit\)\}/);assert.doesNotMatch(page,/rankStrategies|policyStrategies|"점"\)/);
assert.equal(page.match(/formatUnitValue\((?:finalPop|baseFinalPop|policyDifference), "명"\)/g)?.length,4);
for (const text of [forecastCard,evidence,recoPanel,page]) { assert.doesNotMatch(text,/formatValue|maximumFractionDigits: 3/); }
for (const text of [forecastCard,recoPanel]) assert.doesNotMatch(text,/\.toLocaleString\(\)/);
// 슬라이더 현재값 — 데모 정수 칸 표시는 그대로, 실데이터 원은 정수(소수 3자리 원문 금지), 1 미만 칸만 단위 규칙.
assert.deepEqual([sliderValueText(50,"%",1),sliderValueText(1000,"억",50),sliderValueText(18,"㏊",1),sliderValueText(300,"호",50),sliderValueText(60,"명",5)],["50","1,000","18","300","60"]);
assert.equal(sliderValueText(410000000.289,"원",4000000),"410,000,000");assert.equal(sliderValueText(410000000.289,"원",0.5),"410,000,000");
assert.equal(sliderValueText(8.934,"%",0.01),"8.93");assert.equal(sliderValueText(NaN,"원",1),"—");
const dock=read("components","SimulatorDock.jsx");
assert.doesNotMatch(dock,/maximumFractionDigits: 3/);assert.match(dock,/const shown = sliderValueText\(value, unit, step\);/);assert.equal(dock.match(/\{rangeText\((?:min|max|step)\)\}/g)?.length,3);
console.log("realSimulation: 활성 시뮬레이션 선택·추론 합계·근거 전달·관측 민감도 순위·③④ 독립·①② 제목 동일·안내/알림 금지어·모드 분기 없음·요인분석 두 모드 매핑·③ 근거 기여도 제거 PASS");

// 활성 모델의 이름 대신 ID·지역·대상·모드·정확 버전 소유권을 검증한다.
const chosen={id:"chosen",mode:"real",model_type:"factor_analysis",region_id:"namwon",target:"observed_sales_krw",active_version:"active"};
const selectionCatalog={mode:"real",source:"real",instances:[chosen],versions:[{id:"active",instance_id:"chosen",mode:"real"}]};
assert.equal(selectedFactorModel(selectionCatalog,"chosen","namwon","observed_sales_krw","real"),chosen);
for (const [id,region,target,mode] of [["other","namwon","observed_sales_krw","real"],["chosen","shinan","observed_sales_krw","real"],["chosen","namwon","nonlocal_visitors","real"],["chosen","namwon","observed_sales_krw","mock"]]) {
  assert.equal(selectedFactorModel(selectionCatalog,id,region,target,mode),undefined);
}
assert.equal(selectedFactorModel({...selectionCatalog,versions:[{id:"active",instance_id:"other",mode:"real"}]},"chosen","namwon","observed_sales_krw","real"),undefined);
assert.equal(selectedFactorModel({...selectionCatalog,versions:[]},"chosen","namwon","observed_sales_krw","real"),undefined);
// min/step 격자에 맞추려고 관측 기본값을 변형하면 기준 예측이 달라진다.
const exact=realSliderControls({controls:[{feature:"y_lag1",min:250000000,max:650000000,default:400000000.5}],inputs:{}},label,"원")[0];
assert.equal(exact.default,400000000.5);
assert.equal(exact.exactValue,true);
assert.equal(exact.step,4000000);

// 실제 화면/추론 훅이 공유하는 경계: 이전 모드·선택·버전의 결과는 노출하거나 다시 계산에 쓰지 않는다.
const expectedContext={mode:"real",region:"namwon",target:"observed_sales_krw",dong:"45190250",modelId:"factor",versionId:"v1"};
assert.equal(inferenceMatches({provenance:proof},expectedContext),true);
for(const changed of [{mode:"mock"},{region:"shinan"},{target:"nonlocal_visitors"},{modelId:"other"},{versionId:"v2"},{dong:"other"}]) {
  assert.equal(inferenceMatches({provenance:proof},{...expectedContext,...changed}),false);
}
assert.equal(inferenceMatches({provenance:{...proof,source:"mock"}},expectedContext),false);
assert.equal(inferenceMatches(null,expectedContext),false);
const recommendationContext={mode:"real",region:"namwon",target:"observed_sales_krw",dong:"45190250",factorModel:chosen,simulationProof:sim.provenance};
const savedRecommendationKey=inferenceContextKey(recommendationContext);
assert.equal(inferenceContextKey({...recommendationContext,factorModel:{...chosen},simulationProof:{...sim.provenance}}),savedRecommendationKey);
for(const changed of [{mode:"mock"},{region:"shinan"},{target:"nonlocal_visitors"},{dong:"other"},
  {factorModel:{...chosen,id:"new-factor"}},{factorModel:{...chosen,active_version:"new-factor-version"}},
  {simulationProof:{...sim.provenance,instance_id:"new-sim"}},{simulationProof:{...sim.provenance,version_id:"new-sim-version"}},
  {simulationProof:{...sim.provenance,dataset_hash:"c".repeat(64)}}]) {
  assert.notEqual(inferenceContextKey({...recommendationContext,...changed}),savedRecommendationKey);
}
// 늦은 응답은 gate가 취소하고, 렌더 전환 시에도 같은 provenance 경계가 다른 선택의 결과를 배제한다.
let releaseOld;
const late=new Promise((resolve)=>{releaseOld=resolve;});
const lateGate=createRequestGate(()=>1);
const oldRequest=lateGate.begin();
const oldResponse=late.then((payload)=>oldRequest()&&inferenceMatches(payload,{...expectedContext,versionId:"new-active"}));
lateGate.cancel();
releaseOld({provenance:proof});
assert.equal(await oldResponse,false);
// U20: OFF 시나리오 표 — 열 수·순서 그대로, "–" 대신 다음 달 예측·기준 대비 변화율(③ 카드와 같은 계산), 머리글은 모드별 이름
const scenarioSrc=read("components","ScenarioCompare.jsx");
assert.doesNotMatch(scenarioSrc,/popCell|: "–"\)/);
assert.match(scenarioSrc,/\{allowSeed \? "10년 후 인구" : "다음 달 예측"\}[\s\S]*\{allowSeed \? "증감률" : "기준 대비 변화율"\}/);
assert.equal(scenarioSrc.match(/<th scope="col"/g)?.length,6);
assert.match(scenarioSrc,/result\.delta \/ Math\.abs\(result\.baseline_prediction\)/);
assert.match(page,/predict: \(values\) => real\.predictOnly\(/);
assert.match(read("hooks","useRealSimulation.js"),/const predictOnly = useCallback/);
// U21: 요인분석 모델이 없는 지역의 빈 상태에 [모델이 있는 지역으로 이동] — 진입 때 자동 이동 없음(모델 선택과 같은 moveToModel)
assert.match(page,/guide\(FACTOR_GUIDE_TEXT, moveAction\)/);
assert.match(page,/onClick=\{\(\) => moveToModel\(movableModel\)\}[\s\S]*모델이 있는 지역으로 이동/);
assert.equal(page.match(/moveToModel\(/g)?.length,2); // 호출은 모델 선택 1 + 이동 버튼 1뿐(진입 시 자동 호출 없음)
// 결정 18: 자동(가장 나빠진 지표) — 순위·지표별 한 줄. 전년 같은 달 값이 없으면 맨 뒤·"전년 비교 불가".
const entry = (target, change) => ({ target, change });
assert.deepEqual(rankTargets([entry("nonlocal_visitors", 1), entry("observed_sales_krw", -8.2)]).map((e) => e.target), ["observed_sales_krw", "nonlocal_visitors"]);
assert.deepEqual(rankTargets([entry("observed_sales_krw", null), entry("nonlocal_visitors", 3)]).map((e) => e.target), ["nonlocal_visitors", "observed_sales_krw"]);
assert.deepEqual(rankTargets([entry("observed_sales_krw", null), entry("nonlocal_visitors", null)]).map((e) => e.target), ["observed_sales_krw", "nonlocal_visitors"]); // 둘 다 불가 → 입력 순서
assert.equal(AUTO_TARGET_LABEL, "자동(가장 나빠진 지표)");
// 변화율은 ② 진단의 유형 계산과 같다(다음 달 추정 대 전년 같은 달).
const yoyView = (yoy, prediction, has = 1) => factorAnalysisView({ provenance: { ...proof, metrics: {}, baseline: {} }, intercept: 0, prediction,
  contributions: [{ feature: "y_yoy", input: yoy, value: 1 }, { feature: "has_yoy", input: has, value: 0 }] }, { demo: false, dataset: null, label: salesLabel });
assert.equal(yoyChange(yoyView(100, 90)), -10);
assert.equal(yoyChange(yoyView(100, 90, 0)), null); // 전년 같은 달 값 없음
assert.equal(yoyChange(yoyView(0, 90)), null);
// 페이지: 학습 대상 첫 항목 자동·기본값, 실행은 자동이면 runAuto
assert.match(page, /useState\(true\)/);
assert.match(page, /factor\.runAuto\(\{ onPick/);
// 두 모드 같은 학습 대상 선택지 체계(첫 항목 자동, 기본값 자동)·같은 ③ 표지.
assert.match(page, /<option value=\{AUTO_TARGET\}>\{AUTO_TARGET_LABEL\}<\/option>\s*\{demo \? <option value=\{DEMO_FACTOR_TARGET\}>생활인구<\/option>/);
assert.match(page, /status=\{analysisDone \? "② 진단 반영" : null\}/);
assert.doesNotMatch(page + resultStage, /showDerive/);
// ③ 설정 띠 다섯 칸 — 두 모드 같은 형식, 기간은 "2019년 4월", 숫자는 천 단위·단위, 조절변수는 반복 이름 없는 칩.
assert.equal(monthText(201904), "2019년 4월"); assert.equal(monthText(null), "—");
const demoFrame = simFrameView({ demo: true, sim: { objective: "목적제 기반 강화학습 — 정주여건 만족도 극대화", factors: { demand: "주거·정주 수요량", supply: "임대주택·정주여건 시설 공급" }, constraint: "예산" },
  budgetTotal: 1200, controls: [{ label: "임대주택 공급 호수" }, { label: "운영 인력" }] });
assert.deepEqual(demoFrame, {
  objective: { value: "정주여건 만족도 극대화", meta: null }, demand: { value: "주거·정주 수요량", meta: null },
  supply: { value: "임대주택·정주여건 시설 공급", meta: null },
  adjust: { chips: ["복지 예산", "산업 일자리", "주거·주택", "임대주택 공급 호수", "운영 인력"], meta: null },
  constraint: { value: "1,200억 원", meta: "총 예산" } });
const realFrame = simFrameView({ demo: false, target: "observed_sales_krw", dongName: "운봉읍",
  controls: ["y_lag1", "y_lag2", "y_yoy"].map((f) => ({ label: observedFeatureLabel(f, "observed_sales_krw") })),
  context: { inputs: { y_lag1: 400000000 }, provenance: { prediction_period: 202401, range_provenance: { from: 201904, to: 202309 } } } });
assert.deepEqual(realFrame, {
  objective: { value: "관측 소비 증가", meta: "예측월 2024년 1월" }, demand: { value: "400,000,000원", meta: "전월 관측 소비 · 운봉읍" },
  supply: { value: "해당 없음", meta: null }, adjust: { chips: ["전월", "2개월 전", "전년 같은 달"], meta: "관측 소비" },
  constraint: { value: "2019년 4월~2023년 9월", meta: "입력 조정 범위" } });
// 모델 준비 전에는 다섯 칸 모두 같은 자리에 "–"(칩 없음).
assert.deepEqual(Object.values(simFrameView({ demo: false, target: "observed_sales_krw" })).map((c) => c.value ?? c.chips.length), ["–", "–", "–", 0, "–"]);
for (const text of JSON.stringify([demoFrame, realFrame]).split('"')) { assert.doesNotMatch(text, INTERNAL, text); assert.doesNotMatch(text, /\(|강화학습|STAGE/, text); }
assert.match(page, /simFrameView\(\{ demo, sim, budgetTotal, controls, target, context: real\.ready \? real\.context : null, dongName \}\)/);
// 결정 25: 데모 ② 머리 칸 = 생활인구 공표 통계 최근 분기 대 전년 같은 분기(생활인구 = 등록 + 체류).
const q = (year, quarter, registered, stay) => ({ year, quarter, registered, stay });
const shinanQ = [q(2025, 1, 3.9, 19), q(2025, 2, 3.9, 20.2), q(2025, 3, 4, 21), q(2026, 1, 4.1, 17), q(2026, 2, 4.2, 18.2)];
assert.equal(quarterYoyChange(shinanQ).toFixed(1), "-7.1"); // 22.4 대 24.1
assert.equal(quarterYoyChange([q(2025, 4, 1, 9), q(2024, 4, 1, 9.3)]).toFixed(1), "-2.9"); // 정체형 범위
assert.equal(quarterYoyChange([q(2025, 4, 1, 11), q(2024, 4, 1, 9)]).toFixed(1), "20.0"); // 증가
assert.equal(quarterYoyChange([q(2026, 2, 1, 1), q(2025, 3, 1, 1)]), null); // 전년 같은 분기 없음
assert.equal(quarterYoyChange(null), null);
const kindOf = (pct) => diagOf(salesRows, 90, null, [], { change: pct, basis: "전년 같은 분기 대비" }).kind;
assert.deepEqual([kindOf(quarterYoyChange(shinanQ)), kindOf(-3), kindOf(20)], ["감소형", "정체형", "증가형"]);
const demoHead = diagOf(salesRows, 90, [{ target: DEMO_FACTOR_TARGET, change: quarterYoyChange(shinanQ) }], [], { change: quarterYoyChange(shinanQ), basis: "전년 같은 분기 대비" });
assert.deepEqual([demoHead.kind, demoHead.change, demoHead.basis, demoHead.indicators[0].kind, demoHead.indicators[0].change], ["감소형", "−7.1%", "전년 같은 분기 대비", "감소형", "−7.1%"]);
assert.deepEqual([diagOf(salesRows, 90, null, [], { change: null, basis: "전년 같은 분기 대비" }).change, diagOf(salesRows, 90).basis], ["전년 비교 불가", "전년 같은 달 대비"]);
// 결정 26: 요인 범위 — "정책 관련 요인만"은 계절성·과거 관측값(y_*)·전년 유무·행정동 보정을 뺀다(순서 유지). 전체는 그대로.
assert.deepEqual(FACTOR_SCOPES.map((x) => x.label), ["전체", "정책 관련 요인만"]);
const scopeRows = ["y_lag1", "month_sin", "aging_index", "dong_45", "has_yoy", "month_cos", "y_yoy", "edu_facility_density"].map((feature) => ({ feature }));
assert.deepEqual(scopeFactors(scopeRows, "policy").map((r) => r.feature), ["aging_index", "edu_facility_density"]);
assert.equal(scopeFactors(scopeRows, "all"), scopeRows);
assert.deepEqual(scopeFactors(scopeRows.filter((r) => isNonPolicyFeature(r.feature)), "policy"), []); // 실데이터 입력만이면 빈 목록 → 같은 자리 안내
assert.deepEqual(FACTOR_PANEL_FEATURES.filter(isNonPolicyFeature), []); // 데모 요인 패널은 모두 정책 관련
assert.deepEqual([signedAmount(-24000, "원"), signedAmount(3973, "명")], ["−24,000원", "+3,973명"]);
// 모델 선택 칸 이름: 내부 등록 이름이 아니라 종류 · 지역 · 학습 대상.
assert.equal(instanceLabel({ name: "phaseb_factor_namwon_x", model_type: "factor_analysis", region_id: "namwon", target: "observed_sales_krw" }, () => "전북특별자치도 남원시"),
  "요인분석 · 전북특별자치도 남원시 · 관측 소비");
// ③ 릿지·Chronos-2 비교 줄(결정 30) — 합성 값. 선택과 다른 응답은 버리고, 원격 예측이 비정상·없음이면 행동 안내 자리.
{
  const pick = { mode: "real", modelId: "sim", versionId: "v2", dong: "45190250" };
  const ok = { mode: "real", source: "real", provenance: { ...sim.provenance, prediction_period: 202401 }, ridge: { prediction: 123456789 },
    remote: { status: "ok", low: 110000000, median: 120000000, high: 135000000 } };
  assert.deepEqual(forecastCompareView(ok, pick, "원"), { period: "관측 소비 2024-01", ridge: "123,456,789원", remote: { median: "120,000,000원", range: "1.1억~1.35억원" } });
  assert.equal(forecastCompareView({ ...ok, remote: { status: "unavailable" } }, pick, "원").remote, null);
  assert.equal(forecastCompareView({ ...ok, remote: { ...ok.remote, low: 130000000 } }, pick, "원").remote, null); // 10% > 50%
  assert.equal(forecastCompareView({ ...ok, remote: { ...ok.remote, high: NaN } }, pick, "원").remote, null);
  for (const stale of [{ ...pick, versionId: "v1" }, { ...pick, modelId: "other" }, { ...pick, dong: "45190310" }, { ...pick, mode: "mock" }]) {
    assert.equal(forecastCompareView(ok, stale, "원"), null);
  }
  assert.equal(forecastCompareView(null, pick, "원"), null);
  const demo = { mode: "mock", source: "mock", provenance: { ...sim.provenance, mode: "mock", source: "mock", target: "living_population", dong_code: "all", prediction_period: 202412 },
    ridge: { prediction: 1234.4 }, remote: { status: "unavailable" } };
  assert.deepEqual(forecastCompareView(demo, { mode: "mock", modelId: "sim", versionId: "v2", dong: "" }, "명"), { period: "생활인구 2024-12", ridge: "1,234명", remote: null });
  // 두 모드 같은 자리(③ 예측 카드 안, 계산 근거 위)에 같은 컴포넌트. 안내는 행동만(출처·방법·괄호 꼬리표 없음).
  assert.match(page, /<ForecastCompare view=\{compareView\} loading=\{compare\.loading\} \/>\s*\{\/\* 계산 근거/);
  assert.match(page, /const compareModel = demo \? factorModel : real\.ready \?/);
  const compareCard = read("components", "ForecastCompare.jsx");
  assert.match(compareCard, /활성 모델 예측[\s\S]*시계열 모델 예측[\s\S]*범위 \{view\.remote\.range\}[\s\S]*REMOTE_FORECAST_GUIDE_TEXT/);
  assert.doesNotMatch(REMOTE_FORECAST_GUIDE_TEXT, /[()（）]/);
  assert.doesNotMatch(REMOTE_FORECAST_GUIDE_TEXT + compareCard, /출처|DGX|amazon|192\.168/);
}
console.log("PASS: current mode/model/version/district provenance, recommendation identity and late response isolation, U20 시나리오 예측 칸·U21 이동 버튼");
