import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { activeModelPair, inferenceRequest, validateInference, rankSensitivity, observedFeatureLabel, createRequestGate,
  contributionFactors, contributionSplit, realSliderControls, forecastSeries, sensitivityRecommendation, regionRealStatus,
  REAL_GUIDE_TEXT, FACTOR_GUIDE_TEXT, FACTOR_PENDING_TEXT, FACTOR_RUN_LOCK_TEXT, REAL_LOAD_FAILED_TEXT, REAL_PREDICT_FAILED_TEXT, REAL_VERSION_CHANGED_TEXT,
  FACTOR_IDLE_TEXT, FACTOR_FAILED_TEXT, FACTOR_VERSION_CHANGED_TEXT, FACTOR_DIAGNOSIS_TEXT, FACTOR_EXPLAIN_TEXT, FACTOR_FOOTNOTE, CORRELATION_BASIS_TEXT,
  DEMO_FACTOR_TARGET, FACTOR_PANEL_FEATURES, FACTOR_LIST_LIMIT, factorAnalysisView, featureUnit, targetUnit, formatUnitValue } from "./realSimulation.js";
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
assert.match(view.footnote,/인과 효과를 검증한 결과가 아닙니다/);
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
  FACTOR_IDLE_TEXT,FACTOR_FAILED_TEXT,FACTOR_VERSION_CHANGED_TEXT,FACTOR_DIAGNOSIS_TEXT,FACTOR_EXPLAIN_TEXT,FACTOR_FOOTNOTE,CORRELATION_BASIS_TEXT];
// 인과로 과장하는 표현 금지 — 각주의 "인과 효과가 아닙니다"만 허용한다.
const CAUSAL=/효과가 있|때문에|유발|원인|덕분/;
for (const text of guideTexts) { assert.doesNotMatch(text,INTERNAL,text); assert.doesNotMatch(text,CAUSAL,text); }
assert.match(FACTOR_FOOTNOTE,/인과 효과가 아닙니다/);
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
// ①·② 카드 제목은 두 모드가 같다(기준선 제목) — view 가 제목을 바꾸지 않고, 실데이터 전용 제목이 남지 않는다.
const read=(...path) => readFileSync(join(here,"..",...path),"utf-8");
const factorStage=read("components","FactorAnalysisStage.jsx"), resultStage=read("components","FactorResultStage.jsx");
for (const title of ["AI + 인구학적 요인 분석 결과","XAI · 딥러닝 예측 모델","통계적 모델 분석 결과"]) assert.match(factorStage,new RegExp(`card\\(\\d, "${title.replace(/[+.]/g,"\\$&")}"`));
for (const title of ["상관관계 분석","문제 유형 진단 결과"]) assert.match(resultStage,new RegExp(`<Card title="${title}"`));
for (const src of [factorStage,resultStage]) assert.doesNotMatch(src,/title=\{view|view\.cards\[index\]\.title|corrTitle/);
for (const src of [page,factorStage,resultStage]) assert.doesNotMatch(src,/입력 지표 기여도 분석|"기여 방향"|"검증 결과"|title: "예측 모델"/);
// 같은 컨트롤: 단계 칩·조건 select·실행 버튼은 두 모드 공용(실데이터는 실행만 aria-disabled + 사유 title).
assert.match(page,/steps=\{ANALYSIS_STEPS\}/);
assert.match(page,/conditions=\{conditions\}/);
assert.match(page,/runLockTitle=\{factorLockTitle\}/);
assert.match(page,/const factorLockTitle = demo \|\| factorModel \|\| factorRetry \? null/);
assert.match(factorStage,/aria-disabled=\{runLockTitle \? true : undefined\}/);
// 실데이터 조회 실패면 [요인분석 실행]이 잠기지 않고 재시도(목록·입력 다시 읽기)가 된다 — title "다시 확인".
assert.match(page,/const factorRetry = !demo && real\.loadFailed && !factorModel;/);
assert.match(page,/if \(factorRetry\) \{ real\.reload\(\); return; \}/);
assert.match(page,/runTitle=\{factorRetry \? "다시 확인" : null\}/);
assert.match(factorStage,/title=\{runLockTitle \?\? runTitle \?\? undefined\}/);
// ①-1 10행 밖 요인은 같은 카드 안 펼침(접힘 기본)으로 — 삭제된 계산 근거 행을 가리키지 않는다.
assert.doesNotMatch(factorStage,/계산 근거에서 확인/);
assert.match(factorStage,/<details className="pl-stat-baseline">[\s\S]*외 \{view\.more\}개 요인 모두 보기[\s\S]*view\.rest\.map\(factorRow\)/);
// 단위별 반올림(공용 포맷) — 카드 수치는 formatUnitValue 로만, 활성 버전은 짧은 ID + title 전체 ID.
assert.doesNotMatch(factorStage,/formatValue/);
assert.match(factorStage,/const versionTitle = view\?\.model\.versionId;/);assert.equal(factorStage.match(/title=\{versionTitle\}/g)?.length,2);
// 데모 준비 중 학습 요청 409(다른 화면이 먼저 시작) → 진행 중 작업을 이어 폴링한다.
const hookSource=read("hooks","useFactorAnalysis.js");
assert.match(hookSource,/if \(error\?\.status !== 409\) throw error;\s*job = runningJob\(await at\(apiGet\(`\$\{BASE\}\/catalog`/);
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
// 훅의 모드 차이는 데모 자동 준비 한 곳뿐이다.
const factorHook=read("hooks","useFactorAnalysis.js");
assert.equal(factorHook.match(/await ensureDemoFactorModel\(/g)?.length,1);
assert.match(factorHook,/if \(!model && demo\) \{/);
assert.doesNotMatch(factorHook,/set(?:Feedback|Missing)\([^)]*(?:error|err)\b/);
// ② 상관관계는 ① 기여도 부호, 진단은 데모 서사/실데이터 안내.
assert.match(page,/split=\{factorView\?\.split\}/);
assert.match(page,/diagnosisNote=\{demo \? null : FACTOR_DIAGNOSIS_TEXT\}/);
// ④ 추천 전 안내는 실데이터에서 "관측 입력"
assert.match(page,/\{demo \? "정책 변수를" : "관측 입력을"\}/);
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
assert.equal(targetUnit(DEMO_FACTOR_TARGET),"명");assert.equal(featureUnit("edu_facility_density",DEMO_FACTOR_TARGET),"개소/만 명");
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
assert.equal(demoView.sourceLabel,"데모");assert.match(demoView.basis,/^연계 데이터: 데모 요인 패널 · 남원 요인 패널$/);
assert.equal(demoView.validation.baselineName,"전년 같은 달 관측값");assert.equal(demoView.model.version,"…123456");assert.equal(demoView.model.versionId,panelProof.version_id);
assert.match(demoView.runInfo,/활성 버전 …123456$/);assert.deepEqual(demoView.rest,[]);
assert.match(demoView.validation.targets.join("|"),/생활인구 · 요인 9개\|관측 기간 2015-01~2024-12\|검증 기간 2024-01~2024-12/);
const realView=factorAnalysisView({...panelResult,provenance:{...panelProof,mode:"real",source:"real",target:"observed_sales_krw"}},{demo:false,dataset:{name:"남원 관측 소비 스냅샷"},label:panelLabel});
assert.equal(realView.basis,"현재 등록 데이터 기준 · 남원 관측 소비 스냅샷 (관측 2015-01~2024-12)");
assert.equal(realView.sourceLabel,"실데이터 · 남원 관측 소비 스냅샷");assert.equal(realView.unit,"원");
const many={...panelResult,contributions:Array.from({length:14},(_,i)=>({feature:`dong_${i}`,input:1,value:i-7}))};
const manyView=factorAnalysisView(many,{demo:false,dataset:null,label:(f)=>f});
assert.equal(manyView.factors.length,FACTOR_LIST_LIMIT);assert.equal(manyView.more,4);
assert.equal(manyView.rest.length,4);assert.deepEqual([...manyView.factors,...manyView.rest].map((r)=>r.feature).sort(),many.contributions.map((r)=>r.feature).sort());
assert.match(manyView.basis,/demo_factor_ds/);
for (const text of [demoView.basis,realView.basis,demoView.runInfo,...demoView.validation.targets]) assert.doesNotMatch(text,INTERNAL,text);
// 데모 카탈로그(mock)에서 요인분석 활성 모델을 고른다 — 모드가 다르면 거부, 기본은 실데이터.
const mockCatalog={mode:"mock",source:"mock",instances:[{id:"m",mode:"mock",region_id:"namwon",target:DEMO_FACTOR_TARGET,model_type:"factor_analysis",active_version:"v"}]};
assert.equal(activeModelPair(mockCatalog,"namwon",DEMO_FACTOR_TARGET,"mock").factor.id,"m");
assert.throws(() => activeModelPair(mockCatalog,"namwon",DEMO_FACTOR_TARGET));
assert.throws(() => activeModelPair({mode:"real",source:"real",instances:[]},"namwon",DEMO_FACTOR_TARGET,"mock"));
// 알림(role=alert) 문구는 고정 문구만 — 훅이 오류 원문을 화면 상태에 넣지 않는다(원문은 log 로만).
const hook=read("hooks","useRealSimulation.js");
assert.doesNotMatch(hook,/set(?:Feedback|LoadFailed)\([^)]*(?:error|err)\b/);
assert.doesNotMatch(hook,/runAnalysis|factor_analysis/);
console.log("realSimulation: 활성 시뮬레이션 선택·추론 합계·근거 전달·관측 민감도 순위·③④ 독립·①② 제목 동일·안내/알림 금지어·모드 분기 없음·요인분석 두 모드 매핑·③ 근거 기여도 제거 PASS");
