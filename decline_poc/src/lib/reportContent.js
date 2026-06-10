// 보고서 콘텐츠 빌더 — 지자체/템플릿/드리프트 상태로부터 포맷 중립 블록 모델과
// Excel 표·Markdown 문자열을 생성한다. 템플릿 종류에 따라 본문 구성을 분기.

const SHAP_TOP = [
  { rank: "1순위 기여", factor: "청년 복지 예산 가중치", shap: "+0.354" },
  { rank: "2순위 기여", factor: "제조업 공장 일자리 유치", shap: "+0.281" },
  { rank: "3순위 위험", factor: "평균 연령 증가", shap: "-0.152" }
];

function psiText(driftInjected) {
  return driftInjected ? "0.384 (드리프트 위험 감지)" : "0.045 (안정)";
}

// 정책 품의·기안 공통양식 머리말 (제목·기안자·일시·결재).
function approvalHeader(region) {
  return [
    { type: "paragraph", text: `보고서 번호: RD-POP-2026-${region.id.toUpperCase()}` },
    { type: "paragraph", text: "기안자: 인구정책담당관   |   일시: 2026-05-23 13:00" },
    { type: "paragraph", text: "결재: 담당 ___  팀장 ___  과장 ___  국장 ___" }
  ];
}

function populationTrendBlock(populationChange) {
  if (!populationChange) return [];
  const { years, birth_rate, death_rate, net_migration } = populationChange;
  const last = years.length - 1;
  return [
    { type: "heading", level: 2, text: "인구 변화율 추이 (출생율·사망률·인구 유출입)" },
    {
      type: "list",
      items: [
        `출생율: ${birth_rate[0]} → ${birth_rate[last]} (${years[0]}→${years[last]})`,
        `사망률: ${death_rate[0]} → ${death_rate[last]} (지속 상승)`,
        `순이동(전입-전출): ${net_migration[last].toLocaleString()}명 (순유출 확대)`
      ]
    }
  ];
}

/* ---------------- 템플릿별 본문 ---------------- */

function taskOrderBlocks(region) {
  return [
    { type: "heading", level: 2, text: "1. 과업 개요" },
    {
      type: "list",
      items: [
        `과업명: ${region.name} 인구감소 대응 정책 분석`,
        "과업목적: 데이터 기반 맞춤형 인구정책 수립 지원",
        "과업기간: 2026-06-01 ~ 2026-11-30 (6개월)",
        `과업범위: ${region.name} 행정동 전역 생활인구·정주인구`,
        "과업비: 금 일억오천만원정 (₩150,000,000)"
      ]
    },
    { type: "heading", level: 2, text: "2. 세부 내용" },
    {
      type: "list",
      items: [
        `분석 대상지: ${region.name} (위험지수 ${region.riskIndex})`,
        "분석 내용: 인구 유출 요인 분석, 정책 효과 시뮬레이션",
        "요구 데이터: 주민등록·복지재정·산업·공간정보 데이터 소스",
        "정책 시사점: 청년 유입·생활인구 활성화 전략 도출"
      ]
    },
    { type: "heading", level: 2, text: "3. 과업 수행 지침" },
    {
      type: "list",
      items: [
        "보고 사항: 착수·중간·최종 보고 (월간 진척 보고 포함)",
        "성과 기준: 모델 정확도 ≥ 0.85, 시각화 응답속도 ≤ 2초",
        "성과 납품목록: 분석보고서, SW 등록, 기술문서",
        "기타 준수: 개인정보 비식별 처리, 보안 규정 준수"
      ]
    }
  ];
}

function vitalPopulationBlocks(region, vital) {
  if (!vital) return [{ type: "paragraph", text: "생활인구 데이터가 없습니다." }];
  const ages = Object.entries(vital.age_group)
    .map(([k, v]) => `${k} ${v}%`)
    .join(", ");
  return [
    { type: "heading", level: 2, text: "1. 생활인구 구성" },
    {
      type: "list",
      items: [
        `성별: 남 ${vital.gender.male}% / 여 ${vital.gender.female}%`,
        `연령: ${ages}`,
        `외국인 비율: ${vital.foreigner_ratio}%`
      ]
    },
    { type: "heading", level: 2, text: "2. 방문/숙박 패턴" },
    {
      type: "list",
      items: [
        `당일 방문 ${vital.visit.day}% / 숙박 ${vital.visit.overnight}%`,
        `전년 대비 방문율: ${vital.yoy_visit_rate > 0 ? "+" : ""}${vital.yoy_visit_rate}%`,
        `전월 대비 방문율: ${vital.mom_visit_rate > 0 ? "+" : ""}${vital.mom_visit_rate}%`
      ]
    },
    { type: "heading", level: 2, text: "3. 생활인구 유입 전략 인사이트" },
    {
      type: "list",
      items: [
        "관광·통근·귀농 목적성 유동인구별 맞춤 프로그램 설계",
        "숙박 전환율 제고를 위한 체류형 콘텐츠 강화",
        "외국인 생활인구 정주 지원 및 안전망 확충"
      ]
    }
  ];
}

function analysisBlocks(region, driftInjected, populationChange, live) {
  const tenYearBase = Math.round(region.population * 0.81).toLocaleString();
  const tenYearPolicy = Math.round(region.population * 0.95).toLocaleString();
  // 라이브 바인딩 지표가 있으면 우선 사용(API 자동 갱신 반영), 없으면 정적 기본값.
  const accuracy = live ? live.accuracy : 0.892;
  const outliers = live ? live.outliers : driftInjected ? 3 : 0;
  return [
    { type: "heading", level: 2, text: "1. 분석 개요 및 대상 지자체 기본 현황" },
    {
      type: "paragraph",
      text:
        "본 R&D 보고서는 국토 소멸 대응 프로젝트 3차년도 연동 모델 결과에 기반하여 " +
        "지자체 맞춤형 복지 예산과 인구 이동 간의 상관관계를 다각 분석한 권고안입니다."
    },
    {
      type: "list",
      items: [
        `대상 지자체: ${region.name}`,
        `인구수: ${region.population.toLocaleString()}명`,
        `출산율: ${region.birthRate}명`,
        `고령화지수: ${region.agingIndex}%`,
        `인구소멸 위험지수: ${region.riskIndex} (소멸 위험등급 경보)`
      ]
    },
    ...populationTrendBlock(populationChange),
    { type: "heading", level: 2, text: "2. MLOps AI 모델 검증지표" },
    {
      type: "list",
      items: [
        `예측 모델 정확도 (Accuracy): ${accuracy}`,
        `데이터 분산 안정성 (PSI): ${psiText(driftInjected)}`,
        `검출된 이상치 (Outliers): ${outliers}건`
      ]
    },
    { type: "heading", level: 2, text: "3. SHAP 특징 중요도 기여 요인 분석" },
    { type: "list", items: SHAP_TOP.map((s) => `${s.rank}: ${s.factor} (SHAP ${s.shap})`) },
    { type: "heading", level: 2, text: "4. 정책 효과 시뮬레이션 예측" },
    {
      type: "list",
      items: [
        `현 정책 유지 시 10년 후 인구수: ${tenYearBase}명`,
        `추천 맞춤 정책(복지 가중치 집중) 적용 시 10년 후 인구수: ${tenYearPolicy}명`
      ]
    },
    { type: "heading", level: 2, text: "5. 지자체 정책 실무 요약 권고 사항" },
    {
      type: "list",
      items: [
        "청년 영유아 복지 예산 가중치를 확대하여 정주 요건 조기 개선.",
        "메타데이터 API 규격 스키마를 통한 실시간 수집 및 drift 감지 모듈 조기 구축."
      ]
    }
  ];
}

// 사례(스마트팜·정주여건) 본문 — region.case의 요인분석·진단·시뮬레이션을 보고서로 구조화.
function caseBlocks(region) {
  const c = region.case;
  if (!c) return [{ type: "paragraph", text: "사례 데이터가 없습니다." }];
  return [
    { type: "heading", level: 2, text: `1. 현안(${region.theme}) 및 사업 개요` },
    { type: "paragraph", text: c.summary },
    { type: "list", items: c.dataSources },
    { type: "heading", level: 2, text: "2. AI·인구학적 요인분석 (XAI: SHAP)" },
    {
      type: "paragraph",
      text: `딥러닝 예측 모델: ${c.aiModel.deepLearning.join(", ")} · 기준선: ${c.aiModel.statBaseline}`
    },
    { type: "list", items: c.correlations.positive.map((t) => `(+) ${t}`) },
    { type: "list", items: c.correlations.negative.map((t) => `(−) ${t}`) },
    { type: "heading", level: 2, text: "3. 문제 유형 진단 결과" },
    { type: "list", items: c.problemDiagnosis },
    { type: "heading", level: 2, text: "4. 강화학습 시뮬레이션 기반 추진계획" },
    {
      type: "list",
      items: [
        `목적함수: ${c.simulation.objective}`,
        `수요량(x): ${c.simulation.factors.demand}`,
        `공급량(y): ${c.simulation.factors.supply}`,
        `조정 변수: ${c.simulation.factors.adjust}`,
        `제한 요소: ${c.simulation.constraint}`
      ]
    },
    { type: "paragraph", text: `최종 산출: ${c.reportFocus} (AI 정책 보고서)` }
  ];
}

/** 포맷 중립 블록 모델 (docx/hwp/markdown 공통 소스). */
export function buildReportBlocks(region, template, driftInjected, extra = {}) {
  const header = approvalHeader(region);
  if (template.id === "template_task_order") {
    return [...header, ...taskOrderBlocks(region)];
  }
  if (template.id === "template_vital_population") {
    return [...header, ...vitalPopulationBlocks(region, extra.vitalPopulation)];
  }
  if (template.id === "template_smartfarm" || template.id === "template_settlement") {
    return [...header, ...caseBlocks(region)];
  }
  return [...header, ...analysisBlocks(region, driftInjected, extra.populationChange, extra.live)];
}

/** Excel(.xlsx) 용 2차원 표 — 지표 요약. */
export function buildReportRows(region, template, driftInjected, extra = {}) {
  const rows = [
    ["인구감소 대응 R&D 지표 요약", template.title],
    ["보고서 번호", `RD-POP-2026-${region.id.toUpperCase()}`],
    ["기안자", "인구정책담당관"],
    ["일시", "2026-05-23 13:00"],
    [],
    ["구분", "지표", "값"],
    ["기본 현황", "대상 지자체", region.name],
    ["기본 현황", "인구수(명)", region.population],
    ["기본 현황", "출산율(명)", region.birthRate],
    ["기본 현황", "고령화지수(%)", region.agingIndex],
    ["기본 현황", "인구소멸 위험지수", region.riskIndex],
    ["모델 검증", "Accuracy", extra.live ? extra.live.accuracy : 0.892],
    ["모델 검증", "PSI", driftInjected ? 0.384 : 0.045],
    ["모델 검증", "이상치 검출(건)", extra.live ? extra.live.outliers : driftInjected ? 3 : 0],
    ["SHAP 기여", "청년 복지 예산", 0.354],
    ["SHAP 기여", "제조업 일자리", 0.281],
    ["SHAP 위험", "평균 연령 증가", -0.152],
    ["시뮬레이션", "10년 후 인구(현행)", Math.round(region.population * 0.81)],
    ["시뮬레이션", "10년 후 인구(정책적용)", Math.round(region.population * 0.95)]
  ];

  const pc = extra.populationChange;
  if (pc) {
    rows.push([]);
    rows.push(["인구변화 추이", "연도", "출생율", "사망률", "순이동"]);
    pc.years.forEach((y, i) => {
      rows.push(["", y, pc.birth_rate[i], pc.death_rate[i], pc.net_migration[i]]);
    });
  }
  return rows;
}

/** Markdown 문자열 — 블록 모델에서 파생. */
export function blocksToMarkdown(title, blocks) {
  const lines = [`# ${title}`, ""];
  blocks.forEach((block) => {
    if (block.type === "heading") {
      lines.push("", `## ${block.text}`);
    } else if (block.type === "list") {
      (block.items || []).forEach((item) => lines.push(`* ${item}`));
    } else {
      lines.push(block.text || "");
    }
  });
  lines.push("", "---", "© 2026 국토인구소멸대응 공동 R&D 통합 플랫폼 R-Center.");
  return lines.join("\n");
}
