import { useState, useMemo } from "react";
import Card from "../components/Card.jsx";
import { useAppState } from "../context/AppStateContext.jsx";

function buildPreview(region, template, driftInjected) {
  const psiLine = driftInjected ? "0.384 (드리프트 위험 발생)" : "0.045 (매우 안정)";
  const tenYearPop = Math.round(region.population * 0.81).toLocaleString();

  return (
    <>
      <h2>{template.title}</h2>
      <p style={{ textAlign: "right", fontSize: 12, color: "#4b5563" }}>
        보고서 번호: RD-POP-2026-{region.id.toUpperCase()}
      </p>
      <p style={{ textAlign: "right", fontSize: 12, color: "#4b5563" }}>
        발생 일시: 2026년 05월 23일 13:00
      </p>

      <h3>1. 대상 지자체 기본 현황 및 예측 요약</h3>
      <p>
        본 분석서의 대상인 <strong>{region.name}</strong>은 현재 등록 인구수{" "}
        <strong>{region.population.toLocaleString()}명</strong>, 평균 가중 출산율{" "}
        <strong>{region.birthRate}명</strong>으로 고령화 지수가 <strong>{region.agingIndex}%</strong>에
        달해 인구소멸 위험지수 <strong>{region.riskIndex}</strong> 등급의 극심한 소멸 위험 지역입니다.
        AI 예측 모델에 따르면, 현행 유지 시 10년 후 인구는 약 <strong>{tenYearPop}명</strong>{" "}
        수준으로 급감할 것으로 예측됩니다.
      </p>

      <h3>2. MLOps 인공지능 모델 검증지표</h3>
      <ul>
        <li>글로벌 협업 모델 Accuracy: 0.892 (SOTA 기준)</li>
        <li>연합 사일로 데이터 최근 10시간 MSE 오차: 0.041 만족</li>
        <li>입력 데이터 분산 안정성 (PSI): {psiLine}</li>
      </ul>

      <h3>3. SHAP 중요 기여 특성에 따른 최적 맞춤 대책</h3>
      <p>
        인구소멸을 지연시키는 데 가장 큰 양의 기여를 하는 인자는{" "}
        <strong>청년층 복지 재정 (+0.354)</strong> 및{" "}
        <strong>제조업 일자리 수 (+0.281)</strong>이며, 평균 연령 (-0.152)의 증가는 인구 감소를
        가속화하는 핵심 위험 요인으로 파악되었습니다. 따라서 본 지자체는 청년 유입을 극대화할 수 있는
        다음과 같은 특화 예산 배정을 제안합니다.
      </p>

      <h3>4. 제언 및 최종 권고 요약</h3>
      <ol>
        <li>welfare 예산 부문에 청년 보조 자금 배정 가중치를 최소 60% 이상으로 확대 편성.</li>
        <li>industry 부문의 산업단지 유치를 유도하여 청년 근로자의 유입 세제 혜택 가속화.</li>
        <li>정량적 성과 모니터링 강화를 위해 3차년도 MLOps 대시보드 실시간 API 연계 체계 가동.</li>
      </ol>

      <div
        style={{
          marginTop: 30,
          borderTop: "1px solid #d1d5db",
          paddingTop: 20,
          fontSize: 12,
          color: "#6b7280",
          textAlign: "center"
        }}
      >
        국토인구소멸대응 공동 R&D 플랫폼 데이터 연계 승인필
      </div>
    </>
  );
}

function buildMarkdown(region, template, driftInjected) {
  const psiLine = driftInjected ? "0.384 (드리프트 감지)" : "0.045 (정상)";
  const outlierLine = driftInjected ? "3건" : "0건";
  const tenYearBase = Math.round(region.population * 0.81).toLocaleString();
  const tenYearPolicy = Math.round(region.population * 0.95).toLocaleString();

  return `# ${template.title}

보고서 번호: RD-POP-2026-${region.id.toUpperCase()}
발생 일시: 2026년 05월 23일 13:00

## 1. 분석 개요 및 대상 지자체 기본 현황
본 R&D 보고서는 국토 소멸 대응 프로젝트의 3차년도 연동 모델 결과에 기반하여 지자체 맞춤형 복지 예산과 인구 이동간의 상관관계를 다각 분석한 권고안입니다.

* 대상 지자체: ${region.name}
* 인구수: ${region.population.toLocaleString()}명
* 출산율: ${region.birthRate}명
* 고령화지수: ${region.agingIndex}%
* 인구소멸 위험지수: ${region.riskIndex} (소멸 위험등급 경보)

## 2. MLOps AI 모델 검증지표
* 예측 모델 정확도 (Accuracy): 0.892
* 데이터 분산 안정성 (PSI): ${psiLine}
* 검출된 이상치 (Outliers): ${outlierLine}

## 3. SHAP 특징 중요도 기여 요인 분석
* 1순위 기여: 청년 복지 예산 가중치 (SHAP: +0.354)
* 2순위 기여: 제조업 공장 일자리 유치 (SHAP: +0.281)
* 3순위 위험 요인: 평균 연령 증가 (SHAP: -0.152)

## 4. 정책 효과 시뮬레이션 예측
* 현 정책 유지 시 10년 후 인구수: ${tenYearBase}명
* 추천 맞춤 정책 (복지 가중치 집중) 적용 시 10년 후 인구수: ${tenYearPolicy}명

## 5. 지자체 정책 실무 요약 권고 사항
1. 청년 영유아 복지 예산 가중치를 확대하여 정주 요건 조기 개선.
2. 메타데이터 API 규격 스키마를 통해 실시간 데이터 수집 및 drift 감지 모듈의 조기 구축.

---
© 2026 국토인구소멸대응 공동 R&D 통합 플랫폼 R-Center. All rights reserved.
`;
}

export default function ReporterPage() {
  const { appData, driftInjected, addConsoleLog } = useAppState();
  const templates = appData.report_templates;
  const regions = appData.regions;

  const [templateId, setTemplateId] = useState(templates[0].id);
  const [regionId, setRegionId] = useState(regions[0].id);
  const [preview, setPreview] = useState(null);

  const region = useMemo(() => regions.find((r) => r.id === regionId) ?? regions[0], [
    regions,
    regionId
  ]);
  const template = useMemo(
    () => templates.find((t) => t.id === templateId) ?? templates[0],
    [templates, templateId]
  );

  const handleGenerate = () => {
    setPreview(buildPreview(region, template, driftInjected));
    addConsoleLog(`INFO: 보고서 미리보기 생성 성공 - ${region.name}`);
  };

  const handleDownload = () => {
    const markdown = buildMarkdown(region, template, driftInjected);
    const blob = new Blob([markdown], { type: "text/markdown;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", `R_D_인구소멸대응보고서_${region.name.split(" ")[1]}.md`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    addConsoleLog(`INFO: 보고서 다운로드 성공 - R_D_인구소멸대응보고서_${region.name.split(" ")[1]}.md`);
  };

  return (
    <div className="grid-details-split">
      <Card title="인구감소 사회문제해결 리포트 생성기" icon="fa-file-invoice">
        <p style={{ fontSize: 13, color: "var(--text-secondary)", marginBottom: 20 }}>
          지자체 실무자 보고 양식 규격에 맞춰, MLOps 모니터링 성능값, 데이터 드리프트 위험 분석 및
          정책 가이드를 조합한 공식 R&D 보고서를 즉시 출력합니다.
        </p>

        <div className="slider-container">
          <label htmlFor="reporter-template">보고서 표준 템플릿 선택</label>
          <select
            id="reporter-template"
            className="select-control"
            value={templateId}
            onChange={(e) => setTemplateId(e.target.value)}
          >
            {templates.map((t) => (
              <option key={t.id} value={t.id}>
                {t.title}
              </option>
            ))}
          </select>
        </div>

        <div className="slider-container">
          <label htmlFor="reporter-region">대상 지자체 선택</label>
          <select
            id="reporter-region"
            className="select-control"
            value={regionId}
            onChange={(e) => setRegionId(e.target.value)}
          >
            {regions.map((r) => (
              <option key={r.id} value={r.id}>
                {r.name}
              </option>
            ))}
          </select>
        </div>

        <div
          className="slider-container"
          style={{ display: "flex", gap: 12, alignItems: "center", marginTop: 24 }}
        >
          <button className="btn btn-secondary" style={{ flexGrow: 1 }} onClick={handleGenerate}>
            <i className="fa-solid fa-pen-nib"></i> 보고서 실시간 본문 생성
          </button>
          <button className="btn btn-primary" style={{ flexGrow: 1 }} onClick={handleDownload}>
            <i className="fa-solid fa-file-arrow-down"></i> HWP/Word 다운로드
          </button>
        </div>

        <div
          className="card"
          style={{
            marginTop: 20,
            backgroundColor: "rgba(59, 130, 246, 0.05)",
            borderColor: "rgba(59, 130, 246, 0.15)"
          }}
        >
          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>
            <i className="fa-solid fa-circle-info"></i> 지자체 실무자 팁
          </div>
          <div style={{ fontSize: 11, color: "var(--text-secondary)", lineHeight: 1.5 }}>
            생성된 문서 파일은 국가 표준 공문서 규격(HWP/docx)에 바로 붙여넣기 하실 수 있도록 마크다운
            포뜻팅 및 정량 성과 지표 텍스트가 모두 기입되어 생성됩니다.
          </div>
        </div>
      </Card>

      <Card title="보고서 가상 미리보기 (A4 레이아웃)" icon="fa-eye">
        <div className="report-preview-panel">
          {preview ?? (
            <>
              <h2>인구감소 대응 R&D 분석 리포트 요약서</h2>
              <p
                style={{
                  textAlign: "center",
                  color: "#4b5563",
                  fontSize: 12,
                  marginBottom: 30
                }}
              >
                지자체를 선택하시고 [보고서 실시간 본문 생성] 버튼을 누르시면 실시간 메타데이터가
                적용되어 채워집니다.
              </p>
            </>
          )}
        </div>
      </Card>
    </div>
  );
}
