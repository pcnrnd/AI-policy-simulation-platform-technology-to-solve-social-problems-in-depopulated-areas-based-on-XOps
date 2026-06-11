import { useState, useMemo, useEffect, useCallback } from "react";
import Card from "../components/Card.jsx";
import PerfBadge from "../components/PerfBadge.jsx";
import { useAppState } from "../context/AppStateContext.jsx";
import {
  buildReportBlocks,
  buildReportRows,
  blocksToMarkdown
} from "../lib/reportContent.js";
import {
  buildDocx,
  buildXlsx,
  buildHwpHtml,
  downloadBlob
} from "../lib/reportExport.js";
import { fetchReportData } from "../lib/dataopsApi.js";
import { measureAsync } from "../lib/perf.js";

const EXPORT_FORMATS = [
  { id: "docx", label: "Word (.docx)", icon: "fa-file-word", ext: "docx" },
  { id: "xlsx", label: "Excel (.xlsx)", icon: "fa-file-excel", ext: "xlsx" },
  { id: "hwp", label: "한글 (.hwp)", icon: "fa-file-lines", ext: "hwp" },
  { id: "md", label: "Markdown (.md)", icon: "fa-markdown", ext: "md" }
];

function regionShortName(region) {
  const parts = region.name.split(" ");
  return parts[parts.length - 1] || region.name;
}

function buildPreview(region, template, driftInjected, live) {
  const psiLine = driftInjected ? "0.384 (드리프트 위험 발생)" : "0.045 (매우 안정)";
  const tenYearPop = Math.round(region.population * 0.81).toLocaleString();
  const accuracy = live ? live.indicators.accuracy : 0.892;
  const outliers = live ? live.indicators.outliers : driftInjected ? 3 : 0;

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
        <li>글로벌 협업 모델 Accuracy: {accuracy} (SOTA 기준)</li>
        <li>연합 데이터 소스 최근 10시간 MSE 오차: 0.041 만족</li>
        <li>입력 데이터 분산 안정성 (PSI): {psiLine}</li>
        <li>검출 이상치(Outliers): {outliers}건</li>
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

export default function ReporterPage() {
  const { appData, driftInjected, addConsoleLog } = useAppState();
  const templates = appData.report_templates;
  const regions = appData.regions;

  const [templateId, setTemplateId] = useState(templates[0].id);
  const [regionId, setRegionId] = useState(regions[0].id);
  const [format, setFormat] = useState("docx");
  const [preview, setPreview] = useState(null);

  // Data source API 자동 바인딩 상태 (Notion: 데이터 갱신 부분 자동 업데이트)
  const [binding, setBinding] = useState(null);
  const [bindingMs, setBindingMs] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  const [refreshCount, setRefreshCount] = useState(0);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [usedNames, setUsedNames] = useState(() => new Set());

  const region = useMemo(() => regions.find((r) => r.id === regionId) ?? regions[0], [
    regions,
    regionId
  ]);
  const template = useMemo(
    () => templates.find((t) => t.id === templateId) ?? templates[0],
    [templates, templateId]
  );

  // 양식에 연결된 Data source를 API로 호출해 지표를 자동 갱신.
  const refreshBinding = useCallback(
    async (count) => {
      setRefreshing(true);
      try {
        const { result, ms } = await measureAsync(() =>
          fetchReportData(region, driftInjected, count)
        );
        setBinding(result);
        setBindingMs(ms);
        setLastUpdated(new Date().toLocaleTimeString("ko-KR"));
        addConsoleLog(
          `INFO: 리포트 지표 API 자동 갱신 (${result.source}) - 수집 ${result.collected_rows}행, Accuracy ${result.indicators.accuracy}`
        );
      } catch (err) {
        addConsoleLog(`ERROR: 리포트 데이터 바인딩 실패 - ${err?.message ?? "알 수 없는 오류"}`);
      } finally {
        setRefreshing(false);
      }
    },
    [region, driftInjected, addConsoleLog]
  );

  // 지자체/드리프트 변경 시 자동 재바인딩.
  useEffect(() => {
    refreshBinding(0);
    setRefreshCount(0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [regionId, driftInjected]);

  const handleManualRefresh = () => {
    const next = refreshCount + 1;
    setRefreshCount(next);
    refreshBinding(next);
  };

  const handleGenerate = () => {
    setPreview(buildPreview(region, template, driftInjected, binding));
    addConsoleLog(`INFO: 보고서 미리보기 생성 성공 - ${region.name}`);
  };

  // 동일 파일명 중복 시 _(n) 증분 suffix 부여.
  const dedupeFilename = (baseName, ext) => {
    let candidate = `${baseName}.${ext}`;
    let n = 1;
    while (usedNames.has(candidate)) {
      candidate = `${baseName}_(${n}).${ext}`;
      n += 1;
    }
    setUsedNames((prev) => new Set(prev).add(candidate));
    return candidate;
  };

  const handleDownload = () => {
    const fmt = EXPORT_FORMATS.find((f) => f.id === format) ?? EXPORT_FORMATS[0];
    const baseName = `R_D_인구소멸대응보고서_${regionShortName(region)}`;
    const filename = dedupeFilename(baseName, fmt.ext);
    const extra = {
      populationChange: appData.population_change,
      vitalPopulation: appData.vital_population,
      live: binding ? binding.indicators : null
    };

    try {
      const blocks = buildReportBlocks(region, template, driftInjected, extra);

      let blob;
      if (fmt.id === "docx") {
        blob = buildDocx({ title: template.title, blocks });
      } else if (fmt.id === "xlsx") {
        blob = buildXlsx({
          sheetName: "지표요약",
          rows: buildReportRows(region, template, driftInjected, extra)
        });
      } else if (fmt.id === "hwp") {
        blob = buildHwpHtml({ title: template.title, blocks });
      } else {
        const markdown = blocksToMarkdown(template.title, blocks);
        blob = new Blob([markdown], { type: "text/markdown;charset=utf-8;" });
      }

      downloadBlob(blob, filename);
      addConsoleLog(`INFO: 보고서 다운로드 성공 (${fmt.label}) - ${filename}`);
    } catch (err) {
      addConsoleLog(`ERROR: 보고서 저장 실패 (${fmt.label}) - ${err?.message ?? "알 수 없는 오류"}`);
    }
  };

  return (
    <div className="grid-details-split">
      <Card title="인구감소 사회문제해결 리포트 생성기" icon="fa-file-invoice">
        <p style={{ fontSize: 13, color: "var(--text-secondary)", marginBottom: 20 }}>
          지자체 실무자 보고 양식 규격에 맞춰, MLOps 모니터링 성능값과 데이터 드리프트 위험 분석을
          조합해 정책 제안 · 자원 최적화 · 생활인구 유입 전략 인사이트를 담은 공식 R&D 보고서를 즉시
          출력합니다.
        </p>

        <div
          className="card"
          style={{
            marginBottom: 20,
            backgroundColor: "rgba(16, 185, 129, 0.05)",
            borderColor: "rgba(16, 185, 129, 0.18)"
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: 8
            }}
          >
            <div style={{ fontSize: 12, fontWeight: 600 }}>
              <i className="fa-solid fa-link" style={{ color: "var(--accent-teal)" }}></i>{" "}
              템플릿 가변 저장 구조 — Data source API 자동 바인딩
            </div>
            <PerfBadge ms={bindingMs} label="API 응답" />
          </div>
          <div style={{ fontSize: 11, color: "var(--text-secondary)", lineHeight: 1.6 }}>
            <div>
              엔드포인트:{" "}
              <code style={{ color: "var(--accent-blue)" }}>{binding?.source ?? "연결 중…"}</code>
            </div>
            <div>Adapter: {binding?.adapter ?? "—"}</div>
            <div>
              수집 행 수: {binding ? binding.collected_rows.toLocaleString() : "—"} · 마지막 갱신:{" "}
              {lastUpdated ?? "—"}
            </div>
          </div>
          <button
            className="btn btn-secondary"
            style={{ width: "100%", marginTop: 10 }}
            onClick={handleManualRefresh}
            disabled={refreshing}
          >
            <i className={"fa-solid " + (refreshing ? "fa-spinner fa-spin" : "fa-rotate")}></i>{" "}
            {refreshing ? "데이터 갱신 중…" : "데이터 새로고침"}
          </button>
        </div>

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

        <div className="slider-container" style={{ marginTop: 24 }}>
          <label htmlFor="reporter-format">저장 파일 형식</label>
          <select
            id="reporter-format"
            className="select-control"
            value={format}
            onChange={(e) => setFormat(e.target.value)}
          >
            {EXPORT_FORMATS.map((f) => (
              <option key={f.id} value={f.id}>
                {f.label}
              </option>
            ))}
          </select>
        </div>

        <div
          className="slider-container"
          style={{ display: "flex", gap: 12, alignItems: "center" }}
        >
          <button className="btn btn-secondary" style={{ flexGrow: 1 }} onClick={handleGenerate}>
            <i className="fa-solid fa-pen-nib"></i> 보고서 실시간 본문 생성
          </button>
          <button className="btn btn-primary" style={{ flexGrow: 1 }} onClick={handleDownload}>
            <i className="fa-solid fa-file-arrow-down"></i> 보고서 다운로드
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
            Word(.docx)·Excel(.xlsx)는 OOXML 규격으로, 한글(.hwp)은 한컴오피스 호환 형식으로 즉시
            저장됩니다. 정량 성과 지표 텍스트와 표가 모두 자동 기입됩니다. 동일 파일명 재저장 시{" "}
            <code>_(1)</code> 형식으로 자동 증분되며, 저장 실패 시 콘솔에 오류가 기록됩니다.
            (브라우저 보안 정책상 저장 경로는 브라우저 다운로드 폴더로 고정됩니다.)
          </div>
        </div>
      </Card>

      <Card title="보고서 미리보기 (A4 레이아웃)" icon="fa-eye">
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
