import { useState, useMemo, useEffect, useCallback, useRef } from "react";
import Card from "../components/Card.jsx";
import { NO_DEMO_DATA } from "../components/PendingData.jsx";
import { useAppState } from "../context/AppStateContext.jsx";
import {
  buildReportBlocks,
  buildReportRows,
  blocksToMarkdown,
  formatMetric
} from "../lib/reportContent.js";
import {
  buildDocx,
  buildXlsx,
  buildHwpHtml,
  downloadBlob
} from "../lib/reportExport.js";
import { fetchReportData } from "../lib/dataopsApi.js";

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

function buildPreview(region, template, live) {
  const tenYearPop = Math.round(region.population * 0.81).toLocaleString();
  const indicators = live?.indicators ?? null;
  const explain = live?.explain ?? null;

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

      {indicators && (
        <>
          <h3>2. MLOps 인공지능 모델 검증지표</h3>
          <ul>
            <li>예측 오차 (WAPE): {formatMetric(indicators.wape)}</li>
            <li>
              예측 오차 (MAE): {formatMetric(indicators.mae)} (기준선 MAE{" "}
              {formatMetric(indicators.baselineMae)})
            </li>
            {(indicators.psi ?? []).map((entry) => (
              <li key={entry.label}>
                입력 데이터 분산 안정성 (PSI): {entry.label} {formatMetric(entry.psi, 4)}
              </li>
            ))}
          </ul>
        </>
      )}

      {/* SHAP 기여도는 explain 실호출 결과만 싣는다 — 없으면 섹션 자체를 비운다(사유 문구 없음). */}
      {explain && (
        <>
          <h3>
            3. SHAP 중요 기여 특성 ({explain.baseYm} · {explain.dongName ?? explain.dongCode} 기준)
          </h3>
          <ul>
            {explain.contributions.map((c) => (
              <li key={c.feature}>
                {c.feature}: 기여도 {formatMetric(c.phi, 4)}
              </li>
            ))}
          </ul>
        </>
      )}

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
  const { appData, currentRegion, setCurrentRegion, addConsoleLog, mockDataVisible } = useAppState();
  const templates = appData.report_templates;
  const regions = appData.regions;

  // 데모 표시 토글은 화면 구조가 아니라 **데이터 유무**만 바꾼다. 보고서 본문은 시드 지자체 지표와
  // 로컬 생성 바인딩 값으로 채워지므로, OFF에서는 미리보기·바인딩 값 자리에 사유만 남기고
  // 템플릿·지자체·형식 선택과 갱신 버튼은 그대로 조작 가능하게 둔다.
  const allowSeed = mockDataVisible;

  const [templateId, setTemplateId] = useState(templates[0].id);
  const [regionId, setRegionId] = useState(currentRegion?.id ?? regions[0].id);
  const [format, setFormat] = useState("docx");
  const [preview, setPreview] = useState(null);

  // Data source API 자동 바인딩 상태 (Notion: 데이터 갱신 부분 자동 업데이트)
  const [binding, setBinding] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [usedNames, setUsedNames] = useState(() => new Set());
  const [bindingFeedback, setBindingFeedback] = useState(null);
  const [reportFeedback, setReportFeedback] = useState(null);
  const refreshRequestRef = useRef(0);

  const region = useMemo(() => regions.find((r) => r.id === regionId) ?? regions[0], [
    regions,
    regionId
  ]);
  const template = useMemo(
    () => templates.find((t) => t.id === templateId) ?? templates[0],
    [templates, templateId]
  );

  // 양식에 연결된 실데이터 API(평가·드리프트)를 호출해 지표를 자동 갱신.
  const refreshBinding = useCallback(
    async () => {
      const requestId = ++refreshRequestRef.current;
      setRefreshing(true);
      setPreview(null);
      setReportFeedback(null);
      setBinding(null);
      setLastUpdated(null);
      setBindingFeedback({ tone: "pending", message: `${region.name} 데이터를 갱신하고 있습니다.` });
      try {
        const result = await fetchReportData(region);
        if (requestId !== refreshRequestRef.current) return;
        // 활성 모델·지표가 없으면 null — 사유를 지어내지 않고 빈 상태로 둔다.
        if (!result) {
          setBindingFeedback(null);
          addConsoleLog("INFO: 리포트 지표 갱신 - 표시할 데이터 없음");
          return;
        }
        setBinding(result);
        setLastUpdated(new Date().toLocaleTimeString("ko-KR"));
        setBindingFeedback(
          allowSeed
            ? { tone: "success", message: `${region.name} 데이터를 갱신했습니다.` }
            : { tone: "info", message: NO_DEMO_DATA }
        );
        addConsoleLog(
          allowSeed
            ? `INFO: 리포트 지표 API 자동 갱신 (${result.source}) - WAPE ${result.indicators.wape}`
            : "INFO: 리포트 지표 바인딩 갱신 — 표시할 실데이터 없음"
        );
      } catch (err) {
        if (requestId !== refreshRequestRef.current) return;
        // 조회 실패도 빈 상태로 둔다(화면 사유 문구 없음). 원인은 콘솔 기록으로만 남긴다.
        setBindingFeedback(null);
        addConsoleLog(`ERROR: 리포트 데이터 바인딩 실패 - ${err?.message ?? "알 수 없는 오류"}`);
      } finally {
        if (requestId === refreshRequestRef.current) setRefreshing(false);
      }
    },
    [region, allowSeed, addConsoleLog]
  );

  // 지자체 변경 시 자동 재바인딩(드리프트 토글은 실측 지표에 영향을 주지 않는다).
  useEffect(() => {
    refreshBinding();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [regionId]);

  useEffect(() => {
    if (currentRegion?.id && currentRegion.id !== regionId) setRegionId(currentRegion.id);
  }, [currentRegion, regionId]);

  // 템플릿이 바뀌면 기존 템플릿으로 만든 미리보기를 현재 결과처럼 남기지 않는다.
  useEffect(() => {
    setPreview(null);
    setReportFeedback(null);
  }, [templateId]);

  const handleManualRefresh = () => {
    refreshBinding();
  };

  const handleGenerate = () => {
    // 생성 로직은 그대로 두고, 데모 OFF에서는 시드 본문을 미리보기 자리에 넣지 않는다.
    if (!allowSeed) {
      setPreview(null);
      setReportFeedback({
        tone: "error",
        message: NO_DEMO_DATA
      });
      addConsoleLog(`WARN: 보고서 미리보기 생성 중단 - ${region.name} 실데이터 없음`);
      return;
    }
    setPreview(buildPreview(region, template, binding));
    setReportFeedback({ tone: "success", message: `${region.name} 보고서 미리보기를 생성했습니다.` });
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
    // 미리보기는 데이터 계층에서 막혀 있지만 내보내기 경로에는 가드가 없었다 — 화면은 비었는데
    // 시드로 가득 찬 docx/xlsx/hwp/md 가 생성됐다. 데모 OFF에서는 파일을 만들지 않는다.
    if (!allowSeed) {
      setReportFeedback({
        tone: "info",
        message: `${NO_DEMO_DATA} — 설정에서 데모 데이터 표시를 켜면 생성할 수 있습니다.`
      });
      return;
    }
    const fmt = EXPORT_FORMATS.find((f) => f.id === format) ?? EXPORT_FORMATS[0];
    const baseName = `R_D_인구소멸대응보고서_${regionShortName(region)}`;
    const filename = dedupeFilename(baseName, fmt.ext);
    const extra = {
      populationChange: appData.population_change,
      vitalPopulation: appData.vital_population,
      live: binding ? binding.indicators : null,
      explain: binding ? binding.explain : null
    };

    try {
      const blocks = buildReportBlocks(region, template, extra);

      let blob;
      if (fmt.id === "docx") {
        blob = buildDocx({ title: template.title, blocks });
      } else if (fmt.id === "xlsx") {
        blob = buildXlsx({
          sheetName: "지표요약",
          rows: buildReportRows(region, template, extra)
        });
      } else if (fmt.id === "hwp") {
        blob = buildHwpHtml({ title: template.title, blocks });
      } else {
        const markdown = blocksToMarkdown(template.title, blocks);
        blob = new Blob([markdown], { type: "text/markdown;charset=utf-8;" });
      }

      downloadBlob(blob, filename);
      setReportFeedback({ tone: "success", message: `${filename} 다운로드를 시작했습니다.` });
      addConsoleLog(`INFO: 보고서 다운로드 성공 (${fmt.label}) - ${filename}`);
    } catch (err) {
      setReportFeedback({
        tone: "error",
        message: `보고서를 저장하지 못했습니다. ${err?.message ?? "파일 형식을 확인하고 다시 시도하세요."}`
      });
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
          aria-busy={refreshing}
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
          </div>
          <div className="mock-data-output" style={{ fontSize: 11, color: "var(--text-secondary)", lineHeight: 1.6 }}>
            <div>
              엔드포인트:{" "}
              <code style={{ color: "var(--accent-blue)" }}>
                {allowSeed ? binding?.source ?? "—" : NO_DEMO_DATA}
              </code>
            </div>
            <div>마지막 갱신: {allowSeed ? lastUpdated ?? "—" : "—"}</div>
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
          {bindingFeedback && (
            <div
              className={`async-feedback is-${bindingFeedback.tone}`}
              role={bindingFeedback.tone === "error" ? "alert" : "status"}
              aria-live={bindingFeedback.tone === "error" ? "assertive" : "polite"}
            >
              {bindingFeedback.message}
            </div>
          )}
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
            onChange={(e) => {
              const nextId = e.target.value;
              setRegionId(nextId);
              const nextRegion = regions.find((item) => item.id === nextId);
              if (nextRegion) setCurrentRegion(nextRegion);
            }}
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

        <div className="slider-container reporter-action-row">
          <button className="btn btn-secondary" style={{ flexGrow: 1 }} onClick={handleGenerate} disabled={refreshing || !binding}>
            <i className="fa-solid fa-pen-nib" aria-hidden="true"></i> 보고서 실시간 본문 생성
          </button>
          <button className="btn btn-primary" style={{ flexGrow: 1 }} onClick={handleDownload} disabled={refreshing || !binding}>
            <i className="fa-solid fa-file-arrow-down" aria-hidden="true"></i> 보고서 다운로드
          </button>
        </div>
        {reportFeedback && (
          <div
            className={`async-feedback is-${reportFeedback.tone}`}
            role={reportFeedback.tone === "error" ? "alert" : "status"}
            aria-live={reportFeedback.tone === "error" ? "assertive" : "polite"}
          >
            {reportFeedback.message}
          </div>
        )}

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
        {/* 미리보기 패널은 데모 토글과 무관하게 같은 자리에 남긴다 — 예전엔 CSS로 패널을 통째로
            가려서 "왜 비었는지"를 읽을 수 없었다. OFF에서는 본문 대신 사유만 채운다. */}
        <div className="report-preview-panel">
          {!allowSeed ? (
            <>
              <h2>인구감소 대응 R&D 분석 리포트 요약서</h2>
              <p style={{ textAlign: "center", color: "#4b5563", fontSize: 12, marginBottom: 30 }}>
                {NO_DEMO_DATA}
              </p>
            </>
          ) : (
            preview ?? (
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
            )
          )}
        </div>
      </Card>
    </div>
  );
}
