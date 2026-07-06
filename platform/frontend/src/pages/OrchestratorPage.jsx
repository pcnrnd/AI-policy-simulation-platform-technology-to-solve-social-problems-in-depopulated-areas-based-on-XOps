import { useState } from "react";
import Card from "../components/Card.jsx";
import ConsoleLog from "../components/ConsoleLog.jsx";
import InfoTip from "../components/InfoTip.jsx";
import TablePager, { paginate } from "../components/TablePager.jsx";
import { useAppState } from "../context/AppStateContext.jsx";
import { PIPELINE_NODES } from "../constants/pipeline.js";
import { RETRAIN_PIPELINES, MODEL_REGISTRY } from "../constants/models.js";

const PAGE_SIZE = 5;

const STORE_STATUS_STYLE = {
  운영: { color: "var(--accent-teal)", bg: "rgba(16, 185, 129, 0.1)" },
  이전: { color: "var(--text-muted)", bg: "rgba(148, 163, 184, 0.12)" },
  롤백: { color: "var(--accent-red)", bg: "rgba(239, 68, 68, 0.1)" }
};

function nodeStatus(index, currentStep) {
  // index: 0-based, currentStep: 1-based (0 = not started)
  const stepIdx = currentStep - 1;
  if (currentStep === 0) return "";
  if (index < stepIdx) return "success";
  if (index === stepIdx) return "active";
  return "";
}

function connectorStatus(index, currentStep) {
  const stepIdx = currentStep - 1;
  if (currentStep === 0) return "";
  if (index < stepIdx - 1) return "success";
  if (index === stepIdx - 1) return "success";
  if (index === stepIdx) return "active";
  return "";
}

export default function OrchestratorPage() {
  const {
    pipelineRunning,
    pipelineStep,
    pipelineRun,
    pipelineHistory,
    modelStore,
    consoleLogs,
    startPipeline,
    resetPipeline,
    addConsoleLog
  } = useAppState();

  const modelName = (id) => MODEL_REGISTRY.find((m) => m.id === id)?.name ?? id;

  // 테이블 페이징 (파이프라인 카탈로그 / Model Store)
  const [plPage, setPlPage] = useState(1);
  const [storePage, setStorePage] = useState(1);
  const pl = paginate(RETRAIN_PIPELINES, plPage, PAGE_SIZE);
  const store = paginate(modelStore, storePage, PAGE_SIZE);

  const handleReset = () => {
    resetPipeline();
    addConsoleLog("INFO: MLOps 재학습 파이프라인이 초기화되었습니다.");
  };

  // 카탈로그 [실행] → 실행 시작 + 아래 실행 상태 카드로 이동 (누른 곳에서 결과가 보이도록)
  const handleRun = (plDef) => {
    startPipeline("수동 실행 (파이프라인 카탈로그)", plDef);
    setTimeout(() => {
      document.getElementById("pipeline-status-anchor")?.scrollIntoView({ block: "start" });
    }, 120);
  };

  return (
    <>
      {/* ① 진입점: 등록된 재학습 파이프라인 카탈로그 — 선택·실행 */}
      <Card
        title="등록된 재학습 파이프라인"
        icon="fa-list-check"
        style={{ marginBottom: 24 }}
        headerRight={
          <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
            {RETRAIN_PIPELINES.length}건 등록 · 모델 레지스트리 연동 · 드리프트 감지 시 자동 실행
          </span>
        }
      >
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>파이프라인</th>
                <th>대상 모델</th>
                <th>트리거 조건</th>
                <th>마지막 실행</th>
                <th>상태</th>
                <th style={{ textAlign: "right" }}>동작</th>
              </tr>
            </thead>
            <tbody>
              {pl.pageRows.map((p) => {
                const isRunning = pipelineRunning && pipelineRun?.pipelineId === p.id;
                const last = pipelineHistory[p.id];
                return (
                  <tr key={p.id}>
                    <td>
                      <strong style={{ fontSize: 13 }}>{p.name}</strong>
                      <div>
                        <code style={{ fontSize: 11, color: "var(--accent-purple)" }}>{p.id}</code>
                      </div>
                    </td>
                    <td style={{ fontSize: 12 }}>
                      {p.model} {p.baseVersion} → {p.candidateVersion}
                      <div style={{ fontSize: 11, color: "var(--text-muted)" }}>{p.experiment}</div>
                    </td>
                    <td style={{ fontSize: 12, color: "var(--text-secondary)" }}>{p.triggerPolicy}</td>
                    <td style={{ fontSize: 11, color: "var(--text-secondary)" }}>
                      {last ? (
                        <>
                          <code style={{ color: "var(--accent-purple)" }}>{last.runId}</code>
                          <div style={{ color: "var(--text-muted)" }}>
                            {last.finishedAt} · {last.result}
                          </div>
                        </>
                      ) : (
                        "—"
                      )}
                    </td>
                    <td>
                      <span
                        className="system-status"
                        style={{
                          padding: "1px 8px",
                          fontSize: 10,
                          color: isRunning ? "var(--accent-orange)" : "var(--accent-teal)",
                          backgroundColor: isRunning ? "rgba(245, 158, 11, 0.1)" : "rgba(16, 185, 129, 0.1)"
                        }}
                      >
                        {isRunning ? "진행 중" : "대기"}
                      </span>
                    </td>
                    <td style={{ textAlign: "right" }}>
                      <button
                        className="btn btn-primary"
                        style={{ padding: "5px 14px", fontSize: 12 }}
                        onClick={() => handleRun(p)}
                        disabled={pipelineRunning}
                        title={`${p.name} 파이프라인을 즉시 실행합니다`}
                      >
                        <i className="fa-solid fa-play"></i> 실행
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <TablePager
          page={pl.safePage}
          totalPages={pl.totalPages}
          totalCount={RETRAIN_PIPELINES.length}
          onChange={setPlPage}
        />
      </Card>

      {/* ② 실행 상태 — 상시 표시(레이아웃 고정). 유휴 시 대기 상태, 실행 시 같은 자리에 내용만 채움 */}
      <div id="pipeline-status-anchor">
        <Card
          title={
            pipelineRun ? (
              <>
                파이프라인 실행 상태 — {pipelineRun.pipelineName}{" "}
                <code style={{ fontSize: 12, color: "var(--accent-purple)", fontWeight: 500 }}>
                  {pipelineRun.pipelineId}
                </code>
              </>
            ) : (
              "파이프라인 실행 상태 — 대기 중"
            )
          }
          icon="fa-diagram-project"
          style={{ marginBottom: 24 }}
          headerRight={
            <span style={{ display: "inline-flex", alignItems: "center", gap: 10 }}>
              <span
                className="system-status"
                style={{
                  padding: "2px 10px",
                  fontSize: 11,
                  color: !pipelineRun
                    ? "var(--text-muted)"
                    : pipelineRunning
                      ? "var(--accent-orange)"
                      : "var(--accent-teal)",
                  backgroundColor: !pipelineRun
                    ? "rgba(148, 163, 184, 0.12)"
                    : pipelineRunning
                      ? "rgba(245, 158, 11, 0.1)"
                      : "rgba(16, 185, 129, 0.1)"
                }}
              >
                {!pipelineRun ? "대기" : pipelineRunning ? "진행 중" : "완료"}
              </span>
              <button
                className="btn btn-secondary"
                onClick={handleReset}
                disabled={!pipelineRun}
                title={pipelineRun ? "실행 상태를 초기화합니다" : "초기화할 실행 이력이 없습니다"}
              >
                <i className="fa-solid fa-rotate-left"></i> 초기화
              </button>
            </span>
          }
        >
          {pipelineRun ? (
            <div className="run-meta">
              <span className="run-meta-item">
                <span className="run-meta-label">대상 모델</span>
                {pipelineRun.model} {pipelineRun.baseVersion} → 후보 {pipelineRun.candidateVersion}
              </span>
              <span className="run-meta-item">
                <span className="run-meta-label">실행 ID</span>
                <code>{pipelineRun.runId}</code>
              </span>
              <span className="run-meta-item">
                <span className="run-meta-label">실험</span>
                <code>{pipelineRun.experiment}</code>
              </span>
              <span className="run-meta-item">
                <span className="run-meta-label">트리거</span>
                {pipelineRun.trigger}
              </span>
              <span className="run-meta-item">
                <span className="run-meta-label">시작</span>
                {pipelineRun.startedAt}
              </span>
            </div>
          ) : (
            <p className="pipeline-idle-hint">
              <i className="fa-solid fa-circle-info" aria-hidden="true"></i> 위 카탈로그에서 파이프라인을
              선택해 [실행]을 누르면 진행 상황이 이 자리에 표시됩니다. 드리프트 감지 시에는 자동으로
              실행됩니다.
            </p>
          )}

          <div className={"pipeline-visualizer" + (pipelineRun ? "" : " is-idle")}>
            {PIPELINE_NODES.map((node, idx) => (
              <div key={node.id} style={{ display: "contents" }}>
                <div className={"pipeline-node " + nodeStatus(idx, pipelineStep)} id={node.id}>
                  <div className="node-icon">
                    <i className={"fa-solid " + node.icon}></i>
                  </div>
                  <div className="node-label">{node.label}</div>
                </div>
                {idx < PIPELINE_NODES.length - 1 && (
                  <div className={"pipeline-connector " + connectorStatus(idx, pipelineStep)}></div>
                )}
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* ③ Model Store — 2차년도 "Feature/Model Store 기반 버전 관리·최고 성능 모델 선택" 산출물 */}
      <Card
        title={
          <>
            Model Store — 모델·실험 버전 이력
            <InfoTip
              label="자동 모델 승급 기준"
              text="신규 모델은 다음을 모두 충족해야 운영으로 승급됩니다 — 6대 지표 기존 대비 +1.5% 이상 · 유닛·통합·성능 테스트 전체 통과 · 이상치 비율 < 0.5% · P95 지연 < 200ms. 배포 후 운영 Accuracy < 0.80 시 직전 버전으로 자동 롤백."
            />
          </>
        }
        icon="fa-boxes-stacked"
        style={{ marginBottom: 24 }}
        headerRight={
          <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
            최고 성능 모델 자동 선택 · 학습데이터·하이퍼파라미터 버전 추적
          </span>
        }
      >
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>모델</th>
                <th>버전</th>
                <th>학습데이터</th>
                <th>하이퍼파라미터</th>
                <th style={{ textAlign: "right" }}>Accuracy</th>
                <th>상태</th>
                <th>등록일</th>
              </tr>
            </thead>
            <tbody>
              {store.pageRows.map((m) => {
                const st = STORE_STATUS_STYLE[m.status] ?? STORE_STATUS_STYLE.이전;
                return (
                  <tr
                    key={`${m.modelId}-${m.version}`}
                    style={m.status === "운영" ? { backgroundColor: "rgba(16, 185, 129, 0.04)" } : undefined}
                  >
                    <td style={{ fontSize: 13 }}>
                      <strong>{modelName(m.modelId)}</strong>
                      <div>
                        <code style={{ fontSize: 11, color: "var(--accent-purple)" }}>{m.modelId}</code>
                      </div>
                    </td>
                    <td>
                      <code style={{ fontSize: 12, fontWeight: 600 }}>{m.version}</code>
                    </td>
                    <td style={{ fontSize: 12, color: "var(--text-secondary)" }}>{m.dataVersion}</td>
                    <td style={{ fontSize: 11, color: "var(--text-secondary)" }}>{m.params}</td>
                    <td style={{ textAlign: "right", fontWeight: 600 }}>{m.accuracy.toFixed(3)}</td>
                    <td>
                      <span
                        className="system-status"
                        style={{ padding: "1px 8px", fontSize: 10, color: st.color, backgroundColor: st.bg }}
                      >
                        {m.status}
                      </span>
                    </td>
                    <td style={{ fontSize: 11, color: "var(--text-muted)" }}>{m.registeredAt}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <TablePager
          page={store.safePage}
          totalPages={store.totalPages}
          totalCount={modelStore.length}
          onChange={setStorePage}
        />
      </Card>

      <Card title="오케스트레이터 실시간 실행 로그" icon="fa-terminal">
        <ConsoleLog logs={consoleLogs} />
      </Card>
    </>
  );
}
