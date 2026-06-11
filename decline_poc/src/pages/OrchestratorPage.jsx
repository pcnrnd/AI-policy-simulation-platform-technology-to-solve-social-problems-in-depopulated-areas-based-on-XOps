import Card from "../components/Card.jsx";
import ConsoleLog from "../components/ConsoleLog.jsx";
import { useAppState } from "../context/AppStateContext.jsx";
import { PIPELINE_NODES } from "../constants/pipeline.js";
import { RETRAIN_PIPELINES } from "../constants/models.js";

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
    consoleLogs,
    startPipeline,
    resetPipeline,
    addConsoleLog
  } = useAppState();

  const handleReset = () => {
    resetPipeline();
    addConsoleLog("INFO: MLOps 재학습 파이프라인이 초기화되었습니다.");
  };

  return (
    <>
      <Card
        title="이벤트 기반 AI 모델 자동 오케스트레이션 파이프라인"
        icon="fa-diagram-project"
        style={{ marginBottom: 24 }}
        headerRight={
          <button className="btn btn-secondary" onClick={handleReset}>
            <i className="fa-solid fa-rotate-left"></i> 파이프라인 초기화
          </button>
        }
      >
        <p style={{ fontSize: 13, color: "var(--text-secondary)", marginBottom: 12 }}>
          데이터 드리프트(PSI {">"} 0.2) 혹은 모델 성능 이상 감지 이벤트 발생 시 MLOps 오케스트레이터가
          동작을 수신하여{" "}
          <strong>
            학습 데이터 추출 → 데이터 검증 → 파라미터 튜닝 재학습 → 검증 평가 → 블루-그린 승급 배포
          </strong>
          를 전자동으로 수행합니다.
        </p>

        {/* 실행 식별 정보 — 어떤 모델·실험·트리거의 파이프라인인지 표시 */}
        {pipelineRun ? (
          <div className="run-meta">
            <span className="run-meta-item">
              <span className="run-meta-label">파이프라인</span>
              {pipelineRun.pipelineName} (<code>{pipelineRun.pipelineId}</code>)
            </span>
            <span className="run-meta-item">
              <span className="run-meta-label">실행 ID</span>
              <code>{pipelineRun.runId}</code>
            </span>
            <span className="run-meta-item">
              <span className="run-meta-label">대상 모델</span>
              {pipelineRun.model} {pipelineRun.baseVersion} → 후보 {pipelineRun.candidateVersion}
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
            <span
              className="system-status"
              style={{
                padding: "2px 10px",
                fontSize: 11,
                marginLeft: "auto",
                color: pipelineRunning ? "var(--accent-orange)" : "var(--accent-teal)",
                backgroundColor: pipelineRunning ? "rgba(245, 158, 11, 0.1)" : "rgba(16, 185, 129, 0.1)"
              }}
            >
              {pipelineRunning ? "진행 중" : "완료"}
            </span>
          </div>
        ) : (
          <div className="run-meta run-meta-empty">
            아직 실행 이력이 없습니다 — 모니터 탭에서 드리프트 감지 시 자동 실행되거나, 아래 재학습
            파이프라인 목록에서 [실행]으로 시작할 수 있습니다.
          </div>
        )}

        <div className="pipeline-visualizer">
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

      {/* 등록된 재학습 파이프라인 카탈로그 — 어떤 파이프라인을 재학습할지 선택·실행 */}
      <Card
        title="등록된 재학습 파이프라인"
        icon="fa-list-check"
        style={{ marginBottom: 24 }}
        headerRight={
          <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
            {RETRAIN_PIPELINES.length}건 등록 · 모델 레지스트리 연동
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
              {RETRAIN_PIPELINES.map((pl) => {
                const isRunning = pipelineRunning && pipelineRun?.pipelineId === pl.id;
                const last = pipelineHistory[pl.id];
                return (
                  <tr key={pl.id}>
                    <td>
                      <strong style={{ fontSize: 13 }}>{pl.name}</strong>
                      <div>
                        <code style={{ fontSize: 11, color: "var(--accent-purple)" }}>{pl.id}</code>
                      </div>
                    </td>
                    <td style={{ fontSize: 12 }}>
                      {pl.model} {pl.baseVersion} → {pl.candidateVersion}
                      <div style={{ fontSize: 11, color: "var(--text-muted)" }}>{pl.experiment}</div>
                    </td>
                    <td style={{ fontSize: 12, color: "var(--text-secondary)" }}>{pl.triggerPolicy}</td>
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
                        onClick={() => startPipeline("수동 실행 (파이프라인 카탈로그)", pl)}
                        disabled={pipelineRunning}
                        title={`${pl.name} 파이프라인을 즉시 실행합니다`}
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
      </Card>

      <div className="grid-details-split">
        <Card title="오케스트레이터 실시간 실행 로그" icon="fa-terminal">
          <ConsoleLog logs={consoleLogs} />
        </Card>

        <Card title="자동 모델 승격(DoD) 기준" icon="fa-clipboard-check">
          <div style={{ fontSize: 13, lineHeight: 1.8 }}>
            <DodRow label="6대 정확도 지표 개선" value="기존 모델 대비 > 1.5%" color="teal" />
            <DodRow label="데이터 유입 안정성 (Outlier)" value="이상치 비율 < 0.5%" color="teal" />
            <DodRow label="자동 롤백 임계 성능" value="Accuracy < 0.80 미만 시 즉각" color="red" />
            <DodRow label="추론 속도 지연시간 (P95)" value="< 200ms 만족" color="teal" />
          </div>
        </Card>
      </div>
    </>
  );
}

function DodRow({ label, value, color }) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        padding: "6px 0",
        borderBottom: "1px solid var(--border-color)"
      }}
    >
      <span style={{ color: "var(--text-secondary)" }}>{label}</span>
      <span
        style={{
          fontWeight: 600,
          color: color === "red" ? "var(--accent-red)" : "var(--accent-teal)"
        }}
      >
        {value}
      </span>
    </div>
  );
}
