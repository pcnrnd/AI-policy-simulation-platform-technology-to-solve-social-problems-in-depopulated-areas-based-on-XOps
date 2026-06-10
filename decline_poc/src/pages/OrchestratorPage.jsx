import Card from "../components/Card.jsx";
import ConsoleLog from "../components/ConsoleLog.jsx";
import { useAppState } from "../context/AppStateContext.jsx";
import { PIPELINE_NODES } from "../constants/pipeline.js";

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
  const { pipelineRunning, pipelineStep, consoleLogs, startPipeline, resetPipeline, addConsoleLog } =
    useAppState();

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
          <div style={{ display: "flex", gap: 12 }}>
            <button className="btn btn-secondary" onClick={handleReset}>
              <i className="fa-solid fa-rotate-left"></i> 파이프라인 초기화
            </button>
            <button
              className="btn btn-primary"
              onClick={startPipeline}
              disabled={pipelineRunning}
            >
              <i className="fa-solid fa-play"></i> 재학습 파이프라인 실행
            </button>
          </div>
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
