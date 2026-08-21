import { Fragment, useEffect, useRef, useState } from "react";
import Card from "../components/Card.jsx";
import ConsoleLog from "../components/ConsoleLog.jsx";
import InfoTip from "../components/InfoTip.jsx";
import TablePager, { paginate } from "../components/TablePager.jsx";
import { useAppState } from "../context/AppStateContext.jsx";
import { PIPELINE_NODES } from "../constants/pipeline.js";
import { RETRAIN_PIPELINES, MODEL_REGISTRY } from "../constants/models.js";

const PAGE_SIZE = 5;

const STORE_STATUS_STYLE = {
  운영: { color: "var(--accent-teal)", bg: "rgba(var(--accent-teal-rgb), 0.02)" },
  이전: { color: "var(--text-muted)", bg: "var(--surface-hover)" },
  롤백: { color: "var(--accent-red)", bg: "rgba(var(--accent-red-rgb), 0.02)" }
};

function nodeStatus(index, currentStep, terminalState = null) {
  // index: 0-based, currentStep: 1-based (0 = not started)
  if (terminalState) {
    const terminal = {
      failed: { index: 0, className: "failed", label: "실패", icon: "fa-triangle-exclamation" },
      debounced: { index: 0, className: "failed", label: "실행 조정", icon: "fa-pause" },
      rejected: { index: 3, className: "failed", label: "승급 반려", icon: "fa-ban" },
      rolled_back: { index: 4, className: "failed", label: "롤백", icon: "fa-rotate-left" }
    }[terminalState];
    if (terminal) {
      if (index < terminal.index) return { className: "completed", label: "완료", icon: "fa-check" };
      if (index === terminal.index) return terminal;
      return { className: "idle", label: "대기", icon: "fa-clock" };
    }
  }
  const stepIdx = currentStep - 1;
  if (currentStep === 0 || index > stepIdx) {
    return { className: "idle", label: "대기", icon: "fa-clock" };
  }
  if (index < stepIdx) {
    return { className: "completed", label: "완료", icon: "fa-check" };
  }
  return { className: "running", label: "진행 중", icon: "fa-spinner fa-spin" };
}

function connectorStatus(index, currentStep, terminalState = null) {
  const terminalIndex = { failed: 0, debounced: 0, rejected: 3, rolled_back: 4 }[terminalState];
  if (terminalIndex !== undefined) return index < terminalIndex ? "success" : "";
  const stepIdx = currentStep - 1;
  if (currentStep === 0) return "";
  if (index < stepIdx - 1) return "success";
  if (index === stepIdx - 1) return "success";
  if (index === stepIdx) return "active";
  return "";
}

// 빈 값 자리는 대시("–", §10 UI-04)로 채운다 — 숫자 0과 구분되고 문장 구분자(—)와도 섞이지 않는다.
function orDash(value) {
  return value === null || value === undefined || value === "" ? "–" : value;
}

export default function OrchestratorPage() {
  const {
    pipelineRunning,
    pipelineScheduled,
    pipelineStep,
    pipelineRun,
    pipelineResult,
    pipelineHistory,
    modelStore,
    consoleLogs,
    startPipeline,
    resetPipeline,
    addConsoleLog
  } = useAppState();

  const modelName = (id) => MODEL_REGISTRY.find((m) => m.id === id)?.name || id || "–";
  const pipelineBusy = pipelineRunning || pipelineScheduled;

  // 예약(드리프트 감지 후 실행 대기) 구간에는 직전 실행의 단계·종료 상태가 아직 남아 있다
  // (초기화는 startPipeline에서 수행). 끝난 실행의 결과가 새 실행의 결과처럼 보이지 않게 감춘다.
  const visibleStep = pipelineScheduled ? 0 : pipelineStep;
  const terminalState = !pipelineBusy ? pipelineResult?.state : null;
  const pipelineFailed = terminalState === "failed" || terminalState === "rolled_back";
  const pipelineNotPromoted = terminalState === "rejected" || terminalState === "debounced";
  const terminalLabel = pipelineFailed ? (terminalState === "rolled_back" ? "롤백" : "실패") : pipelineNotPromoted ? "승급 없음" : "완료";
  const statusLabel = pipelineScheduled
    ? "예약"
    : !pipelineRun
      ? "대기"
      : pipelineRunning
        ? "진행 중"
        : terminalLabel;
  const pipelineAnnouncement = pipelineScheduled
    ? "드리프트 대응 재학습이 예약되어 곧 실행됩니다"
    : !pipelineRun
      ? "파이프라인 실행 대기 중"
      : pipelineRunning
        ? `${PIPELINE_NODES[visibleStep - 1]?.label ?? "파이프라인"} 단계 진행 중, 전체 ${PIPELINE_NODES.length}단계 중 ${Math.min(visibleStep, PIPELINE_NODES.length)}단계`
        : pipelineFailed
          ? `파이프라인 ${terminalLabel}, ${pipelineResult.reason ?? pipelineResult.deploy?.reason ?? "오류 원인을 확인하세요."}`
          : pipelineNotPromoted
            ? `파이프라인 실행 완료, 후보 모델 승급 없음 (${terminalState})`
            : `파이프라인 실행 완료, 전체 ${PIPELINE_NODES.length}단계 완료`;

  // 테이블 페이징 (파이프라인 카탈로그 / Model Store)
  const [plPage, setPlPage] = useState(1);
  const [storePage, setStorePage] = useState(1);
  const statusAnchorRef = useRef(null);
  const runBusyRef = useRef(false);
  const resetBusyRef = useRef(false);
  const [statusFocusRequest, setStatusFocusRequest] = useState(0);
  const pl = paginate(RETRAIN_PIPELINES, plPage, PAGE_SIZE);
  const store = paginate(modelStore, storePage, PAGE_SIZE);

  // 잠금은 native disabled 대신 aria-disabled로 건다. disabled를 걸면 자기 활성화로 잠기는 순간
  // 브라우저가 초점을 body로 떨어뜨린다(§9 A11Y-01). 실행 차단은 핸들러 가드가 담당한다.
  // ref 가드: 같은 틱의 연타는 state 갱신 전이라 resetLocked로 막을 수 없다(초기화 로그 중복 방지).
  const resetLocked = !pipelineRun || pipelineBusy;
  const handleReset = (event) => {
    if (resetLocked || resetBusyRef.current) {
      event.preventDefault();
      return;
    }
    resetBusyRef.current = true;
    Promise.resolve(resetPipeline()).finally(() => {
      resetBusyRef.current = false;
    });
    addConsoleLog("INFO: MLOps 재학습 파이프라인이 초기화되었습니다.");
  };

  // 카탈로그 [실행] → 실행 시작 + 아래 실행 상태 카드로 초점·스크롤 (누른 곳에서 결과가 보이도록)
  // ref 가드: 같은 틱의 연타는 state 갱신 전이라 pipelineBusy로 막을 수 없다(중복 오케스트레이션 요청 방지).
  const handleRun = (event, plDef, locked) => {
    if (locked || runBusyRef.current) {
      event.preventDefault();
      return;
    }
    runBusyRef.current = true;
    setStatusFocusRequest((request) => request + 1);
    Promise.resolve(startPipeline("수동 실행 (파이프라인 카탈로그)", plDef)).finally(() => {
      runBusyRef.current = false;
    });
  };

  // startPipeline이 세운 상태와 같은 틱에 커밋되므로, 이 effect는 갱신된 실행 상태 카드에서 실행된다.
  // 고정 지연(setTimeout) 없이 결정적으로 초점·스크롤을 옮긴다.
  useEffect(() => {
    if (statusFocusRequest === 0) return;
    const anchor = statusAnchorRef.current;
    if (!anchor) return;
    anchor.focus({ preventScroll: true });
    anchor.scrollIntoView({ block: "start" });
  }, [statusFocusRequest]);

  return (
    <>
      {/* ① 진입점: 등록된 재학습 파이프라인 카탈로그 — 선택·실행 */}
      <Card
        title="등록된 재학습 파이프라인"
        icon="fa-list-check"
        className="page-section"
        headerRight={
          <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
            {RETRAIN_PIPELINES.length}건 등록 · 모델 레지스트리 연동 · 드리프트 감지 시 자동 실행
          </span>
        }
      >
        <div className="table-container">
          <table>
            <caption className="sr-only">등록된 재학습 파이프라인과 실행 상태</caption>
            <thead>
              <tr>
                <th scope="col">파이프라인</th>
                <th scope="col">대상 모델</th>
                <th scope="col">트리거 조건</th>
                <th scope="col">마지막 실행</th>
                <th scope="col">상태</th>
                <th scope="col" className="cell-actions">동작</th>
              </tr>
            </thead>
            <tbody>
              {pl.pageRows.map((p) => {
                const isRunning = pipelineRunning && pipelineRun?.pipelineId === p.id;
                const last = pipelineHistory[p.id];
                const currentServing = modelStore.find((model) => model.modelId === p.model && model.status === "운영");
                const candidateAvailable = currentServing?.version !== p.candidateVersion;
                const runLocked = pipelineBusy || !candidateAvailable;
                return (
                  <tr key={p.id}>
                    <td>
                      <strong style={{ fontSize: 13 }}>{orDash(p.name)}</strong>
                      <div>
                        <code style={{ fontSize: 11, color: "var(--accent-purple-text)" }}>{orDash(p.id)}</code>
                      </div>
                    </td>
                    <td style={{ fontSize: 12 }}>
                      {orDash(p.model)} {orDash(currentServing?.version ?? p.baseVersion)}
                      {candidateAvailable ? ` → ${orDash(p.candidateVersion)}` : " · 다음 후보 미등록"}
                      <div style={{ fontSize: 11, color: "var(--text-muted)" }}>{orDash(p.experiment)}</div>
                    </td>
                    <td style={{ fontSize: 12, color: "var(--text-secondary)" }}>{orDash(p.triggerPolicy)}</td>
                    <td style={{ fontSize: 11, color: "var(--text-secondary)" }}>
                      {last ? (
                        <>
                          <code style={{ color: "var(--accent-purple-text)" }}>{orDash(last.runId)}</code>
                          <div style={{ color: "var(--text-muted)" }}>
                            {orDash(last.finishedAt)} · {orDash(last.result)}
                          </div>
                        </>
                      ) : (
                        "–"
                      )}
                    </td>
                    <td>
                      <span
                        className="system-status"
                        style={{
                          padding: "1px 8px",
                          fontSize: 10,
                          color: isRunning || !candidateAvailable ? "var(--accent-orange)" : "var(--accent-teal)",
                          backgroundColor: isRunning || !candidateAvailable
                            ? "rgba(var(--accent-orange-rgb), 0.02)"
                            : "rgba(var(--accent-teal-rgb), 0.02)",
                          borderColor: "currentColor"
                        }}
                      >
                        {isRunning ? "진행 중" : candidateAvailable ? "대기" : "후보 필요"}
                      </span>
                    </td>
                    <td className="cell-actions">
                      <button
                        className="btn btn-primary"
                        style={{ padding: "5px 14px", fontSize: 12 }}
                        onClick={(event) => handleRun(event, p, runLocked)}
                        aria-disabled={runLocked}
                        title={
                          !candidateAvailable
                            ? "현재 운영 버전보다 새로운 후보 모델이 백엔드 모델 레지스트리에 등록되어야 실행할 수 있습니다. 이 화면에서는 후보를 등록할 수 없습니다."
                            : pipelineBusy
                              ? "다른 재학습 파이프라인이 실행 중입니다. 완료 후 실행할 수 있습니다."
                              : `${p.name} 파이프라인을 즉시 실행합니다`
                        }
                        aria-label={`${orDash(p.name)} 파이프라인 실행`}
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
          pageSize={PAGE_SIZE}
          onChange={setPlPage}
        />
      </Card>

      {/* ② 실행 상태 — 상시 표시(레이아웃 고정). 유휴 시 대기 상태, 실행 시 같은 자리에 내용만 채움 */}
      {/* tabIndex=-1: [실행] 직후 결과 영역으로 초점을 옮기기 위한 프로그램 초점 대상 (초점이 body로 떨어지지 않게) */}
      <div id="pipeline-status-anchor" ref={statusAnchorRef} tabIndex={-1}>
        <Card
          title={
            pipelineRun ? (
              <>
                파이프라인 실행 상태 — {pipelineRun.pipelineName}{" "}
                <code style={{ fontSize: 12, color: "var(--accent-purple-text)", fontWeight: 500 }}>
                  {pipelineRun.pipelineId}
                </code>
              </>
            ) : (
              "파이프라인 실행 상태 — 대기 중"
            )
          }
          icon="fa-diagram-project"
          className="page-section"
          headerRight={
            <span style={{ display: "inline-flex", alignItems: "center", gap: 10 }}>
              {/* 단계 전환 알림은 아래 sr-only 라이브 리전 한 곳만 담당한다(같은 전환을 두 번 읽지 않게).
                  이 칩은 시각적 상태 표기를 그대로 유지한다. */}
              <span
                className="system-status"
                style={{
                  padding: "2px 10px",
                  fontSize: 11,
                  color: pipelineBusy
                    ? "var(--accent-orange)"
                    : !pipelineRun
                      ? "var(--text-muted)"
                      : pipelineFailed
                        ? "var(--accent-red)"
                        : pipelineNotPromoted
                          ? "var(--accent-orange)"
                          : "var(--accent-teal)",
                  backgroundColor: pipelineBusy
                    ? "rgba(var(--accent-orange-rgb), 0.02)"
                    : !pipelineRun
                      ? "var(--surface-hover)"
                      : pipelineFailed
                        ? "rgba(var(--accent-red-rgb), 0.02)"
                        : pipelineNotPromoted
                          ? "rgba(var(--accent-orange-rgb), 0.02)"
                          : "rgba(var(--accent-teal-rgb), 0.02)"
                }}
              >
                {statusLabel}
              </span>
              <button
                className="btn btn-secondary"
                onClick={handleReset}
                aria-disabled={resetLocked}
                title={
                  pipelineBusy
                    ? "실행이 끝난 뒤 초기화할 수 있습니다"
                    : pipelineRun
                      ? "실행 상태를 초기화합니다"
                      : "초기화할 실행 이력이 없습니다"
                }
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

          <span className="sr-only" role="status" aria-live="polite" aria-atomic="true">
            {pipelineAnnouncement}
          </span>
          <div
            className={"pipeline-visualizer" + (pipelineRun ? "" : " is-idle")}
            role="list"
            aria-label="재학습 파이프라인 단계별 상태"
          >
            {PIPELINE_NODES.map((node, idx) => {
              const status = nodeStatus(idx, visibleStep, terminalState);
              return (
                <Fragment key={node.id}>
                  <div
                    className={`pipeline-node ${status.className}`}
                    id={node.id}
                    role="listitem"
                    aria-current={status.className === "running" ? "step" : undefined}
                  >
                    <div className="node-icon" aria-hidden="true">
                      <i className={"fa-solid " + node.icon}></i>
                    </div>
                    <div className="node-label">{node.label}</div>
                    <div className="node-status">
                      <i className={`fa-solid ${status.icon}`} aria-hidden="true"></i>
                      <span>{status.label}</span>
                    </div>
                  </div>
                  {idx < PIPELINE_NODES.length - 1 && (
                    <div
                      className={"pipeline-connector " + connectorStatus(idx, visibleStep, terminalState)}
                      aria-hidden="true"
                    ></div>
                  )}
                </Fragment>
              );
            })}
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
        className="page-section"
        headerRight={
          <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
            최고 성능 모델 자동 선택 · 학습데이터·하이퍼파라미터 버전 추적
          </span>
        }
      >
        <div className="promotion-criteria" aria-labelledby="promotion-criteria-title">
          <strong id="promotion-criteria-title">자동 모델 승급 기준</strong>
          <span>
            6대 지표가 기존 대비 1.5% 이상 개선되고, 유닛·통합·성능 테스트를 모두 통과하며,
            이상치 비율 0.5% 미만·P95 지연 200ms 미만이어야 합니다. 배포 후 운영 Accuracy가
            0.80 미만이면 직전 버전으로 자동 롤백합니다.
          </span>
        </div>
        <div className="table-container">
          <table>
            <caption className="sr-only">모델과 실험의 버전 이력 및 운영 상태</caption>
            <thead>
              <tr>
                <th scope="col">모델</th>
                <th scope="col">버전</th>
                <th scope="col">학습데이터</th>
                <th scope="col">하이퍼파라미터</th>
                <th scope="col" className="cell-num">Accuracy</th>
                <th scope="col">상태</th>
                <th scope="col">등록일</th>
              </tr>
            </thead>
            <tbody>
              {store.pageRows.map((m) => {
                const st = STORE_STATUS_STYLE[m.status] ?? STORE_STATUS_STYLE.이전;
                return (
                  <tr
                    key={`${m.modelId}-${m.version}`}
                    style={m.status === "운영" ? { backgroundColor: "rgba(var(--accent-teal-rgb), 0.04)" } : undefined}
                  >
                    <td style={{ fontSize: 13 }}>
                      <strong>{modelName(m.modelId)}</strong>
                      <div>
                        <code style={{ fontSize: 11, color: "var(--accent-purple-text)" }}>{orDash(m.modelId)}</code>
                      </div>
                    </td>
                    <td>
                      <code style={{ fontSize: 12, fontWeight: 600 }}>{orDash(m.version)}</code>
                    </td>
                    <td style={{ fontSize: 12, color: "var(--text-secondary)" }}>{orDash(m.dataVersion)}</td>
                    <td style={{ fontSize: 11, color: "var(--text-secondary)" }}>{orDash(m.params)}</td>
                    <td className="cell-num" style={{ fontWeight: 600 }}>
                      {typeof m.accuracy === "number" ? m.accuracy.toFixed(3) : "–"}
                    </td>
                    <td>
                      <span
                        className="system-status"
                        style={{ padding: "1px 8px", fontSize: 10, color: st.color, backgroundColor: st.bg }}
                      >
                        {orDash(m.status)}
                      </span>
                    </td>
                    <td style={{ fontSize: 11, color: "var(--text-muted)" }}>{orDash(m.registeredAt)}</td>
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
          pageSize={PAGE_SIZE}
          onChange={setStorePage}
        />
      </Card>

      <Card title="오케스트레이터 실시간 실행 로그" icon="fa-terminal">
        <ConsoleLog logs={consoleLogs} />
      </Card>
    </>
  );
}
