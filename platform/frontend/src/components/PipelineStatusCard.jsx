import { Fragment } from "react";
import Card from "./Card.jsx";
import { PIPELINE_NODES } from "../constants/pipeline.js";
import { TARGET_LABELS, auditTime, connectorStatusFrom, displayId, jobNodeStatuses, jobStatusLabel, typeName } from "../lib/modelops.js";

const shortCode = (id) => <code title={id}>{displayId(id)}</code>;

// 이전 경로(데모 모드 과거 기록) 실행 결과 라벨 — 읽기 전용 표시에만 쓴다.
export const LEGACY_RUN_STATE_LABEL = {
  running: "실행 중",
  succeeded: "승급 완료",
  rejected: "승급 반려",
  rolled_back: "자동 롤백",
  debounced: "실행 조정",
  failed: "실패"
};

const LEGACY_TONE = { running: "busy", succeeded: "ok", rejected: "warn", debounced: "warn", rolled_back: "failed", failed: "failed" };

const TONE_STYLE = {
  idle: { color: "var(--text-muted)", backgroundColor: "var(--surface-hover)" },
  busy: { color: "var(--accent-orange)", backgroundColor: "rgba(var(--accent-orange-rgb), 0.02)" },
  warn: { color: "var(--accent-orange)", backgroundColor: "rgba(var(--accent-orange-rgb), 0.02)" },
  failed: { color: "var(--accent-red)", backgroundColor: "rgba(var(--accent-red-rgb), 0.02)" },
  ok: { color: "var(--accent-teal)", backgroundColor: "rgba(var(--accent-teal-rgb), 0.02)" }
};

// 저장된 실행 로그의 ISO 타임스탬프를 콘솔 표기(hh:mm:ss)로. 파싱 실패 시 원문을 남긴다.
export function logTime(ts) {
  const parsed = new Date(ts);
  return Number.isNaN(parsed.getTime()) ? String(ts ?? "") : parsed.toTimeString().slice(0, 8);
}

export function orDash(value) {
  return value === null || value === undefined || value === "" ? "–" : value;
}

// 이전 경로 PIPELINE_NODES(6칸) ↔ 옛 상태머신 stage(5단계). 마지막 칸은 승급 확정 여부다.
const NODE_STAGE = ["queued", "preparing", "training", "evaluating", "deploying", null];

/** 이전 경로 실행 레코드의 단계 상태를 노드 상태로(옛 화면 규칙 그대로). */
function legacyNodeStatus(index, run) {
  const stageName = NODE_STAGE[index];
  if (stageName === null) {
    if (run.state === "succeeded") return { className: "completed", label: "승급", icon: "fa-check" };
    if (run.state === "rolled_back") return { className: "failed", label: "롤백", icon: "fa-rotate-left" };
    return { className: "idle", label: "대기", icon: "fa-clock" };
  }
  const stage = (run.stages ?? []).find((s) => s.stage === stageName);
  if (!stage) {
    const reached = (run.stages ?? []).length;
    if (index === reached) {
      if (run.state === "running") return { className: "idle", label: "실행 중", icon: "fa-clock" };
      return run.state === "debounced"
        ? { className: "failed", label: "실행 조정", icon: "fa-pause" }
        : { className: "failed", label: run.state === "rejected" ? "승급 반려" : "미실행", icon: "fa-ban" };
    }
    return { className: "idle", label: "대기", icon: "fa-clock" };
  }
  if (stage.status === "skipped") return { className: "completed", label: "건너뜀", icon: "fa-forward" };
  return { className: "completed", label: "완료", icon: "fa-check" };
}

function announcement(nodes, status, target) {
  if (!target) return "파이프라인 실행 대기 중";
  const running = nodes.findIndex((n) => n.className === "running");
  if (running >= 0) return `${PIPELINE_NODES[running].label} 단계 진행 중, 전체 ${PIPELINE_NODES.length}단계 중 ${running + 1}단계`;
  if (target.kind === "modelops" && target.job.state === "failed") return `학습 실패, ${target.job.error ?? "원인을 확인하세요."}`;
  if (status.label === "승급 없음") return "검증 성능이 기준선을 넘지 못해 활성화할 수 없습니다";
  if (status.label === "등록 완료") return "학습 결과 등록 완료, 활성화 전";
  return `파이프라인 실행 결과 — ${status.label}`;
}

/**
 * ② 파이프라인 실행 상태 — 상시 표시(레이아웃 고정). 새 경로 작업 또는 선택한 이전 경로 기록을 그린다.
 * target: { kind: "modelops", job, version, instance, pipeline } | { kind: "legacy", run } | null
 * pollError: { message, stopped } — 이 작업의 상태 폴링 실패(화면 잠금은 풀려 있다). onRecheck 로 다시 폴링한다.
 */
export default function PipelineStatusCard({ target, regionName, onReset, anchorRef, pollError, onRecheck, sourceLabel }) {
  const nodes = !target ? jobNodeStatuses(null)
    : target.kind === "legacy" ? PIPELINE_NODES.map((_, index) => legacyNodeStatus(index, target.run))
      : jobNodeStatuses(target.job, target.version);
  const status = !target ? { label: "대기", tone: "idle" }
    : target.kind === "legacy" ? { label: LEGACY_RUN_STATE_LABEL[target.run.state] ?? target.run.state, tone: LEGACY_TONE[target.run.state] ?? "idle" }
      : jobStatusLabel(target.job, target.version);
  const busy = status.tone === "busy";
  const job = target?.kind === "modelops" ? target.job : null;
  const run = target?.kind === "legacy" ? target.run : null;
  const titleName = job ? target.pipeline?.name ?? job.pipeline_id : run?.pipeline_id ?? run?.model_id;
  const titleId = job ? job.pipeline_id : run?.run_id;
  const instance = target?.instance;

  return (
    // tabIndex=-1: [실행] 직후 결과 영역으로 초점을 옮기기 위한 프로그램 초점 대상
    <div id="pipeline-status-anchor" ref={anchorRef} tabIndex={-1}>
      <Card
        title={target ? <>파이프라인 실행 상태 — {orDash(titleName)}{" "}
          <code style={{ fontSize: 12, color: "var(--accent-purple-text)", fontWeight: 500 }} title={titleId}>{displayId(titleId)}</code></> : <>파이프라인 실행 상태 — 대기 중</>}
        icon="fa-diagram-project"
        className="page-section orchestrator-live"
        dataSource={target ? "api" : undefined}
        sourceLabel={target ? sourceLabel : undefined}
        headerRight={
          <span style={{ display: "inline-flex", alignItems: "center", gap: 10 }}>
            <span className="system-status" style={{ padding: "2px 10px", fontSize: 11, ...TONE_STYLE[status.tone] }}>
              {status.label}
            </span>
            <button
              className="btn btn-secondary"
              onClick={(event) => { if (!target) { event.preventDefault(); return; } onReset(); }}
              aria-disabled={!target}
              title={!target ? "초기화할 실행 이력이 없습니다" : busy ? "화면 표시만 초기화합니다. 서버의 학습 작업은 계속됩니다." : "실행 상태를 초기화합니다"}
            >
              <i className="fa-solid fa-rotate-left"></i> 초기화
            </button>
          </span>
        }
      >
        {job && (
          <div className="run-meta" data-values-source="api">
            <span className="run-meta-item">
              <span className="run-meta-label">대상 모델</span>
              {typeName(job.model_type)}{instance ? ` · ${regionName(instance.region_id)} · ${TARGET_LABELS[instance.target] ?? instance.target}` : ""}{" "}
              {instance?.active_version ? shortCode(instance.active_version) : "활성 없음"} → 후보 {job.version_id ? shortCode(job.version_id) : "–"}
            </span>
            <span className="run-meta-item"><span className="run-meta-label">실행 ID</span>{shortCode(job.id)}</span>
            <span className="run-meta-item"><span className="run-meta-label">실험</span><code>{orDash(target.pipeline?.experiment)}</code></span>
            <span className="run-meta-item"><span className="run-meta-label">트리거</span>{orDash(target.pipeline?.trigger_policy ?? "수동")} · {orDash(job.actor)}</span>
            <span className="run-meta-item"><span className="run-meta-label">시작</span>{orDash(auditTime(job.started_at ?? job.created_at))}</span>
            <span className="run-meta-item"><span className="run-meta-label">종료</span>{orDash(job.finished_at ? auditTime(job.finished_at) : null)}</span>
          </div>
        )}
        {job && pollError && (
          <p className="async-feedback is-error" role="alert">
            <i className="fa-solid fa-circle-exclamation" aria-hidden="true"></i>
            <span>
              학습 상태를 확인하지 못했습니다{pollError.message ? ` — ${pollError.message}` : "."}{" "}
              {pollError.stopped ? "연속 실패로 자동 확인을 멈췄습니다." : "잠시 후 자동으로 다시 확인합니다."}{" "}
              서버의 학습 작업은 계속될 수 있으며, 다른 작업은 지금 실행할 수 있습니다.
            </span>
            <button className="btn btn-secondary" onClick={onRecheck}>다시 확인</button>
          </p>
        )}
        {run && (
          <div className="run-meta" data-values-source="api">
            <span className="run-meta-item"><span className="run-meta-label">대상 모델</span>{orDash(run.model_id)} → {orDash(displayId(run.active_version))} (과거 기록)</span>
            <span className="run-meta-item"><span className="run-meta-label">실행 ID</span>{shortCode(run.run_id)}</span>
            <span className="run-meta-item"><span className="run-meta-label">트리거</span>{orDash(run.trigger)}</span>
            <span className="run-meta-item"><span className="run-meta-label">시작</span>{orDash(run.started_at ? auditTime(run.started_at) : null)}</span>
            <span className="run-meta-item"><span className="run-meta-label">종료</span>{orDash(run.finished_at ? auditTime(run.finished_at) : null)}</span>
          </div>
        )}
        {!target && (
          <p className="pipeline-idle-hint">
            <i className="fa-solid fa-circle-info" aria-hidden="true"></i> 위 목록에서 [실행]을 누르면 진행 상황이 이 자리에
            표시됩니다. 실행 이력에서 [로그]를 누르면 그 실행의 단계 결과를 다시 볼 수 있습니다.
          </p>
        )}
        <span className="sr-only" role="status" aria-live="polite" aria-atomic="true">
          {announcement(nodes, status, target)}
        </span>
        <div className={"pipeline-visualizer" + (target ? "" : " is-idle")} role="list" aria-label="재학습 파이프라인 단계별 상태">
          {PIPELINE_NODES.map((node, idx) => {
            const nodeState = nodes[idx];
            const firstRunning = nodes.findIndex((n) => n.className === "running") === idx;
            return (
              <Fragment key={node.id}>
                <div className={`pipeline-node ${nodeState.className}`} id={node.id} role="listitem" aria-current={firstRunning ? "step" : undefined}>
                  <div className="node-icon" aria-hidden="true"><i className={"fa-solid " + node.icon}></i></div>
                  <div className="node-label">{node.label}</div>
                  <div className="node-status">
                    <i className={`fa-solid ${nodeState.icon}`} aria-hidden="true"></i>
                    <span>{nodeState.label}</span>
                  </div>
                </div>
                {idx < PIPELINE_NODES.length - 1 && (
                  <div className={"pipeline-connector " + connectorStatusFrom(nodes, idx)} aria-hidden="true"></div>
                )}
              </Fragment>
            );
          })}
        </div>
      </Card>
    </div>
  );
}
