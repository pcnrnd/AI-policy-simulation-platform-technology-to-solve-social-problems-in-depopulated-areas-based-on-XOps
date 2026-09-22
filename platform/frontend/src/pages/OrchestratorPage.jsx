import { Fragment, useCallback, useEffect, useRef, useState } from "react";
import Card from "../components/Card.jsx";
import ConsoleLog from "../components/ConsoleLog.jsx";
import InfoTip from "../components/InfoTip.jsx";
import NextStepBanner from "../components/NextStepBanner.jsx";
import PipelineRegisterForm from "../components/PipelineRegisterForm.jsx";
import TablePager, { paginate } from "../components/TablePager.jsx";
import RealdataTrainingPanel from "../components/realdata/RealdataTrainingPanel.jsx";
import { getRealdataToken, getTrainingRun, startTrainingRun } from "../components/realdata/realdataClient.jsx";
import { useAppState } from "../context/AppStateContext.jsx";
import { PIPELINE_NODES } from "../constants/pipeline.js";
import { MODEL_REGISTRY } from "../constants/models.js";
import { apiGet, apiSend } from "../lib/api.js";

const PAGE_SIZE = 5;

// 실데이터 학습 job 폴링 — 하단 실데이터 학습 패널과 같은 종결 상태·주기를 쓴다.
const JOB_POLL_MS = 1500;
const JOB_POLL_LIMIT = 40;
const JOB_TERMINAL_STATES = new Set(["saved", "failed", "cancelled"]);

const RUN_STATE_LABEL = {
  running: "실행 중",
  succeeded: "승급 완료",
  rejected: "승급 반려",
  rolled_back: "자동 롤백",
  debounced: "실행 조정",
  failed: "실패"
};

const REALDATA_RUN_STATE_LABEL = {
  saved: "후보 등록",
  succeeded: "후보 등록",
  failed: "실패",
  cancelled: "취소"
};

function runStateLabel(run) {
  const label =
    run.source === "realdata" ? REALDATA_RUN_STATE_LABEL[run.state] : RUN_STATE_LABEL[run.state];
  return label ?? run.state;
}

// 저장된 실행 로그의 ISO 타임스탬프를 콘솔 표기(hh:mm:ss)로. 파싱 실패 시 원문을 남긴다.
function logTime(ts) {
  const parsed = new Date(ts);
  return Number.isNaN(parsed.getTime()) ? String(ts ?? "") : parsed.toTimeString().slice(0, 8);
}

// 백엔드 파이프라인 정의(snake_case) → AppStateContext.startPipeline 이 기대하는 실행 정의.
function toRunDef(pipeline) {
  return {
    id: pipeline.id,
    name: pipeline.name,
    model: pipeline.model_id,
    modelId: pipeline.model_id,
    baseVersion: pipeline.base_version,
    candidateVersion: pipeline.candidate_version,
    experiment: pipeline.experiment,
    triggerPolicy: pipeline.trigger_policy
  };
}

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

// PIPELINE_NODES(6칸) ↔ 백엔드 상태머신 stage(5단계) 대응. 마지막 칸은 승급 확정 여부다.
const NODE_STAGE = ["queued", "preparing", "training", "evaluating", "deploying", null];

/**
 * 저장된 실행 레코드의 단계 상태를 노드 상태로. 애니메이션(setTimeout) 대신 실제 실행 결과를 그린다.
 * stage 가 기록되지 않았으면 그 단계까지 못 갔다는 뜻이라 실패/대기로 남긴다.
 */
function storedNodeStatus(index, run) {
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
      // 접수만 된 실행은 아직 이 단계에 닿지 않았을 뿐이라 실패로 그리지 않는다.
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
// 실데이터 회귀 지표 표기 — 원 단위(수만~수억)라 천단위 구분으로 줄여 읽는다.
function fmtMetric(value) {
  return Number.isFinite(value) ? Number(value).toLocaleString(undefined, { maximumFractionDigits: 1 }) : "–";
}

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
    modelStore,
    consoleLogs,
    startPipeline,
    resetPipeline,
    addConsoleLog,
    mockDataVisible
  } = useAppState();

  // 데모 표시 토글은 화면 구조가 아니라 **데이터 유무**만 바꾼다. OFF에서는 시드(백엔드 부트스트랩
  // 파이프라인·시드 지표 모델·상수 Model Store 이력)를 데이터 계층에서 끊고 빈 상태 문구를 남긴다.
  // 가림 CSS로 지우면 실데이터 행까지 함께 사라지고 "왜 비었는지"를 읽을 수 없다.
  const allowSeed = mockDataVisible;
  // 저장된 실행 레코드가 없을 때만 쓰이는 프런트 상수 폴백값 — 데모 OFF에서는 넘기지 않는다.
  const seedOr = (seedValue) => (allowSeed ? seedValue : null);

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

  // 파이프라인 정의·실행 이력·실행 로그는 모두 백엔드 저장소(SQLite)에서 읽는다.
  // pipelines=null 은 "아직 안 불러옴", [] 는 "등록 없음"(빈 상태 문구)으로 구분한다.
  const [pipelines, setPipelines] = useState(null);
  const [runs, setRuns] = useState([]);
  const [models, setModels] = useState([]);
  const [catalogError, setCatalogError] = useState(null);
  const [registerOpen, setRegisterOpen] = useState(false);
  const [selectedRunId, setSelectedRunId] = useState(null);
  const [runLogs, setRunLogs] = useState(null);
  // 실데이터(rd_*) 접근 토큰 — 백엔드는 `data:read` 토큰이 있는 호출에만 실데이터 job·모델을 싣는다.
  // undefined = 확인 전(카탈로그 요청을 보내지 않는다), null = 발급 실패, 문자열 = 발급 완료.
  const [realdataToken, setRealdataToken] = useState(allowSeed ? null : undefined);
  // 실데이터 학습 실행 중인 모델 id — 카탈로그 행의 상태 배지·실행 잠금이 이 값을 본다.
  const [realdataBusyModel, setRealdataBusyModel] = useState(null);
  const realdataRunBusyRef = useRef(false);

  useEffect(() => {
    if (allowSeed) {
      setRealdataToken(null);
      return undefined;
    }
    let alive = true;
    setRealdataToken(undefined);
    // 토큰은 사용자 관심사가 아니다 — 실패하면 1회 자동 재시도하고(캐시는 실패 시 비워진다),
    // 그래도 못 받으면 사유 문구 없이 잠긴 상태로 둔다.
    getRealdataToken()
      .catch(() => getRealdataToken())
      .then((token) => alive && setRealdataToken(token))
      .catch(() => alive && setRealdataToken(null));
    return () => {
      alive = false;
    };
  }, [allowSeed]);

  const reloadCatalog = useCallback(async () => {
    try {
      // 데모 OFF면 서버가 시드 파이프라인·시드 지표 모델·그 파이프라인이 만든 실행을 빼고 내려준다.
      // include_seed 는 호출마다 적어 둔다(공유 변수로 감추면 새 호출에서 빠져도 드러나지 않는다).
      // 데모 OFF에서는 토큰을 함께 보낸다 — 실데이터 학습 job·모델은 인증된 호출에만 실린다.
      const token = realdataToken ?? undefined;
      const [pipelineRows, runRows, modelRows] = await Promise.all([
        apiGet(`/api/v3/orchestration/pipelines?include_seed=${mockDataVisible}`, { token }),
        apiGet(`/api/v3/orchestration/runs?include_seed=${mockDataVisible}`, { token }),
        apiGet(`/api/v3/orchestration/models?include_seed=${mockDataVisible}`, { token })
      ]);
      setPipelines(Array.isArray(pipelineRows) ? pipelineRows : []);
      setRuns(Array.isArray(runRows) ? runRows : []);
      setModels(Array.isArray(modelRows) ? modelRows : []);
      setCatalogError(null);
    } catch (err) {
      const message = err?.message ?? "알 수 없는 오류";
      setPipelines((prev) => prev ?? []);
      setCatalogError(message);
      addConsoleLog(`ERROR: 파이프라인 카탈로그 로드 실패 — ${message}`);
    }
  }, [addConsoleLog, mockDataVisible, realdataToken]);

  // 토큰 확인 전에 보내면 실데이터가 빠진 목록을 먼저 그리고 곧바로 다시 그린다.
  const catalogReady = allowSeed || realdataToken !== undefined;

  useEffect(() => {
    if (!catalogReady) return;
    reloadCatalog();
  }, [reloadCatalog, catalogReady]);

  // 실행이 끝나면(백엔드 PipelineRun 수신) 이력을 다시 읽고 그 실행의 로그를 펼친다.
  useEffect(() => {
    if (!pipelineResult?.run_id) return;
    setSelectedRunId(pipelineResult.run_id);
    reloadCatalog();
  }, [pipelineResult, reloadCatalog]);

  useEffect(() => {
    if (!selectedRunId) {
      setRunLogs(null);
      return undefined;
    }
    let alive = true;
    apiGet(`/api/v3/orchestration/runs/${encodeURIComponent(selectedRunId)}/logs`, {
      token: realdataToken ?? undefined
    })
      .then((data) => alive && setRunLogs(data))
      .catch((err) => {
        if (!alive) return;
        setRunLogs(null);
        addConsoleLog(`WARN: 실행 로그 조회 실패 (${selectedRunId}) — ${err?.message ?? "알 수 없는 오류"}`);
      });
    return () => {
      alive = false;
    };
  }, [selectedRunId, addConsoleLog, realdataToken]);

  // aria-disabled 는 클릭을 막지 않으므로(§9 A11Y-01 규약) 실행 차단은 여기서 한다.
  const handleDeletePipeline = async (pipelineId) => {
    if (pipelineRunning || pipelineScheduled) return;
    try {
      await apiSend("DELETE", `/api/v3/orchestration/pipelines/${encodeURIComponent(pipelineId)}`);
      addConsoleLog(`INFO: 재학습 파이프라인 등록 해제 — ${pipelineId} (실행 이력은 보존됩니다)`);
      await reloadCatalog();
    } catch (err) {
      addConsoleLog(`WARN: 파이프라인 삭제 실패 — ${err?.message ?? "알 수 없는 오류"}`, false, true);
    }
  };

  // 테이블 페이징 (파이프라인 카탈로그 / Model Store / 실행 이력)
  const [plPage, setPlPage] = useState(1);
  const [storePage, setStorePage] = useState(1);
  const [runPage, setRunPage] = useState(1);
  const statusAnchorRef = useRef(null);
  const runBusyRef = useRef(false);
  const resetBusyRef = useRef(false);
  const [statusFocusRequest, setStatusFocusRequest] = useState(0);
  const pipelineRows = pipelines ?? [];
  const pl = paginate(pipelineRows, plPage, PAGE_SIZE);
  // Model Store 는 상수 이력(MODEL_STORE)으로 시작하고 백엔드가 확인해 준 행만 source="api" 가 된다.
  // 데모 OFF에서는 확인된 행만 남긴다 — 표 전체를 CSS로 가리던 예전 방식은 실데이터 행까지 지웠다.
  const visibleStore = allowSeed ? modelStore : modelStore.filter((m) => m.source === "api");
  const store = paginate(visibleStore, storePage, PAGE_SIZE);
  const runList = paginate(runs, runPage, PAGE_SIZE);

  // 현재(또는 선택한) 실행의 저장 레코드 — 실행 ID·시각·단계는 프런트 생성값이 아니라 이 값을 쓴다.
  const activeRun = runs.find((r) => r.run_id === (pipelineResult?.run_id ?? selectedRunId)) ?? null;
  // ponytail: pipeline_id 가 없는 실행(공용 /orchestration/events 경로)은 모델로 짝짓는다.
  // AppStateContext 가 pipeline_id 를 함께 보내면 이 fallback 은 지워도 된다(보고서 변경 요청 참조).
  const lastRunOf = (pipeline) =>
    runs.find((r) => r.pipeline_id === pipeline.id || (!r.pipeline_id && r.model_id === pipeline.model_id)) ?? null;

  // 잠금은 native disabled 대신 aria-disabled로 건다. disabled를 걸면 자기 활성화로 잠기는 순간
  // 브라우저가 초점을 body로 떨어뜨린다(§9 A11Y-01). 실행 차단은 핸들러 가드가 담당한다.
  // ref 가드: 같은 틱의 연타는 state 갱신 전이라 resetLocked로 막을 수 없다(초기화 로그 중복 방지).
  // 진행 중에도 초기화는 열어 둔다 — resetPipeline이 타이머·요청 세대를 정리하므로,
  // 실행이 끝나지 않는 상태(응답 없음)에서 새로고침 말고 빠져나갈 길이 여기뿐이다.
  const resetLocked = !pipelineRun;
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

  // 실데이터 행 [실행] — 하단 실데이터 학습 패널의 [학습 실행]과 같은 API를 호출한다.
  // 중복 실행 방지는 백엔드가 담당한다(모델별 활성 job이 있으면 409 JobConflict).
  const handleRealdataRun = async (pipeline) => {
    if (realdataRunBusyRef.current || !realdataToken) return;
    if (!pipeline.dataset_id) {
      addConsoleLog(
        `WARN: ${pipeline.name} 실행 불가 — 사용할 스냅샷이 없습니다. [실데이터 학습] 패널에서 스냅샷을 먼저 생성하세요.`,
        false,
        true
      );
      return;
    }
    realdataRunBusyRef.current = true;
    setRealdataBusyModel(pipeline.model_id);
    try {
      const started = await startTrainingRun(realdataToken, pipeline.model_id, pipeline.dataset_id);
      const jobId = started.data.job_id;
      addConsoleLog(`INFO: 실데이터 학습 실행 — ${jobId} (${pipeline.name} · 데이터셋 ${pipeline.dataset_id})`);
      setSelectedRunId(jobId);
      await reloadCatalog();
      // job은 백그라운드 스레드에서 돈다 — 종결 상태까지 폴링한 뒤 이력·로그를 다시 읽는다.
      for (let attempt = 0; attempt < JOB_POLL_LIMIT; attempt += 1) {
        await new Promise((resolve) => setTimeout(resolve, JOB_POLL_MS));
        const polled = await getTrainingRun(realdataToken, jobId);
        if (!JOB_TERMINAL_STATES.has(polled.data?.state)) continue;
        addConsoleLog(
          polled.data.error
            ? `WARN: 실데이터 학습 종료 — ${jobId} · ${polled.data.state} (${polled.data.error})`
            : `INFO: 실데이터 학습 완료 — ${jobId} · 후보 ${polled.data.candidate_version}`,
          false,
          Boolean(polled.data.error)
        );
        break;
      }
    } catch (err) {
      addConsoleLog(
        err?.status === 409
          ? `WARN: ${pipeline.name} 실행 거부 — 이미 활성 학습 작업이 있습니다.`
          : `WARN: 실데이터 학습 실행 실패 — ${err?.message ?? "알 수 없는 오류"}`,
        false,
        true
      );
    } finally {
      realdataRunBusyRef.current = false;
      setRealdataBusyModel(null);
      await reloadCatalog();
    }
  };

  // 카탈로그 [실행] → 실행 시작 + 아래 실행 상태 카드로 초점·스크롤 (누른 곳에서 결과가 보이도록)
  // ref 가드: 같은 틱의 연타는 state 갱신 전이라 pipelineBusy로 막을 수 없다(중복 오케스트레이션 요청 방지).
  const handleRun = (event, plDef, locked, lockReason) => {
    if (locked || runBusyRef.current) {
      event.preventDefault();
      // 잠긴 버튼을 눌렀는데 아무 흔적도 남지 않으면 "먹통"으로 보인다. 사유는 버튼 title과 같은 문구를 쓴다.
      if (locked) addConsoleLog(`WARN: 파이프라인 실행 불가 — ${lockReason}`, false, true);
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
        className="page-section orchestrator-live"
        headerRight={
          <span style={{ display: "inline-flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
              {pipelines === null ? "불러오는 중" : `${pipelineRows.length}건 등록`} · 드리프트 감지 시 자동 실행
            </span>
            <button
              className="btn btn-primary"
              style={{ padding: "5px 14px", fontSize: 12 }}
              onClick={() => setRegisterOpen((open) => !open)}
              aria-expanded={registerOpen}
            >
              <i className="fa-solid fa-plus" aria-hidden="true"></i> 파이프라인 등록
            </button>
          </span>
        }
      >
        {registerOpen && (
          <PipelineRegisterForm
            models={models}
            onCancel={() => setRegisterOpen(false)}
            onRegistered={(created) => {
              setRegisterOpen(false);
              addConsoleLog(`INFO: 재학습 파이프라인 등록 완료 — ${created.id} (${created.name}) · 대상 모델 ${created.model_id}`);
              reloadCatalog();
            }}
          />
        )}
        {catalogError && (
          <p className="pipeline-empty" role="alert">
            <i className="fa-solid fa-circle-exclamation" aria-hidden="true"></i> 파이프라인 카탈로그를 불러오지
            못했습니다 — {catalogError}
          </p>
        )}
        {pipelines !== null && pipelineRows.length === 0 && !catalogError && (
          <p className="pipeline-empty">
            <i className="fa-solid fa-circle-info" aria-hidden="true"></i> 등록된 재학습 파이프라인이 없습니다.
            [파이프라인 등록]으로 대상 모델과 트리거 조건을 지정해 추가하세요.
          </p>
        )}
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
                // 실데이터 행은 등록 파이프라인이 아니라 실데이터 학습 job의 진입점이다 —
                // 상태·실행 잠금·삭제 가능 여부가 데모 파이프라인과 다르다.
                const isRealdata = p.source === "realdata";
                const isRunning = isRealdata
                  ? realdataBusyModel === p.model_id
                  : pipelineRunning && pipelineRun?.pipelineId === p.id;
                const last = lastRunOf(p);
                // 후보 버전은 백엔드가 현행에서 파생해 내려준다 — 상수 후보와 달리 승급 후에도 잠기지 않는다.
                // 실데이터 학습은 같은 스냅샷으로도 다시 돌릴 수 있어 후보 유무로 잠그지 않는다.
                const candidateAvailable = isRealdata
                  ? Boolean(p.dataset_id)
                  : Boolean(p.candidate_version) && p.candidate_version !== p.base_version;
                const runLocked = isRealdata
                  ? isRunning || !realdataToken || !p.dataset_id
                  : pipelineBusy || !candidateAvailable;
                // 버튼 title = 잠금 사유(또는 실행 안내). 잠긴 클릭의 콘솔 로그도 같은 문구를 쓴다.
                const runTitle = isRealdata
                  ? !p.dataset_id
                    ? "사용할 스냅샷이 없습니다. [실데이터 학습] 패널에서 스냅샷을 먼저 생성하세요."
                    : isRunning
                      ? "이 모델의 실데이터 학습이 실행 중입니다."
                      : `최신 스냅샷(${p.dataset_id})으로 실데이터 학습을 실행합니다 — [실데이터 학습] 패널의 [학습 실행]과 같은 동작입니다.`
                  : !candidateAvailable
                    ? "대상 모델의 다음 후보 버전을 확인할 수 없습니다."
                    : pipelineBusy
                      ? "다른 재학습 파이프라인이 실행 중입니다. 완료 후 실행할 수 있습니다."
                      : `${p.name} 파이프라인을 즉시 실행합니다`;
                return (
                  <tr key={p.id}>
                    <td>
                      <strong style={{ fontSize: 13 }}>{orDash(p.name)}</strong>
                      <div>
                        <code style={{ fontSize: 11, color: "var(--accent-purple-text)" }}>{orDash(p.id)}</code>
                      </div>
                    </td>
                    <td style={{ fontSize: 12 }}>
                      {orDash(p.model_id)}{" "}
                      {isRealdata && !p.base_version ? "· 반영 버전 없음" : orDash(p.base_version)}
                      {isRealdata
                        ? ` · 최신 후보 ${orDash(p.candidate_version)}`
                        : candidateAvailable
                          ? ` → ${orDash(p.candidate_version)}`
                          : " · 다음 후보 미등록"}
                      <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
                        {orDash(p.experiment)}
                        {isRealdata && p.dataset_id ? ` · 스냅샷 ${p.dataset_id}` : ""}
                      </div>
                    </td>
                    <td style={{ fontSize: 12, color: "var(--text-secondary)" }}>{orDash(p.trigger_policy)}</td>
                    <td style={{ fontSize: 11, color: "var(--text-secondary)" }}>
                      {last ? (
                        <>
                          <button
                            className="pipeline-run-link"
                            onClick={() => setSelectedRunId(last.run_id)}
                            title="이 실행의 저장된 로그를 아래에서 봅니다"
                          >
                            {orDash(last.run_id)}
                          </button>
                          <div style={{ color: "var(--text-muted)" }}>
                            {orDash(logTime(last.finished_at))} · {orDash(runStateLabel(last))}
                          </div>
                        </>
                      ) : (
                        "실행 이력 없음"
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
                        {isRunning ? "진행 중" : candidateAvailable ? "대기" : isRealdata ? "스냅샷 필요" : "후보 필요"}
                      </span>
                    </td>
                    <td className="cell-actions">
                      <button
                        className="btn btn-primary"
                        style={{ padding: "5px 14px", fontSize: 12 }}
                        onClick={(event) => {
                          if (!isRealdata) {
                            handleRun(event, toRunDef(p), runLocked, runTitle);
                            return;
                          }
                          event.preventDefault();
                          if (runLocked) {
                            addConsoleLog(`WARN: 파이프라인 실행 불가 — ${runTitle}`, false, true);
                            return;
                          }
                          handleRealdataRun(p);
                        }}
                        aria-disabled={runLocked}
                        title={runTitle}
                        aria-label={`${orDash(p.name)} 파이프라인 실행`}
                      >
                        <i className="fa-solid fa-play"></i> 실행
                      </button>
                      {!isRealdata && (
                        <button
                          className="btn btn-secondary"
                          style={{ padding: "5px 12px", fontSize: 12, marginLeft: 6 }}
                          onClick={() => handleDeletePipeline(p.id)}
                          aria-disabled={pipelineBusy}
                          title="등록을 해제합니다. 실행 이력은 남습니다."
                          aria-label={`${orDash(p.name)} 파이프라인 등록 해제`}
                        >
                          <i className="fa-solid fa-trash" aria-hidden="true"></i>
                        </button>
                      )}
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
          totalCount={pipelineRows.length}
          pageSize={PAGE_SIZE}
          onChange={setPlPage}
        />
      </Card>

      {/* 실데이터 연계(R3) — 남원 실데이터 학습·후보·반영. 위 데모 파이프라인과는 별개 경로. */}
      {/* 선행 조작(스냅샷 생성)이므로 실행 상태·결과 카드보다 위에 둔다. */}
      <RealdataTrainingPanel />

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
              <>파이프라인 실행 상태 — 대기 중</>
            )
          }
          icon="fa-diagram-project"
          className="page-section orchestrator-live"
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
            // 실행 ID·시각·버전은 저장된 실행 레코드(activeRun)를 우선한다 — 프런트 생성값이 아니다.
            <div className="run-meta" data-values-source={activeRun ? "api" : "mock"}>
              <span className="run-meta-item">
                <span className="run-meta-label">대상 모델</span>
                {orDash(activeRun?.model_id ?? seedOr(pipelineRun.model))} {orDash(seedOr(pipelineRun.baseVersion))} →
                후보 {orDash(activeRun?.active_version ?? seedOr(pipelineRun.candidateVersion))}
              </span>
              <span className="run-meta-item">
                <span className="run-meta-label">실행 ID</span>
                <code>{orDash(activeRun?.run_id ?? seedOr(pipelineRun.runId))}</code>
              </span>
              <span className="run-meta-item">
                <span className="run-meta-label">실험</span>
                <code>{orDash(seedOr(pipelineRun.experiment))}</code>
              </span>
              <span className="run-meta-item">
                <span className="run-meta-label">트리거</span>
                {orDash(activeRun?.trigger ?? seedOr(pipelineRun.trigger))}
              </span>
              <span className="run-meta-item">
                <span className="run-meta-label">시작</span>
                {orDash(activeRun ? logTime(activeRun.started_at) : seedOr(pipelineRun.startedAt))}
              </span>
              <span className="run-meta-item">
                <span className="run-meta-label">종료</span>
                {orDash(activeRun?.finished_at ? logTime(activeRun.finished_at) : null)}
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
              // 실행이 끝났으면 저장된 단계 상태로 그린다. 진행 중에는 애니메이션이 진척을 보여준다.
              const status =
                !pipelineBusy && activeRun
                  ? storedNodeStatus(idx, activeRun)
                  : nodeStatus(idx, visibleStep, terminalState);
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

      {/* 승급 완료 후의 다음 단계 — 승급된 모델은 정책 시뮬레이터에서 쓰인다.
          방금 끝난 실행 카드 바로 아래에 붙여 결과와 같은 맥락에서 읽히게 한다. */}
      {terminalState === "succeeded" && (
        <NextStepBanner
          tone="success"
          message={`${pipelineResult?.active_version ?? pipelineRun?.candidateVersion ?? "신규 버전"} 승급 완료 — 운영 모델이 교체되었습니다`}
          actionLabel="정책 시뮬레이터에서 활용"
          targetTab="tab-simulator"
        />
      )}

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
        {/* 전달 경로를 셀 단위로 기록한다. 데모 표시 OFF에서는 표 전체의 데이터를 함께 숨긴다. */}
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
              {store.pageRows.length === 0 && (
                <tr>
                  <td
                    colSpan={7}
                    style={{ fontSize: 12, color: "var(--text-muted)", textAlign: "center", padding: "18px 8px" }}
                  >
                    재학습 후 표시됩니다.
                  </td>
                </tr>
              )}
              {store.pageRows.map((m) => {
                const st = STORE_STATUS_STYLE[m.status] ?? STORE_STATUS_STYLE.이전;
                return (
                  <tr
                    key={`${m.modelId}-${m.version}`}
                    data-values-source={m.source ?? "mock"}
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
                    {/* 학습데이터·하이퍼파라미터·등록일은 응답에 없는 프런트 합성값이라 api 행 안에서도 mock */}
                    <td style={{ fontSize: 12, color: "var(--text-secondary)" }} data-values-source="mock">
                      {orDash(m.dataVersion)}
                    </td>
                    <td style={{ fontSize: 11, color: "var(--text-secondary)" }} data-values-source="mock">
                      {orDash(m.params)}
                    </td>
                    <td
                      className="cell-num"
                      style={{ fontWeight: 600 }}
                      data-values-source={m.accuracySource ?? "mock"}
                    >
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
                    <td style={{ fontSize: 11, color: "var(--text-muted)" }} data-values-source="mock">
                      {orDash(m.registeredAt)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <TablePager
          page={store.safePage}
          totalPages={store.totalPages}
          totalCount={visibleStore.length}
          pageSize={PAGE_SIZE}
          onChange={setStorePage}
        />
      </Card>

      {/* ④ 실행 이력 — 백엔드 SQLite `runs` 테이블. 행을 고르면 그 실행의 저장 로그를 아래에 편다. */}
      <Card
        title="파이프라인 실행 이력"
        icon="fa-clock-rotate-left"
        className="page-section orchestrator-live"
        headerRight={
          <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
            {runs.length}건 · 재학습 실행 레코드 저장소(SQLite)
          </span>
        }
      >
        {runs.length === 0 ? (
          <p className="pipeline-empty">
            <i className="fa-solid fa-circle-info" aria-hidden="true"></i> 저장된 실행 기록이 없습니다. 위
            카탈로그에서 [실행]을 누르면 실행 레코드·단계 상태·로그가 여기에 남습니다.
          </p>
        ) : (
          <>
            <div className="table-container">
              <table>
                <caption className="sr-only">저장된 재학습 실행 이력</caption>
                <thead>
                  <tr>
                    <th scope="col">실행 ID</th>
                    <th scope="col">파이프라인 / 모델</th>
                    <th scope="col">트리거</th>
                    <th scope="col">결과</th>
                    <th scope="col">시작 → 종료</th>
                    <th scope="col" className="cell-actions">동작</th>
                  </tr>
                </thead>
                <tbody>
                  {runList.pageRows.map((r) => (
                    <tr key={r.run_id} className={r.run_id === selectedRunId ? "is-selected-run" : undefined}>
                      <td>
                        <code style={{ fontSize: 11, color: "var(--accent-purple-text)" }}>{orDash(r.run_id)}</code>
                      </td>
                      <td style={{ fontSize: 12 }}>
                        {orDash(r.pipeline_id)}
                        <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
                          {orDash(r.model_id)}
                          {r.dataset_id ? ` · ${r.dataset_id}` : ""}
                        </div>
                      </td>
                      <td style={{ fontSize: 12, color: "var(--text-secondary)" }}>{orDash(r.trigger)}</td>
                      <td style={{ fontSize: 12 }}>
                        {orDash(runStateLabel(r))}
                        <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
                          {orDash(r.active_version)}
                          {/* 실데이터 job은 승급 지표가 아니라 등록된 후보의 검증 지표를 남긴다. */}
                          {r.candidate_metrics?.mae !== undefined && r.candidate_metrics?.mae !== null
                            ? ` · MAE ${fmtMetric(r.candidate_metrics.mae)}`
                            : ""}
                          {r.error ? ` · ${r.error}` : ""}
                        </div>
                      </td>
                      <td style={{ fontSize: 11, color: "var(--text-secondary)" }}>
                        {orDash(logTime(r.started_at))} → {orDash(r.finished_at ? logTime(r.finished_at) : null)}
                      </td>
                      <td className="cell-actions">
                        <button
                          className="btn btn-secondary"
                          style={{ padding: "4px 12px", fontSize: 12 }}
                          onClick={() => setSelectedRunId(r.run_id)}
                          aria-label={`${r.run_id} 실행 로그 보기`}
                        >
                          로그
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <TablePager
              page={runList.safePage}
              totalPages={runList.totalPages}
              totalCount={runs.length}
              pageSize={PAGE_SIZE}
              onChange={setRunPage}
            />
          </>
        )}
      </Card>

      {/* ⑤ 선택한 실행의 저장 로그·단계 상태 — 프런트 배열이 아니라 실행 레코드에 함께 적재된 라인 */}
      <Card
        title={
          <>
            실행 로그
            {selectedRunId ? (
              <code style={{ marginLeft: 8, fontSize: 12, color: "var(--accent-purple-text)", fontWeight: 500 }}>
                {selectedRunId}
              </code>
            ) : null}
          </>
        }
        icon="fa-terminal"
        className="page-section orchestrator-live"
      >
        {runLogs?.logs?.length ? (
          <ConsoleLog
            logs={runLogs.logs.map((entry) => ({
              time: logTime(entry.ts),
              level: entry.level,
              message: entry.message
            }))}
            height={220}
          />
        ) : (
          <p className="pipeline-empty">
            <i className="fa-solid fa-circle-info" aria-hidden="true"></i>{" "}
            {selectedRunId
              ? "이 실행에 저장된 로그 라인이 없습니다."
              : "실행 이력에서 [로그]를 누르거나 파이프라인을 실행하면 저장된 로그가 여기에 표시됩니다."}
          </p>
        )}
      </Card>

      <Card title="UI 활동 로그" icon="fa-list">
        <ConsoleLog logs={consoleLogs} />
      </Card>
    </>
  );
}
