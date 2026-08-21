import { Fragment, useEffect, useRef, useState } from "react";
import Card from "../components/Card.jsx";
import PerfBadge from "../components/PerfBadge.jsx";
import TablePager, { paginate } from "../components/TablePager.jsx";
import PipelineStepper from "../components/PipelineStepper.jsx";
import CollapsibleStage from "../components/CollapsibleStage.jsx";
import ArchiveRegisterForm from "../components/ArchiveRegisterForm.jsx";
import ConfirmDialog from "../components/ConfirmDialog.jsx";
import { useAppState } from "../context/AppStateContext.jsx";
import { apiGet, apiSend } from "../lib/api.js";
import { HTTP_METHODS, AUTH_METHODS, adapterOf, buildQuery } from "../lib/dataopsApi.js";
import { getScrollBehavior } from "../lib/motion.js";

const READY_RESPONSE = `// [POST·GET·PUT·PATCH·DELETE] 요청을 전송하면 표준 REST 응답이 표시됩니다.`;
const CATALOG_URL = "/api/v3/dataops/catalog";
const FILTER_PATTERN = /^(\w+)\s*(>=|<=|!=|=|>|<)\s*('[^';]*'|"[^";]*"|-?\d+(?:\.\d+)?|\w+)$/;

function filterValidationMessage(value, columns = []) {
  const expression = String(value ?? "").trim();
  if (!expression) return null;
  const match = expression.match(FILTER_PATTERN);
  if (!match) return "‘컬럼 연산자 값’ 형식으로 한 조건만 입력하세요. 예: in_flow_count > 100";
  if (!columns.some((column) => column.name === match[1])) {
    return `현재 스키마에 ‘${match[1]}’ 컬럼이 없습니다. 목록에 있는 컬럼명을 사용하세요.`;
  }
  return null;
}

// 카탈로그 선택 → 스키마 검토 → API 호출의 순차 흐름 (시뮬레이터 탭과 동일 패턴)
const DATAOPS_STAGES = [
  { id: "dstep-source", no: "①", label: "데이터 소스 선택", icon: "fa-layer-group" },
  { id: "dstep-schema", no: "②", label: "스키마 확인", icon: "fa-table-columns" },
  { id: "dstep-builder", no: "③", label: "API 빌드·호출", icon: "fa-code" }
];

// HTTP 메서드 칩은 테마별 액센트 토큰을 사용한다.
const METHOD_STYLES = {
  GET: { color: "var(--accent-blue)", bg: "rgba(var(--accent-blue-rgb), 0.02)" },
  POST: { color: "var(--accent-teal)", bg: "rgba(var(--accent-teal-rgb), 0.02)" },
  PUT: { color: "var(--accent-orange)", bg: "rgba(var(--accent-orange-rgb), 0.02)" },
  PATCH: { color: "var(--accent-purple-text)", bg: "rgba(var(--accent-purple-rgb), 0.02)" },
  DELETE: { color: "var(--accent-red)", bg: "rgba(var(--accent-red-rgb), 0.02)" }
};

// 빌드·등록된 API 목록 — "API생성기 + 요청 관리·기록" 명세 반영. 브라우저(localStorage) UI 편의 스냅샷.
const BUILT_APIS_KEY = "decline_poc_built_apis";
const MAX_BUILT_APIS = 12;

function loadStoredList(key) {
  try {
    const arr = JSON.parse(localStorage.getItem(key) ?? "[]");
    return Array.isArray(arr) ? arr : [];
  } catch {
    return [];
  }
}

function persistStoredList(key, list) {
  try {
    localStorage.setItem(key, JSON.stringify(list));
    return true;
  } catch {
    // 저장 불가 환경(시크릿 모드 등)에서는 목록을 세션 한정으로만 유지
    return false;
  }
}

const loadBuiltApis = () => loadStoredList(BUILT_APIS_KEY);
const persistBuiltApis = (list) => persistStoredList(BUILT_APIS_KEY, list);

// ArchiveRegisterForm이 만든 schema를 백엔드 등록 DTO(ArchiveRegisterRequest)로 매핑.
function toRegisterBody(schema) {
  return {
    id: schema.id,
    label: schema.label,
    source: schema.source,
    object: schema.object,
    description: schema.description,
    tier: schema.archive.tier,
    retention: schema.archive.retention,
    tags: schema.tags,
    columns: schema.columns,
    ...(schema.range ? { range: schema.range } : {})
  };
}

// 아카이브 스토리지 티어 칩 색상 (Hot/Warm/Cold)
const TIER_COLORS = {
  Hot: { color: "var(--accent-red)", bg: "rgba(var(--accent-red-rgb), 0.02)" },
  Warm: { color: "var(--accent-orange)", bg: "rgba(var(--accent-orange-rgb), 0.02)" },
  Cold: { color: "var(--accent-blue)", bg: "rgba(var(--accent-blue-rgb), 0.02)" }
};

// DataOps 워크플로우(DAG) 실행 상태 — 한 줄 상태 바 (2차년도 Workflow 관리 기술 기반).
function WorkflowStatus({ workflow }) {
  if (!workflow) return null;
  const fmtAgo = (minAgo) => {
    const d = new Date(Date.now() - minAgo * 60000);
    return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
  };
  const batchTasks = workflow.tasks.filter((t) => t.lastRunMinAgo !== null);
  const lastDone = batchTasks.length
    ? fmtAgo(Math.min(...batchTasks.map((t) => t.lastRunMinAgo)))
    : null;
  return (
    <div className="workflow-statusbar" aria-label="DataOps 워크플로우 실행 상태">
      <span className="workflow-dot" aria-label="정상"></span>
      워크플로우 <code>{workflow.dag_id}</code> 정상 —{" "}
      {batchTasks.map((t) => t.label).join(" → ")} 완료
      {lastDone && ` (최근 ${lastDone})`} · {batchTasks[0]?.schedule ?? ""} 스케줄 · API 상시 제공
    </div>
  );
}

// 메타데이터 가상화 라우팅 단계 — 저장소 유형(RDB/NoSQL)에 따라 쿼리 언어(SQL/MQL)가 달라진다.
function RoutingFlow({ method, source, adapter, queryLang }) {
  const range = source.range;
  const steps = [
    { icon: "fa-paper-plane", title: `${method} 요청`, sub: "API Endpoint" },
    {
      icon: "fa-magnifying-glass",
      title: "메타데이터 검색",
      sub: `${source.source} · ${source.object}${range ? ` · ${range.column} ${range.from}~${range.to}` : ""}`
    },
    { icon: "fa-plug", title: "Adapter 선택", sub: adapter },
    { icon: "fa-database", title: `${queryLang} 생성·실행`, sub: "In-Memory 처리" },
    { icon: "fa-reply", title: "REST 응답", sub: "표준 JSON" }
  ];
  return (
    <div className="routing-flow">
      {steps.map((s, i) => (
        <div key={s.title} className="routing-step-wrap">
          <div className="routing-step">
            <i className={"fa-solid " + s.icon} aria-hidden="true"></i>
            <div className="routing-step-title">{s.title}</div>
            <div className="routing-step-sub">{s.sub}</div>
          </div>
          {i < steps.length - 1 && (
            <i className="fa-solid fa-chevron-right routing-arrow" aria-hidden="true"></i>
          )}
        </div>
      ))}
    </div>
  );
}

export default function DataOpsPage() {
  const { appData, addConsoleLog } = useAppState();
  // 메타데이터 카탈로그 — 백엔드(/api/v3/dataops/catalog)에서 로드. 사용자 등록 소스 병합은 서버가 담당.
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(true);
  const [catalogError, setCatalogError] = useState(null);
  const [showRegForm, setShowRegForm] = useState(false);

  // 카탈로그·스키마·API 빌더가 모두 같은 소스를 바라보도록 선택 상태를 단일화.
  const [sourceId, setSourceId] = useState(null);
  const [method, setMethod] = useState("GET");
  const [filterText, setFilterText] = useState("");
  const [filterError, setFilterError] = useState(null);
  const filterRef = useRef(null);
  const [sortCol, setSortCol] = useState("");
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);
  const [token, setToken] = useState(null);
  const [authMethod, setAuthMethod] = useState("JWT");
  const [responseText, setResponseText] = useState(READY_RESPONSE);
  const [apiMs, setApiMs] = useState(null);
  const [responseOk, setResponseOk] = useState(false);
  const [builtApis, setBuiltApis] = useState(loadBuiltApis);
  const [builtStorageAvailable, setBuiltStorageAvailable] = useState(true);
  const [sentAt, setSentAt] = useState(null);
  const [builtResult, setBuiltResult] = useState(null);
  const [catalogQuery, setCatalogQuery] = useState("");
  const [catalogSort, setCatalogSort] = useState("name");
  const [catalogPage, setCatalogPage] = useState(1);
  const [pendingAction, setPendingAction] = useState(null);
  const [asyncFeedback, setAsyncFeedback] = useState(null);
  const [confirmAction, setConfirmAction] = useState(null);
  const [confirmBusy, setConfirmBusy] = useState(false);
  const [registrationSubmitting, setRegistrationSubmitting] = useState(false);
  const BUILT_PAGE_SIZE = 10;
  const [builtPage, setBuiltPage] = useState(1);

  // 카탈로그 최초 로드
  useEffect(() => {
    let alive = true;
    setCatalogError(null);
    apiGet(CATALOG_URL)
      .then((list) => {
        if (!alive) return;
        setSources(list);
        setCatalogError(null);
        setSourceId((cur) => cur ?? list[0]?.id ?? null);
        setLoading(false);
      })
      .catch((err) => {
        if (!alive) return;
        setCatalogError(err?.message ?? "카탈로그 서버에 연결할 수 없습니다.");
        addConsoleLog(`ERROR: 카탈로그 로드 실패 — ${err.message}`);
        setLoading(false);
      });
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const refreshCatalog = async () => {
    const list = await apiGet(CATALOG_URL);
    setSources(list);
    return list;
  };

  const retryCatalog = async () => {
    setLoading(true);
    setCatalogError(null);
    try {
      const list = await refreshCatalog();
      setSourceId((current) => current ?? list[0]?.id ?? null);
    } catch (err) {
      setCatalogError(err?.message ?? "카탈로그 서버에 연결할 수 없습니다.");
      addConsoleLog(`ERROR: 카탈로그 다시 불러오기 실패 — ${err?.message ?? "알 수 없는 오류"}`);
    } finally {
      setLoading(false);
    }
  };

  // 단계 접기/펼치기 + 스텝퍼 내비게이션
  const [openStages, setOpenStages] = useState({
    "dstep-source": true,
    "dstep-schema": true,
    "dstep-builder": true
  });
  const toggleStage = (id) => setOpenStages((s) => ({ ...s, [id]: !s[id] }));
  const allOpen = Object.values(openStages).every(Boolean);
  const setAllStages = (open) =>
    setOpenStages({ "dstep-source": open, "dstep-schema": open, "dstep-builder": open });
  const jumpToStage = (id) => {
    setOpenStages((s) => ({ ...s, [id]: true }));
    setTimeout(() => {
      document.getElementById(id)?.scrollIntoView({ behavior: getScrollBehavior(), block: "start" });
    }, 60);
  };

  // 스크롤 스파이 — 카탈로그 로드 후 단계가 렌더되면 재부착 (deps: loading)
  const [activeStageId, setActiveStageId] = useState("dstep-source");
  useEffect(() => {
    if (loading) return undefined;
    const ids = DATAOPS_STAGES.map((s) => s.id);
    const visibility = new Map();
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => visibility.set(e.target.id, e.isIntersecting));
        const active = ids.find((id) => visibility.get(id));
        if (active) setActiveStageId(active);
      },
      { rootMargin: "-10% 0px -60% 0px" }
    );
    ids.forEach((id) => {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    });
    return () => observer.disconnect();
  }, [loading]);

  const handleSelectSource = (id) => {
    setSourceId(id);
    setSortCol("");
    setFilterText("");
    setFilterError(null);
    setResponseOk(false);
    setResponseText(READY_RESPONSE);
    setApiMs(null);
    setSentAt(null);
  };

  // 신규 아카이브 등록 — 백엔드 카탈로그 CRUD로 서버 영속화 (localStorage 제거)
  const handleRegisterSource = async (schema) => {
    try {
      await apiSend("POST", CATALOG_URL, { body: toRegisterBody(schema) });
      await refreshCatalog();
      setShowRegForm(false);
      handleSelectSource(schema.id);
      addConsoleLog(
        `INFO: 메타데이터 등록·적재 완료 — ${schema.label} (${schema.source}, ${schema.archive.tier} 티어, ${schema.archive.retention}) → /api/v3/dataops/${schema.id} 가상화 제공 시작`
      );
      setAsyncFeedback({ tone: "success", message: `${schema.label} 아카이브를 등록했습니다.` });
    } catch (err) {
      addConsoleLog(`WARN: 아카이브 등록 실패 — ${err.message}`, false, true);
      throw err;
    }
  };

  const handleDeleteSource = async (id) => {
    try {
      await apiSend("DELETE", `${CATALOG_URL}/${id}`, {});
      const list = await refreshCatalog();
      if (sourceId === id) handleSelectSource(list[0]?.id ?? null);
      addConsoleLog(`WARN: 사용자 등록 아카이브 삭제 — ${id} (메타데이터·가상화 API 제공 중지)`, false, true);
      setAsyncFeedback({ tone: "success", message: "아카이브 등록을 해제하고 API 제공을 중지했습니다." });
    } catch (err) {
      addConsoleLog(`WARN: 아카이브 삭제 실패 — ${err.message}`, false, true);
      setAsyncFeedback({ tone: "error", message: `아카이브를 삭제하지 못했습니다. ${err.message}` });
      throw err;
    }
  };

  const handleIssueToken = async () => {
    if (pendingAction) return;
    setPendingAction("token");
    setAsyncFeedback({ tone: "pending", message: `${authMethod} 토큰을 발급하는 중입니다.` });
    try {
      const path = authMethod === "OAuth2" ? `/api/v3/dataops/oauth2/${sourceId}` : `/api/v3/dataops/token/${sourceId}`;
      const res = await apiSend("POST", path, {});
      if (authMethod === "OAuth2") {
        setToken(res.access_token);
        addConsoleLog(
          `INFO: OAuth2 토큰 발급 완료 (grant_type=${res.grant_type}, code=${res.authorization_code}, token_type=Bearer, expires_in=${res.expires_in}).`
        );
      } else {
        setToken(res.access_token);
        addConsoleLog(`INFO: JWT 토큰 발급 완료 (scope: ${res.scope}, exp: 1h).`);
      }
      setAsyncFeedback({ tone: "success", message: `${authMethod} 토큰을 발급했습니다.` });
    } catch (err) {
      addConsoleLog(`WARN: 토큰 발급 실패 — ${err.message}`, false, true);
      setAsyncFeedback({ tone: "error", message: `토큰을 발급하지 못했습니다. ${err.message}` });
    } finally {
      setPendingAction(null);
    }
  };

  // 인증·라우팅·쿼리·응답을 백엔드에 위임하는 공용 경로 — 빌더 [전송]과 목록 [호출]이 결과 표시만 달리한다
  const executeRequest = async (cfg) => {
    const path = `/api/v3/dataops/${cfg.source.id}`;
    try {
      let body;
      if (cfg.method === "GET") {
        body = await apiGet(path, {
          token,
          params: { filter: cfg.filter, sort: cfg.sort, page: cfg.page, page_size: cfg.pageSize }
        });
      } else if (cfg.method === "DELETE") {
        body = await apiSend("DELETE", path, { token, params: { filter: cfg.filter } });
      } else if (cfg.method === "POST") {
        body = await apiSend("POST", path, { token, body: { data: {} } });
      } else {
        body = await apiSend(cfg.method, path, { token, params: { filter: cfg.filter }, body: { data: {} } });
      }
      return { ok: true, body };
    } catch (err) {
      return { ok: false, body: err.body ?? { status: err.status ?? 500, error: "RequestFailed", message: err.message } };
    }
  };

  const logRequestResult = (cfg, result, elapsed) => {
    if (result.ok) {
      addConsoleLog(
        `INFO: DataOps ${cfg.method} 성공 (${result.body.status}) - /api/v3/dataops/${cfg.source.id} (${elapsed.toFixed(0)}ms, 아카이브 ${cfg.source.archive?.tier ?? "-"} 티어 경유)`
      );
    } else {
      addConsoleLog(`WARN: ${cfg.method} 요청 실패 (${result.body.status ?? "-"}) - ${result.body.message ?? "오류"}`, false, true);
    }
  };

  // 빌더 [전송] — 우측 응답 카드에 표시
  const validateBuilderFilter = ({ focus = false } = {}) => {
    const message = filterValidationMessage(filterText, target?.columns);
    setFilterError(message);
    if (message && focus) requestAnimationFrame(() => filterRef.current?.focus());
    return !message;
  };

  const runApi = async () => {
    if (pendingAction) return;
    if (!validateBuilderFilter({ focus: true })) return;
    const cfg = { method, source: target, filter: filterText.trim(), sort: sortCol, page, pageSize };
    const start = performance.now();
    setPendingAction("builder-request");
    setAsyncFeedback({ tone: "pending", message: `${cfg.method} 요청을 전송하는 중입니다.` });
    setApiMs(null);
    setResponseText(`Sending ${cfg.method} request...`);
    setSentAt(new Date().toLocaleTimeString("ko-KR", { hour12: false }));

    const result = await executeRequest(cfg);
    const elapsed = performance.now() - start;
    setApiMs(elapsed);
    setResponseText(JSON.stringify(result.body, null, 2));
    setResponseOk(result.ok);
    logRequestResult(cfg, result, elapsed);
    setAsyncFeedback({
      tone: result.ok ? "success" : "error",
      message: result.ok ? `${cfg.method} 요청을 완료했습니다.` : `${cfg.method} 요청에 실패했습니다.`
    });
    setPendingAction(null);
  };

  const handleRunApi = () => {
    if (pendingAction) return;
    if (!validateBuilderFilter({ focus: true })) return;
    if (method !== "DELETE") {
      runApi();
      return;
    }
    const deleteScope = filterText.trim()
      ? `필터 “${filterText.trim()}”에 해당하는 데이터`
      : "필터가 없어 원천의 전체 데이터";
    setConfirmAction({
      kind: "builder-request",
      title: "DELETE 요청을 전송할까요?",
      description: `${target.label}(${target.object})에서 ${deleteScope}를 삭제합니다. 이 요청은 되돌릴 수 없습니다.`,
      confirmLabel: "DELETE 요청 전송"
    });
  };

  // 현재 구성을 API 자산으로 빌드·등록 (localStorage UI 스냅샷) — 동일 구성은 갱신
  const handleBuildApi = () => {
    if (!validateBuilderFilter({ focus: true })) return;
    const sig = [method, target.id, filterText.trim(), sortCol, page, pageSize].join("|");
    const entry = {
      id: `api_${Date.now().toString(36)}`,
      sig,
      method,
      sourceId: target.id,
      sourceLabel: target.label,
      endpoint: `/api/v3/dataops/${target.id}`,
      filter: filterText.trim(),
      sort: sortCol,
      page,
      pageSize,
      authMethod,
      createdAt: new Date().toLocaleString("ko-KR", { hour12: false })
    };
    const next = [entry, ...builtApis.filter((a) => a.sig !== sig)].slice(0, MAX_BUILT_APIS);
    const persisted = persistBuiltApis(next);
    setBuiltApis(next);
    setBuiltStorageAvailable(persisted);
    addConsoleLog(
      `INFO: Data API 빌드·등록 — ${method} ${entry.endpoint}${entry.filter ? ` (filter: ${entry.filter})` : ""}`
    );
    setAsyncFeedback({
      tone: persisted ? "success" : "error",
      message: persisted
        ? `${method} ${entry.endpoint} 구성을 발급 목록에 저장했습니다.`
        : `${method} ${entry.endpoint} 구성은 현재 세션에만 유지됩니다. 브라우저 저장공간을 확인하세요.`
    });
  };

  // 등록된 API [호출] — 빌더 상태를 건드리지 않고 스냅샷 그대로 실행, 결과는 해당 행 아래 인라인 표시
  const handleInvokeBuilt = async (api) => {
    if (pendingAction) return;
    const source = sources.find((s) => s.id === api.sourceId);
    if (!source) {
      const message = `원천 데이터 소스(${api.sourceId})를 찾을 수 없습니다. 이 API를 삭제하거나 소스를 다시 등록하세요.`;
      addConsoleLog(`WARN: 등록 API 호출 실패 — ${message}`);
      setAsyncFeedback({ tone: "error", message });
      return;
    }
    const storedFilterError = filterValidationMessage(api.filter, source.columns);
    if (storedFilterError) {
      const message = `저장된 필터가 유효하지 않습니다. ${storedFilterError} API 구성을 삭제하고 다시 빌드하세요.`;
      setAsyncFeedback({ tone: "error", message });
      addConsoleLog(`WARN: 등록 API 호출 실패 — ${message}`);
      return;
    }
    const cfg = { method: api.method, source, filter: api.filter, sort: api.sort, page: api.page, pageSize: api.pageSize };
    const start = performance.now();
    setPendingAction(`invoke-${api.id}`);
    setAsyncFeedback({ tone: "pending", message: `${api.method} ${api.endpoint} 호출 중입니다.` });
    setBuiltResult({ apiId: api.id, text: `Sending ${api.method} request...`, ms: null, time: null, ok: false });

    const result = await executeRequest(cfg);
    const elapsed = performance.now() - start;
    setBuiltResult({
      apiId: api.id,
      text: JSON.stringify(result.body, null, 2),
      ms: elapsed,
      time: new Date().toLocaleTimeString("ko-KR", { hour12: false }),
      ok: result.ok
    });
    if (result.ok) setResponseOk(true);
    logRequestResult(cfg, result, elapsed);
    setAsyncFeedback({
      tone: result.ok ? "success" : "error",
      message: result.ok ? `${api.endpoint} 호출을 완료했습니다.` : `${api.endpoint} 호출에 실패했습니다.`
    });
    setPendingAction(null);
  };

  const requestInvokeBuilt = (api) => {
    if (pendingAction) return;
    if (api.method !== "DELETE") {
      handleInvokeBuilt(api);
      return;
    }
    setConfirmAction({
      kind: "built-request",
      api,
      title: "저장된 DELETE 요청을 실행할까요?",
      description: api.filter
        ? `${api.sourceLabel}의 ${api.endpoint}에서 필터 “${api.filter}”에 해당하는 데이터를 삭제합니다.`
        : `${api.sourceLabel}의 ${api.endpoint}에서 필터가 없어 전체 데이터를 삭제합니다. 이 요청은 되돌릴 수 없습니다.`,
      confirmLabel: "DELETE 요청 실행"
    });
  };

  const handleDeleteBuilt = (id) => {
    const next = builtApis.filter((a) => a.id !== id);
    const persisted = persistBuiltApis(next);
    setBuiltApis(next);
    setBuiltStorageAvailable(persisted);
    setBuiltResult((r) => (r?.apiId === id ? null : r));
    setAsyncFeedback({
      tone: persisted ? "success" : "error",
      message: persisted
        ? "발급 API를 목록에서 삭제했습니다."
        : "현재 화면에서는 API를 삭제했지만 브라우저 저장소를 갱신하지 못했습니다. 새로고침하면 다시 나타날 수 있습니다."
    });
  };

  const requestDeleteSource = (source) => {
    if (pendingAction) return;
    const rowIndex = catalogSources.findIndex((item) => item.id === source.id);
    const candidates = [
      ...catalogSources.slice(rowIndex + 1),
      ...catalogSources.slice(0, rowIndex).reverse()
    ];
    const next = candidates.find((item) => item.user_registered);
    const nextIndex = next ? catalogSources.findIndex((item) => item.id === next.id) : -1;
    const nextIndexAfterDelete = nextIndex > rowIndex ? nextIndex - 1 : nextIndex;
    setConfirmAction({
      kind: "source",
      source,
      title: "아카이브 등록을 해제할까요?",
      description: `${source.label}의 메타데이터를 삭제하고 가상화 API 제공을 중지합니다. 원본 저장소 데이터는 삭제하지 않습니다.`,
      confirmLabel: "아카이브 등록 해제",
      nextFocusSelector: next ? `[data-source-delete="${next.id}"]` : "#dataops-catalog-title",
      fallbackFocusSelector: "#dataops-catalog-title",
      nextPage: nextIndexAfterDelete >= 0 ? Math.floor(nextIndexAfterDelete / 10) + 1 : 1
    });
  };

  const requestDeleteBuilt = (api, rowIndex) => {
    if (pendingAction) return;
    const absoluteIndex = builtApis.findIndex((item) => item.id === api.id);
    const candidates = [
      ...builtApis.slice(absoluteIndex + 1),
      ...builtApis.slice(0, absoluteIndex).reverse()
    ];
    const next = candidates[0];
    const nextIndex = next ? builtApis.findIndex((item) => item.id === next.id) : -1;
    const nextIndexAfterDelete = nextIndex > absoluteIndex ? nextIndex - 1 : nextIndex;
    setConfirmAction({
      kind: "built-api",
      api,
      title: "발급 API를 삭제할까요?",
      description: `${api.method} ${api.endpoint} 구성을 브라우저의 발급 목록에서 삭제합니다. 원본 데이터는 변경하지 않습니다.`,
      confirmLabel: "API 삭제",
      nextFocusSelector: next ? `[data-built-delete="${next.id}"]` : "#dataops-built-title",
      fallbackFocusSelector: "#dataops-built-title",
      nextPage: nextIndexAfterDelete >= 0 ? Math.floor(nextIndexAfterDelete / BUILT_PAGE_SIZE) + 1 : 1
    });
  };

  const executeConfirmedAction = async () => {
    const action = confirmAction;
    if (!action || confirmBusy) return;
    setConfirmBusy(true);
    try {
      if (action.kind === "source") {
        await handleDeleteSource(action.source.id);
        setCatalogPage(action.nextPage);
      }
      if (action.kind === "built-api") {
        handleDeleteBuilt(action.api.id);
        setBuiltPage(action.nextPage);
      }
      if (action.kind === "builder-request") await runApi();
      if (action.kind === "built-request") await handleInvokeBuilt(action.api);
      setConfirmAction(null);
    } catch {
      setConfirmAction(null);
    } finally {
      setConfirmBusy(false);
    }
  };

  if (loading) {
    return (
      <Card>
        <p role="status" aria-live="polite" style={{ fontSize: 13, color: "var(--text-secondary)" }}>
          <i className="fa-solid fa-spinner fa-spin" aria-hidden="true"></i> 메타데이터 카탈로그를 불러오는 중…
        </p>
      </Card>
    );
  }

  if (catalogError) {
    return (
      <Card title="메타데이터 카탈로그">
        <div className="empty-state" role="alert">
          <i className="fa-solid fa-triangle-exclamation" aria-hidden="true"></i>
          <p>카탈로그를 불러오지 못했습니다. {catalogError}</p>
          <button type="button" className="btn btn-primary" onClick={retryCatalog}>
            <i className="fa-solid fa-rotate" aria-hidden="true"></i> 다시 시도
          </button>
        </div>
      </Card>
    );
  }

  const target = sources.find((s) => s.id === sourceId) ?? sources[0];
  if (!target) {
    return (
      <>
        <p className="dataops-page-sub">
          <i className="fa-solid fa-box-archive" aria-hidden="true"></i> 데이터 라이프사이클 관리(DataOps)
        </p>
        <Card title="메타데이터 카탈로그" titleId="dataops-catalog-title" titleTabIndex={-1}>
          <div className="empty-state">
            <i className="fa-solid fa-box-open" aria-hidden="true"></i>
            <p>등록된 데이터 소스가 없습니다. 아카이브 메타데이터를 등록해 시작하세요.</p>
            {!showRegForm && (
              <button type="button" className="btn btn-primary" onClick={() => setShowRegForm(true)}>
                <i className="fa-solid fa-plus" aria-hidden="true"></i> 신규 아카이브 등록
              </button>
            )}
          </div>
          {showRegForm && (
            <ArchiveRegisterForm
              onRegister={handleRegisterSource}
              onCancel={() => setShowRegForm(false)}
              onSubmittingChange={setRegistrationSubmitting}
            />
          )}
        </Card>
      </>
    );
  }

  const adapter = adapterOf(target);
  const generatedQuery = buildQuery({
    method,
    schema: target,
    filter: filterValidationMessage(filterText, target.columns) ? "" : filterText.trim(),
    sort: sortCol,
    page,
    pageSize
  });

  // 카탈로그 검색 — 소스명·태그·설명·객체명 부분 일치
  const q = catalogQuery.trim().toLowerCase();
  const filteredSources = q
    ? sources.filter((s) =>
        [s.label, s.description, s.object, ...(s.tags ?? [])].join(" ").toLowerCase().includes(q)
      )
    : sources;
  const catalogSources = [...filteredSources].sort((a, b) => {
    if (catalogSort === "loaded") {
      return String(b.archive?.loaded_at ?? "").localeCompare(String(a.archive?.loaded_at ?? ""), "ko");
    }
    if (catalogSort === "tier") {
      return String(a.archive?.tier ?? "").localeCompare(String(b.archive?.tier ?? ""), "ko");
    }
    return a.label.localeCompare(b.label, "ko");
  });
  const catalogPg = paginate(catalogSources, catalogPage, 10);

  const builtPg = paginate(builtApis, builtPage, BUILT_PAGE_SIZE);
  const doneStages = ["dstep-source", "dstep-schema", ...(responseOk ? ["dstep-builder"] : [])];

  return (
    <>
      <p className="dataops-page-sub">
        <i className="fa-solid fa-box-archive" aria-hidden="true"></i> 데이터 라이프사이클 관리
        기술(DataOps) — 빅데이터 관리 아카이빙 · 메타데이터 기반 다기종 데이터 관리
      </p>

      {asyncFeedback && (
        <p
          className={`async-feedback is-${asyncFeedback.tone}`}
          role={asyncFeedback.tone === "error" ? "alert" : "status"}
          aria-live={asyncFeedback.tone === "error" ? "assertive" : "polite"}
        >
          <i
            className={`fa-solid ${
              asyncFeedback.tone === "pending"
                ? "fa-spinner fa-spin"
                : asyncFeedback.tone === "error"
                  ? "fa-circle-exclamation"
                  : "fa-circle-check"
            }`}
            aria-hidden="true"
          ></i>{" "}
          {asyncFeedback.message}
        </p>
      )}

      <div className="pl-toolbar">
        <PipelineStepper
          stages={DATAOPS_STAGES}
          activeId={activeStageId}
          doneIds={doneStages}
          onJump={jumpToStage}
          ariaLabel="DataOps API 발급 단계"
        />
        <button type="button" className="pl-collapse-all" onClick={() => setAllStages(!allOpen)}>
          <i className={`fa-solid ${allOpen ? "fa-compress" : "fa-expand"}`} aria-hidden="true"></i>
          {allOpen ? "모두 접기" : "모두 펼치기"}
        </button>
      </div>

      {/* ── STEP ① 데이터 소스 선택 ── */}
      <CollapsibleStage
        id="dstep-source"
        no="STEP ①"
        title="데이터 소스 선택 - 빅데이터 아카이브"
        sub="아카이빙된 다기종 데이터 소스를 메타데이터 카탈로그에서 선택"
        open={openStages["dstep-source"]}
        onToggle={() => toggleStage("dstep-source")}
      >
        <Card
          title="메타데이터 카탈로그"
          titleId="dataops-catalog-title"
          titleTabIndex={-1}
          icon="fa-database"
        >
          <WorkflowStatus workflow={appData.dataops_workflow} />
          <p style={{ fontSize: 13, color: "var(--text-secondary)", marginBottom: 12 }}>
            수집·가공 데이터를 메타데이터 기반으로 아카이빙하고, 물리 저장소를 직접 노출하지 않고
            단일 API로 연계 제공합니다. 소스를 선택하면 STEP ② 스키마와 STEP ③ API 빌더 대상이 함께
            전환됩니다.
          </p>

          <div className="catalog-search-row">
            <i className="fa-solid fa-magnifying-glass" aria-hidden="true"></i>
            <input
              className="input-control"
              placeholder="소스명·태그·설명 검색 (예: 인구이동, 시계열)"
              value={catalogQuery}
              onChange={(e) => {
                setCatalogQuery(e.target.value);
                setCatalogPage(1);
              }}
              aria-label="데이터 소스 검색"
            />
            <label className="compact-select-field catalog-sort-field">
              <span>정렬</span>
              <select
                className="select-control"
                value={catalogSort}
                onChange={(event) => {
                  setCatalogSort(event.target.value);
                  setCatalogPage(1);
                }}
              >
                <option value="name">소스명</option>
                <option value="tier">아카이브 티어</option>
                <option value="loaded">최근 적재일</option>
              </select>
            </label>
            <button
              type="button"
              className={`btn ${showRegForm ? "btn-secondary" : "btn-primary"} catalog-reg-btn`}
              onClick={() => setShowRegForm((v) => !v)}
              aria-expanded={showRegForm}
              disabled={registrationSubmitting}
            >
              <i className={`fa-solid ${showRegForm ? "fa-xmark" : "fa-plus"}`} aria-hidden="true"></i>{" "}
              {showRegForm ? "등록 닫기" : "신규 아카이브 등록"}
            </button>
          </div>

          <div className="catalog-result-status" role="status" aria-live="polite">
            <span>{q ? `전체 ${sources.length}건 중 검색 결과 ${filteredSources.length}건` : `전체 ${sources.length}건`}</span>
            {q && (
              <button type="button" className="btn btn-tertiary" onClick={() => { setCatalogQuery(""); setCatalogPage(1); }}>
                검색 해제
              </button>
            )}
          </div>

          {showRegForm && (
            <ArchiveRegisterForm
              onRegister={handleRegisterSource}
              onCancel={() => setShowRegForm(false)}
              onSubmittingChange={setRegistrationSubmitting}
            />
          )}

          <div className="table-container">
            <table id="dataops-catalog-table" className="catalog-table" tabIndex="-1">
              <caption className="sr-only">메타데이터 카탈로그의 데이터 소스 목록. 선택한 정렬 기준으로 표시합니다.</caption>
              <thead>
                <tr>
                  <th scope="col">데이터 소스</th>
                  <th scope="col">저장소 유형</th>
                  <th scope="col">데이터 객체</th>
                  <th scope="col">수집 범위</th>
                  <th scope="col">아카이브 티어</th>
                  <th scope="col">보존 정책</th>
                  <th scope="col">적재일</th>
                  <th scope="col" className="cell-num">컬럼</th>
                </tr>
              </thead>
              <tbody>
                {catalogPg.pageRows.map((s) => {
                  const isActive = s.id === sourceId;
                  return (
                    <tr
                      key={s.id}
                      className={isActive ? "catalog-row-active" : ""}
                    >
                      <td>
                        <button
                          type="button"
                          className="table-row-action"
                          onClick={() => handleSelectSource(s.id)}
                          aria-pressed={isActive}
                        >
                          {s.label}
                          <span className="sr-only"> API 대상으로 선택</span>
                        </button>
                        {(s.tags ?? []).map((tag) => (
                          <span key={tag} className="catalog-tag-chip">
                            #{tag}
                          </span>
                        ))}
                        {s.user_registered && (
                          <span className="catalog-user-chip" title="사용자가 등록한 아카이브 (서버 보존)">
                            <i className="fa-solid fa-user-pen" aria-hidden="true"></i> 사용자 등록
                          </span>
                        )}
                        {isActive && (
                          <span className="catalog-selected-chip">
                            <i className="fa-solid fa-check" aria-hidden="true"></i> 선택됨
                          </span>
                        )}
                        {s.user_registered && (
                          <button
                            type="button"
                            className="btn btn-secondary catalog-row-del"
                            onClick={() => requestDeleteSource(s)}
                            data-source-delete={s.id}
                            disabled={Boolean(pendingAction)}
                            aria-label={`${s.label} 아카이브 삭제`}
                            title="등록 해제 (메타데이터·API 제공 중지)"
                          >
                            <i className="fa-solid fa-trash-can" aria-hidden="true"></i>
                          </button>
                        )}
                      </td>
                      <td style={{ fontSize: 12, color: "var(--text-secondary)" }}>{s.source}</td>
                      <td>
                        <code style={{ fontSize: 11, color: "var(--accent-purple-text)" }}>{s.object}</code>
                      </td>
                      <td style={{ fontSize: 11, color: "var(--text-secondary)", whiteSpace: "nowrap" }}>
                        {s.range ? (
                          <span title={`Adapter가 쿼리에 자동 주입하는 적재 범위 (${s.range.column})`}>
                            <code style={{ fontSize: 10 }}>{s.range.column}</code> {s.range.from}~{s.range.to}
                          </span>
                        ) : (
                          "–"
                        )}
                      </td>
                      <td>
                        {s.archive ? (
                          <span
                            className="system-status"
                            style={{
                              padding: "1px 8px",
                              fontSize: 10,
                              fontWeight: 700,
                              color: (TIER_COLORS[s.archive.tier] ?? TIER_COLORS.Cold).color,
                              backgroundColor: (TIER_COLORS[s.archive.tier] ?? TIER_COLORS.Cold).bg
                            }}
                          >
                            {s.archive.tier}
                          </span>
                        ) : "–"}
                      </td>
                      <td style={{ fontSize: 12, color: "var(--text-secondary)" }}>
                        {s.archive?.retention ?? "–"}
                      </td>
                      <td style={{ fontSize: 12, color: "var(--text-muted)" }}>
                        {s.archive?.loaded_at ?? "–"}
                      </td>
                      <td className="cell-num" style={{ color: "var(--text-muted)" }}>
                        {(s.columns ?? []).length}
                      </td>
                    </tr>
                  );
                })}
                {filteredSources.length === 0 && (
                  <tr>
                      <td colSpan={8} className="empty-table-cell">
                        <span>“{catalogQuery}” 검색 결과가 없습니다.</span>{" "}
                        <button type="button" className="btn btn-tertiary" onClick={() => { setCatalogQuery(""); setCatalogPage(1); }}>
                          검색 해제
                        </button>
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          <TablePager
            page={catalogPg.safePage}
            totalPages={catalogPg.totalPages}
            totalCount={catalogSources.length}
            pageSize={10}
            onChange={setCatalogPage}
          />

          <div className="stage-next-row">
            <button type="button" className="btn btn-secondary" onClick={() => jumpToStage("dstep-schema")}>
              다음 — {target.label} 스키마 확인 <i className="fa-solid fa-arrow-down" aria-hidden="true"></i>
            </button>
          </div>
        </Card>
      </CollapsibleStage>

      {/* ── STEP ② 스키마 확인 ── */}
      <CollapsibleStage
        id="dstep-schema"
        no="STEP ②"
        title="스키마 확인"
        sub="선택한 소스의 컬럼 구조·데이터 타입 검토"
        open={openStages["dstep-schema"]}
        onToggle={() => toggleStage("dstep-schema")}
      >
        <Card>
          <div style={{ display: "flex", alignItems: "baseline", gap: 8, flexWrap: "wrap" }}>
            <h4 style={{ color: "var(--accent-blue)", margin: 0 }}>{target.label}</h4>
            <span style={{ fontSize: 12, color: "var(--text-muted)" }}>
              {target.source} · {target.object}
            </span>
          </div>
          <p style={{ fontSize: 12, color: "var(--text-secondary)", margin: "6px 0 10px", fontStyle: "italic" }}>
            {target.description}
          </p>

          {target.lineage && (
            <p className="schema-meta-line">
              <i className="fa-solid fa-database" aria-hidden="true"></i> 원천 {target.lineage.origin}{" "}
              · 아카이브 {target.archive?.tier ?? "-"} ·{" "}
              <span title="Git/DVC 기반 데이터 버전 관리">
                <i className="fa-solid fa-code-branch" aria-hidden="true"></i> 데이터 버전{" "}
                {target.lineage.version} <code>({target.lineage.commit})</code>
              </span>
              {target.range && (
                <span title="Adapter가 쿼리에 자동 주입하는 적재 범위">
                  {" "}· <i className="fa-solid fa-arrows-left-right" aria-hidden="true"></i> 수집 범위{" "}
                  <code>{target.range.column}</code> {target.range.from}~{target.range.to}
                </span>
              )}
            </p>
          )}

          <div className="table-container">
            <table>
              <caption className="sr-only">선택한 데이터 소스의 스키마 컬럼 정의</caption>
              <thead>
                <tr>
                  <th scope="col">컬럼명</th>
                  <th scope="col">데이터 타입</th>
                  <th scope="col">설명</th>
                </tr>
              </thead>
              <tbody>
                {target.columns.map((col) => (
                  <tr key={col.name}>
                    <td>
                      <code style={{ color: "var(--accent-purple-text)", fontWeight: 600 }}>{col.name}</code>
                    </td>
                    <td>
                      <span
                        className="system-status"
                        style={{
                          padding: "1px 6px",
                          fontSize: 10,
                          backgroundColor: "rgba(59, 130, 246, 0.08)",
                          color: "var(--accent-blue)"
                        }}
                      >
                        {col.type}
                      </span>
                    </td>
                    <td>{col.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="stage-next-row">
            <button type="button" className="btn btn-secondary" onClick={() => jumpToStage("dstep-builder")}>
              다음 — 이 스키마로 API 빌드 <i className="fa-solid fa-arrow-down" aria-hidden="true"></i>
            </button>
          </div>
        </Card>
      </CollapsibleStage>

      {/* ── STEP ③ API 빌드·호출: 좌(요청 구성) | 우(응답) ── */}
      <CollapsibleStage
        id="dstep-builder"
        no="STEP ③"
        title="Data API 빌드 · 호출"
        sub="표준 SQL 설정 기반 CRUD·필터·정렬·페이징을 In-Memory로 처리 — JWT/OAuth2 인증으로 저장소 비노출"
        open={openStages["dstep-builder"]}
        onToggle={() => toggleStage("dstep-builder")}
      >
        <div className="dataops-builder-grid">
          <Card title="요청 구성" icon="fa-sliders">
            {/* 3-1 인증 */}
            <div className="builder-section">
              <div className="builder-section-label">3-1 · 인증 (JWT / OAuth2)</div>
              <div
                className="auth-box"
                style={{
                  backgroundColor: token ? "rgba(16,185,129,0.05)" : "rgba(239,68,68,0.05)",
                  borderColor: token ? "rgba(16,185,129,0.2)" : "rgba(239,68,68,0.2)"
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8 }}>
                  <span style={{ fontSize: 12, fontWeight: 600 }}>
                    <i className="fa-solid fa-key" style={{ marginRight: 6 }}></i>
                    인증 상태:{" "}
                    <span style={{ color: token ? "var(--accent-teal)" : "var(--accent-red)" }}>
                      {token ? `${authMethod} 인증됨 (Bearer)` : "미인증 (401 반환)"}
                    </span>
                  </span>
                  <div className="auth-controls">
                    <label className="control-field is-compact">
                      <span>인증 방식</span>
                      <select
                        className="select-control"
                        value={authMethod}
                        onChange={(e) => {
                          setAuthMethod(e.target.value);
                          setToken(null);
                        }}
                      >
                        {AUTH_METHODS.map((m) => (
                          <option key={m} value={m}>
                            {m}
                          </option>
                        ))}
                      </select>
                    </label>
                    <button
                      type="button"
                      className="btn btn-secondary"
                      style={{ padding: "6px 12px" }}
                      onClick={handleIssueToken}
                      disabled={Boolean(pendingAction)}
                    >
                      <i className={`fa-solid ${pendingAction === "token" ? "fa-spinner fa-spin" : "fa-fingerprint"}`} aria-hidden="true"></i>{" "}
                      {pendingAction === "token" ? "발급 중" : "토큰 발급"}
                    </button>
                  </div>
                </div>
                {authMethod === "OAuth2" && (
                  <div style={{ marginTop: 8, fontSize: 11, color: "var(--text-muted)", lineHeight: 1.5 }}>
                    <i className="fa-solid fa-circle-info" style={{ marginRight: 4 }}></i>
                    OAuth2 Authorization Code Grant: 인가코드 발급 → access_token(JWT) 교환 흐름.
                  </div>
                )}
                {token && <code className="token-line">Authorization: Bearer {token}</code>}
              </div>
            </div>

            {/* 3-2 요청 구성 */}
            <div className="builder-section">
              <div className="builder-section-label">3-2 · 요청 구성 (CRUD · 필터 · 정렬 · 페이징)</div>
              <div className="builder-control-grid">
                <label className="control-field">
                  <span>HTTP 메서드</span>
                  <select className="select-control" value={method} onChange={(e) => setMethod(e.target.value)}>
                    {HTTP_METHODS.map((m) => (
                      <option key={m} value={m}>{m}</option>
                    ))}
                  </select>
                </label>
                <label className="control-field">
                  <span>대상 데이터 소스</span>
                  <select className="select-control" value={sourceId} onChange={(e) => handleSelectSource(e.target.value)}>
                    {sources.map((s) => (
                      <option key={s.id} value={s.id}>
                        {s.label} — /api/v3/dataops/{s.id}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <div className="builder-control-grid">
                <label className="control-field">
                  <span>필터 조건 (선택)</span>
                  <input
                    ref={filterRef}
                    className="input-control"
                    placeholder="예: in_flow_count > 100"
                    value={filterText}
                    onChange={(e) => {
                      const nextValue = e.target.value;
                      setFilterText(nextValue);
                      if (filterError) setFilterError(filterValidationMessage(nextValue, target.columns));
                    }}
                    onBlur={() => validateBuilderFilter()}
                    aria-invalid={Boolean(filterError)}
                    aria-describedby={filterError ? "dataops-filter-error" : undefined}
                  />
                  {filterError && (
                    <span id="dataops-filter-error" className="field-error">
                      {filterError}
                    </span>
                  )}
                </label>
                <label className="control-field">
                  <span>정렬 컬럼 (선택)</span>
                  <select className="select-control" value={sortCol} onChange={(e) => setSortCol(e.target.value)}>
                    <option value="">정렬 없음</option>
                    {target.columns.map((c) => (
                      <option key={c.name} value={c.name}>sort: {c.name}</option>
                    ))}
                  </select>
                </label>
                <label className="field-inline">
                  <span>page</span>
                  <input
                    className="input-control"
                    type="number"
                    min="1"
                    value={page}
                    onChange={(e) => setPage(Math.max(1, parseInt(e.target.value, 10) || 1))}
                    aria-label="페이지 번호"
                  />
                </label>
                <label className="field-inline">
                  <span>page_size</span>
                  <input
                    className="input-control"
                    type="number"
                    min="1"
                    max="200"
                    value={pageSize}
                    onChange={(e) => setPageSize(Math.min(200, Math.max(1, parseInt(e.target.value, 10) || 1)))}
                    aria-label="페이지 크기"
                  />
                </label>
              </div>
            </div>

            {/* 3-3 가상화 라우팅 + 저장소별 쿼리(SQL/MQL) */}
            <div className="builder-section">
              <div className="builder-section-label">
                3-3 · 메타데이터 가상화 라우팅
                <span className={`query-lang-chip ${generatedQuery.lang === "MQL" ? "is-mql" : ""}`}>
                  {generatedQuery.lang === "MQL" ? "MQL · MongoDB" : "SQL · RDB"}
                </span>
              </div>
              <RoutingFlow method={method} source={target} adapter={adapter} queryLang={generatedQuery.lang} />
              <pre className="sql-preview">{generatedQuery.text}</pre>
            </div>

            <div className="builder-action-row">
              <button className="btn btn-secondary" onClick={handleBuildApi}>
                <i className="fa-solid fa-hammer"></i> API 빌드·등록
              </button>
              <button type="button" className="btn btn-primary" onClick={handleRunApi} disabled={Boolean(pendingAction)}>
                <i className={`fa-solid ${pendingAction === "builder-request" ? "fa-spinner fa-spin" : "fa-paper-plane"}`} aria-hidden="true"></i>{" "}
                {pendingAction === "builder-request" ? "요청 중" : `${method} /api/v3/dataops/${target.id} 전송`}
              </button>
            </div>
          </Card>

          <Card
            title="REST API JSON 응답"
            icon="fa-reply"
            className="dataops-resp-card"
            headerRight={
              <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
                {sentAt && (
                  <span className="resp-origin-chip">
                    <i className="fa-solid fa-clock" aria-hidden="true"></i> 전송 {sentAt}
                  </span>
                )}
                <PerfBadge ms={apiMs} label="API 응답" />
              </span>
            }
          >
            <pre className="api-response" aria-live="polite" aria-busy={pendingAction === "builder-request"}>{responseText}</pre>
          </Card>
        </div>

        {/* 빌드된 API 자산 목록 — API생성기의 생성·관리·기록 흐름 */}
        <Card
          title="발급된 API 목록"
          titleId="dataops-built-title"
          titleTabIndex={-1}
          icon="fa-list-check"
          className="dataops-built-card"
          headerRight={
            <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
              {builtApis.length}건 · {builtStorageAvailable ? "브라우저에 보존" : "현재 세션에만 유지"}
            </span>
          }
        >
          {builtApis.length === 0 ? (
            <p style={{ fontSize: 12, color: "var(--text-muted)", margin: 0 }}>
              아직 빌드된 API가 없습니다. 요청을 구성한 뒤 <strong>[API 빌드·등록]</strong>을 누르면
              발급된 API가 이 목록에 보존되고, [호출]로 언제든 재실행할 수 있습니다.
            </p>
          ) : (
            <div className="table-container">
              <table id="dataops-built-table" tabIndex="-1">
                <caption className="sr-only">빌드하여 등록한 DataOps API 목록</caption>
                <thead>
                  <tr>
                    <th scope="col">메서드</th>
                    <th scope="col">엔드포인트</th>
                    <th scope="col">대상 소스</th>
                    <th scope="col">쿼리</th>
                    <th scope="col">인증</th>
                    <th scope="col">빌드 일시</th>
                    <th scope="col">상태</th>
                    <th scope="col" className="cell-actions">동작</th>
                  </tr>
                </thead>
                <tbody>
                  {builtPg.pageRows.map((api, rowIndex) => {
                    const methodStyle = METHOD_STYLES[api.method] ?? METHOD_STYLES.GET;
                    const isInvoked = builtResult?.apiId === api.id;
                    const sourceAvailable = sources.some((source) => source.id === api.sourceId);
                    const querySummary = [
                      api.filter ? `filter: ${api.filter}` : null,
                      api.sort ? `sort: ${api.sort}` : null,
                      `p${api.page} · ${api.pageSize}행`
                    ]
                      .filter(Boolean)
                      .join(" / ");
                    return (
                      <Fragment key={api.id}>
                        <tr className={isInvoked ? "built-row-active" : ""}>
                          <td>
                            <span
                              className="system-status"
                              style={{
                                padding: "1px 8px",
                                fontSize: 10,
                                fontWeight: 700,
                                color: methodStyle.color,
                                backgroundColor: methodStyle.bg,
                                borderColor: methodStyle.color
                              }}
                            >
                              {api.method}
                            </span>
                          </td>
                          <td>
                            <code style={{ fontSize: 11, color: "var(--accent-purple-text)" }}>{api.endpoint}</code>
                          </td>
                          <td style={{ fontSize: 12 }}>{api.sourceLabel}</td>
                          <td style={{ fontSize: 11, color: "var(--text-secondary)" }}>{querySummary}</td>
                          <td style={{ fontSize: 11, color: "var(--text-secondary)" }}>{api.authMethod}</td>
                          <td style={{ fontSize: 11, color: "var(--text-muted)" }}>{api.createdAt}</td>
                          <td>
                            <span
                              className="system-status"
                              style={{
                                padding: "1px 8px",
                                fontSize: 10,
                                color: sourceAvailable ? "var(--accent-teal)" : "var(--accent-red)",
                                backgroundColor: sourceAvailable
                                  ? "rgba(var(--accent-teal-rgb), 0.02)"
                                  : "rgba(var(--accent-red-rgb), 0.02)",
                                borderColor: "currentColor"
                              }}
                            >
                              <i
                                className={`fa-solid ${sourceAvailable ? "fa-circle-check" : "fa-triangle-exclamation"}`}
                                aria-hidden="true"
                              ></i>{" "}
                              {sourceAvailable ? "Active" : "원천 없음"}
                            </span>
                          </td>
                          <td className="cell-actions">
                            <button
                              className="btn btn-secondary"
                              style={{ padding: "4px 10px", fontSize: 11 }}
                              onClick={() => requestInvokeBuilt(api)}
                              disabled={Boolean(pendingAction) || !sourceAvailable}
                              aria-label={`${api.method} ${api.endpoint} 호출`}
                              title={sourceAvailable ? "등록된 구성으로 즉시 호출" : "원천 소스를 다시 등록하거나 이 API를 삭제하세요"}
                            >
                              <i className={`fa-solid ${pendingAction === `invoke-${api.id}` ? "fa-spinner fa-spin" : "fa-play"}`} aria-hidden="true"></i>{" "}
                              {pendingAction === `invoke-${api.id}` ? "호출 중" : "호출"}
                            </button>
                            <button
                              className="btn btn-secondary"
                              style={{ padding: "4px 8px", fontSize: 11, marginLeft: 6 }}
                              onClick={() => requestDeleteBuilt(api, rowIndex)}
                              data-built-delete={api.id}
                              disabled={Boolean(pendingAction)}
                              aria-label={`${api.method} ${api.endpoint} 삭제`}
                              title="목록에서 삭제"
                            >
                              <i className="fa-solid fa-trash-can"></i>
                            </button>
                          </td>
                        </tr>
                        {isInvoked && (
                          <tr className="built-resp-row">
                            <td colSpan={8}>
                              <div className="built-resp-head">
                                <span>
                                  <i className="fa-solid fa-reply" aria-hidden="true"></i> {api.method}{" "}
                                  {api.endpoint} 호출 응답
                                  {builtResult.time ? ` · ${builtResult.time}` : ""}
                                </span>
                                <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
                                  <PerfBadge ms={builtResult.ms} label="API 응답" />
                                  <button
                                    type="button"
                                    className="btn btn-secondary"
                                    style={{ padding: "3px 10px", fontSize: 11 }}
                                    onClick={() => setBuiltResult(null)}
                                  >
                                    닫기
                                  </button>
                                </span>
                              </div>
                              <pre className="api-response built-resp-pre">{builtResult.text}</pre>
                            </td>
                          </tr>
                        )}
                      </Fragment>
                    );
                  })}
                </tbody>
              </table>
              <TablePager
                page={builtPg.safePage}
                totalPages={builtPg.totalPages}
                totalCount={builtApis.length}
                pageSize={BUILT_PAGE_SIZE}
                onChange={setBuiltPage}
              />
            </div>
          )}
        </Card>
      </CollapsibleStage>

      <ConfirmDialog
        open={Boolean(confirmAction)}
        title={confirmAction?.title}
        description={confirmAction?.description}
        confirmLabel={confirmAction?.confirmLabel}
        busy={confirmBusy}
        nextFocusSelector={confirmAction?.nextFocusSelector}
        fallbackFocusSelector={confirmAction?.fallbackFocusSelector}
        onCancel={() => setConfirmAction(null)}
        onConfirm={executeConfirmedAction}
      />
    </>
  );
}
