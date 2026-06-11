import { Fragment, useEffect, useState } from "react";
import Card from "../components/Card.jsx";
import PerfBadge from "../components/PerfBadge.jsx";
import PipelineStepper from "../components/PipelineStepper.jsx";
import CollapsibleStage from "../components/CollapsibleStage.jsx";
import { useAppState } from "../context/AppStateContext.jsx";
import {
  HTTP_METHODS,
  AUTH_METHODS,
  issueMockJwt,
  issueMockOAuth2,
  decodeJwtPayload,
  pickAdapter,
  buildSql,
  buildApiResponse,
  buildUnauthorized
} from "../lib/dataopsApi.js";

const READY_RESPONSE = `// [POST·GET·PUT·PATCH·DELETE] 요청을 전송하면 표준 REST 응답이 표시됩니다.`;

// 카탈로그 선택 → 스키마 검토 → API 호출의 순차 흐름 (시뮬레이터 탭과 동일 패턴)
const DATAOPS_STAGES = [
  { id: "dstep-source", no: "①", label: "데이터 소스 선택", icon: "fa-layer-group" },
  { id: "dstep-schema", no: "②", label: "스키마 확인", icon: "fa-table-columns" },
  { id: "dstep-builder", no: "③", label: "API 빌드·호출", icon: "fa-code" }
];

// HTTP 메서드 칩 색상 (rgb triplet)
const METHOD_COLORS = {
  GET: "59, 130, 246",
  POST: "16, 185, 129",
  PUT: "245, 158, 11",
  PATCH: "139, 92, 246",
  DELETE: "239, 68, 68"
};

// 빌드·등록된 API 목록 — "API생성기 + 요청 관리·기록" 명세 반영. 브라우저(localStorage)에 보존.
const BUILT_APIS_KEY = "decline_poc_built_apis";
const MAX_BUILT_APIS = 12;

function loadBuiltApis() {
  try {
    const arr = JSON.parse(localStorage.getItem(BUILT_APIS_KEY) ?? "[]");
    return Array.isArray(arr) ? arr : [];
  } catch {
    return [];
  }
}

function persistBuiltApis(list) {
  try {
    localStorage.setItem(BUILT_APIS_KEY, JSON.stringify(list));
  } catch {
    // 저장 불가 환경(시크릿 모드 등)에서는 목록을 세션 한정으로만 유지
  }
}

// 아카이브 스토리지 티어 칩 색상 (Hot/Warm/Cold)
const TIER_COLORS = {
  Hot: { color: "var(--accent-red)", bg: "rgba(239, 68, 68, 0.1)" },
  Warm: { color: "var(--accent-yellow, #f59e0b)", bg: "rgba(245, 158, 11, 0.1)" },
  Cold: { color: "var(--accent-blue)", bg: "rgba(59, 130, 246, 0.1)" }
};

// 데이터 라이프사이클 흐름 — "데이터 라이프사이클 관리 기술(DataOps)" 과제 구조 반영.
const LIFECYCLE_STEPS = [
  { icon: "fa-cloud-arrow-down", title: "수집", sub: "다기관·다기종 원천" },
  { icon: "fa-box-archive", title: "적재 (아카이빙)", sub: "티어·보존 정책 관리" },
  { icon: "fa-tags", title: "메타데이터 등록", sub: "카탈로그 가상화" },
  { icon: "fa-plug-circle-bolt", title: "API 제공", sub: "저장소 비노출 연계" }
];

// 칩 한 줄로 압축 표시 — STEP ③의 가상화 라우팅 흐름 띠와 시각적 중복을 피한다.
function LifecycleFlow() {
  return (
    <div className="lifecycle-line" aria-label="데이터 라이프사이클">
      {LIFECYCLE_STEPS.map((s, i) => (
        <span key={s.title} style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          <span className="lifecycle-chip" title={s.sub}>
            <i className={"fa-solid " + s.icon} aria-hidden="true"></i> {s.title}
          </span>
          {i < LIFECYCLE_STEPS.length - 1 && (
            <i className="fa-solid fa-chevron-right lifecycle-arrow" aria-hidden="true"></i>
          )}
        </span>
      ))}
    </div>
  );
}

// 메타데이터 가상화 라우팅 단계 — 이미지 "메타데이터를 이용한 데이터 관리 구조" 반영.
function RoutingFlow({ method, source, adapter }) {
  const steps = [
    { icon: "fa-paper-plane", title: `${method} 요청`, sub: "API Endpoint" },
    { icon: "fa-magnifying-glass", title: "메타데이터 검색", sub: `${source.source} · ${source.object}` },
    { icon: "fa-plug", title: "Adapter 선택", sub: adapter },
    { icon: "fa-database", title: "SQL 실행", sub: "In-Memory 처리" },
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
  const sources = appData.metadata_schemas;

  // 카탈로그·스키마·API 빌더가 모두 같은 소스를 바라보도록 선택 상태를 단일화.
  const [sourceId, setSourceId] = useState(sources[0].id);
  const [method, setMethod] = useState("GET");
  const [filterText, setFilterText] = useState("");
  const [sortCol, setSortCol] = useState("");
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);
  const [token, setToken] = useState(null);
  const [authMethod, setAuthMethod] = useState("JWT");
  const [responseText, setResponseText] = useState(READY_RESPONSE);
  const [apiMs, setApiMs] = useState(null);
  // STEP ③ 완료 판정: 인증된 요청이 표준 응답(2xx)을 수신했는지
  const [responseOk, setResponseOk] = useState(false);
  // 빌드·등록된 API 목록 (localStorage 보존)
  const [builtApis, setBuiltApis] = useState(loadBuiltApis);
  // 우측 응답 카드의 최근 전송 시각 (빌더 [전송] 전용)
  const [sentAt, setSentAt] = useState(null);
  // 발급 API [호출] 결과 — 호출한 행 아래 인라인으로 표시 (빌더 응답 패널과 분리)
  const [builtResult, setBuiltResult] = useState(null);

  // 단계 접기/펼치기 + 스텝퍼 내비게이션 (시뮬레이터 탭과 동일 UX)
  const [openStages, setOpenStages] = useState({
    "dstep-source": true,
    "dstep-schema": true,
    "dstep-builder": true
  });
  const toggleStage = (id) => setOpenStages((s) => ({ ...s, [id]: !s[id] }));
  const allOpen = Object.values(openStages).every(Boolean);
  const setAllStages = (open) =>
    setOpenStages({ "dstep-source": open, "dstep-schema": open, "dstep-builder": open });
  // 스텝퍼/다음 단계 버튼 클릭 → 대상 단계를 펼친 뒤 스크롤 이동
  const jumpToStage = (id) => {
    setOpenStages((s) => ({ ...s, [id]: true }));
    setTimeout(() => {
      document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 60);
  };

  // 스크롤 스파이 — 화면 상단 밴드에 걸린 단계를 스텝퍼에 하이라이트
  const [activeStageId, setActiveStageId] = useState("dstep-source");
  useEffect(() => {
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
  }, []);

  const target = sources.find((s) => s.id === sourceId) ?? sources[0];
  const adapter = pickAdapter(target.id);
  const generatedSql = buildSql({
    method,
    table: target.object,
    columns: target.columns,
    filter: filterText.trim(),
    sort: sortCol,
    page,
    pageSize
  });

  const handleSelectSource = (id) => {
    setSourceId(id);
    setSortCol(""); // 소스가 바뀌면 이전 소스의 정렬 컬럼은 무효
    // 이전 소스 기준 응답은 무효 — 완료 해제 + 응답 패널 초기화
    setResponseOk(false);
    setResponseText(READY_RESPONSE);
    setApiMs(null);
    setSentAt(null);
  };

  const handleIssueToken = () => {
    if (authMethod === "OAuth2") {
      const grant = issueMockOAuth2(target.id);
      setToken(grant.access_token);
      addConsoleLog(
        `INFO: OAuth2 토큰 발급 완료 (grant_type=authorization_code, code=${grant.authorization_code}, token_type=Bearer, expires_in=${grant.expires_in}).`
      );
      return;
    }
    const t = issueMockJwt(target.id);
    setToken(t);
    addConsoleLog("INFO: JWT 토큰 발급 완료 (scope: data:read data:write, exp: 1h).");
  };

  // 인증·라우팅·SQL·응답 생성의 공용 경로 — 빌더 [전송]과 목록 [호출]이 결과 표시만 달리한다
  const executeRequest = (cfg) => {
    if (!token) return { ok: false, body: buildUnauthorized(cfg.source.id) };
    const reqAdapter = pickAdapter(cfg.source.id);
    const sql = buildSql({
      method: cfg.method,
      table: cfg.source.object,
      columns: cfg.source.columns,
      filter: cfg.filter,
      sort: cfg.sort,
      page: cfg.page,
      pageSize: cfg.pageSize
    });
    const payload = decodeJwtPayload(token);
    const body = buildApiResponse({
      method: cfg.method,
      schema: cfg.source,
      adapter: reqAdapter,
      sql,
      payload,
      filter: cfg.filter,
      sort: cfg.sort,
      page: cfg.page,
      pageSize: cfg.pageSize
    });
    return { ok: true, body };
  };

  const logRequestResult = (cfg, result, elapsed) => {
    if (result.ok) {
      addConsoleLog(
        `INFO: DataOps ${cfg.method} 성공 (${result.body.status}) - /api/v3/dataops/${cfg.source.id} (${elapsed.toFixed(0)}ms, 아카이브 ${cfg.source.archive?.tier ?? "-"} 티어 경유)`
      );
    } else {
      addConsoleLog(`WARN: 미인증 ${cfg.method} 요청 거부 (401) - 토큰 미발급`, false, true);
    }
  };

  // 빌더 [전송] — 우측 응답 카드에 표시
  const handleRunApi = () => {
    const cfg = { method, source: target, filter: filterText.trim(), sort: sortCol, page, pageSize };
    const start = performance.now();
    setApiMs(null);
    setResponseText(`Sending ${cfg.method} request...`);
    setSentAt(new Date().toLocaleTimeString("ko-KR", { hour12: false }));

    setTimeout(() => {
      const elapsed = performance.now() - start;
      setApiMs(elapsed);
      const result = executeRequest(cfg);
      setResponseText(JSON.stringify(result.body, null, 2));
      setResponseOk(result.ok);
      logRequestResult(cfg, result, elapsed);
    }, 420);
  };

  // 현재 구성을 API 자산으로 빌드·등록 — 동일 구성은 갱신(최신을 맨 위로)
  const handleBuildApi = () => {
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
    setBuiltApis((prev) => {
      const next = [entry, ...prev.filter((a) => a.sig !== sig)].slice(0, MAX_BUILT_APIS);
      persistBuiltApis(next);
      return next;
    });
    addConsoleLog(
      `INFO: Data API 빌드·등록 — ${method} ${entry.endpoint}${entry.filter ? ` (filter: ${entry.filter})` : ""}`
    );
  };

  // 등록된 API [호출] — 빌더 상태를 건드리지 않고 스냅샷 그대로 실행, 결과는 해당 행 아래 인라인 표시
  const handleInvokeBuilt = (api) => {
    const source = sources.find((s) => s.id === api.sourceId);
    if (!source) {
      addConsoleLog(`WARN: 등록 API 호출 실패 — 데이터 소스(${api.sourceId})를 찾을 수 없습니다.`, false, true);
      return;
    }
    const cfg = { method: api.method, source, filter: api.filter, sort: api.sort, page: api.page, pageSize: api.pageSize };
    const start = performance.now();
    setBuiltResult({ apiId: api.id, text: `Sending ${api.method} request...`, ms: null, time: null, ok: false });

    setTimeout(() => {
      const elapsed = performance.now() - start;
      const result = executeRequest(cfg);
      setBuiltResult({
        apiId: api.id,
        text: JSON.stringify(result.body, null, 2),
        ms: elapsed,
        time: new Date().toLocaleTimeString("ko-KR", { hour12: false }),
        ok: result.ok
      });
      if (result.ok) setResponseOk(true); // 발급 API의 정상 응답도 STEP ③ 완료 근거
      logRequestResult(cfg, result, elapsed);
    }, 420);
  };

  const handleDeleteBuilt = (id) => {
    setBuiltApis((prev) => {
      const next = prev.filter((a) => a.id !== id);
      persistBuiltApis(next);
      return next;
    });
    setBuiltResult((r) => (r?.apiId === id ? null : r));
  };

  // ①②는 유효한 기본 선택이 항상 존재하므로 완료, ③은 표준 응답 수신 시 완료
  const doneStages = ["dstep-source", "dstep-schema", ...(responseOk ? ["dstep-builder"] : [])];

  return (
    <>
      <p className="dataops-page-sub">
        <i className="fa-solid fa-box-archive" aria-hidden="true"></i> 데이터 라이프사이클 관리
        기술(DataOps) — 빅데이터 관리 아카이빙 · 메타데이터 기반 다기종 데이터 관리
      </p>

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
        title="데이터 소스 선택 — 빅데이터 아카이브 카탈로그"
        sub="아카이빙된 다기종 데이터 소스를 메타데이터 카탈로그에서 선택"
        open={openStages["dstep-source"]}
        onToggle={() => toggleStage("dstep-source")}
      >
        <Card>
          <LifecycleFlow />
          <p style={{ fontSize: 13, color: "var(--text-secondary)", marginBottom: 16 }}>
            수집·가공 데이터를 메타데이터 기반으로 아카이빙하고, 물리 저장소를 직접 노출하지 않고
            단일 API로 연계 제공합니다. 소스를 선택하면 STEP ② 스키마와 STEP ③ API 빌더 대상이 함께
            전환됩니다.
          </p>

          <div className="table-container">
            <table className="catalog-table">
              <thead>
                <tr>
                  <th>데이터 소스</th>
                  <th>저장소 유형</th>
                  <th>데이터 객체</th>
                  <th>아카이브 티어</th>
                  <th>보존 정책</th>
                  <th>적재일</th>
                  <th style={{ textAlign: "right" }}>컬럼</th>
                </tr>
              </thead>
              <tbody>
                {sources.map((s) => {
                  const isActive = s.id === sourceId;
                  return (
                    <tr
                      key={s.id}
                      onClick={() => handleSelectSource(s.id)}
                      className={isActive ? "catalog-row-active" : ""}
                      style={{ cursor: "pointer" }}
                      title={`${s.label} 소스를 API 대상으로 선택`}
                    >
                      <td>
                        <strong>{s.label}</strong>
                        {isActive && (
                          <span className="catalog-selected-chip">
                            <i className="fa-solid fa-check" aria-hidden="true"></i> 선택됨
                          </span>
                        )}
                      </td>
                      <td style={{ fontSize: 12, color: "var(--text-secondary)" }}>{s.source}</td>
                      <td>
                        <code style={{ fontSize: 11, color: "var(--accent-purple)" }}>{s.object}</code>
                      </td>
                      <td>
                        {s.archive && (
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
                        )}
                      </td>
                      <td style={{ fontSize: 12, color: "var(--text-secondary)" }}>
                        {s.archive?.retention ?? "—"}
                      </td>
                      <td style={{ fontSize: 12, color: "var(--text-muted)" }}>
                        {s.archive?.loaded_at ?? "—"}
                      </td>
                      <td style={{ textAlign: "right", color: "var(--text-muted)" }}>
                        {s.columns.length}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

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
          <div
            style={{
              display: "flex",
              alignItems: "baseline",
              gap: 8,
              flexWrap: "wrap"
            }}
          >
            <h4 style={{ color: "var(--accent-blue)", margin: 0 }}>{target.label}</h4>
            <span style={{ fontSize: 12, color: "var(--text-muted)" }}>
              {target.source} · {target.object}
            </span>
          </div>
          <p style={{ fontSize: 12, color: "var(--text-secondary)", margin: "6px 0 12px", fontStyle: "italic" }}>
            {target.description}
          </p>

          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>컬럼명</th>
                  <th>데이터 타입</th>
                  <th>설명</th>
                </tr>
              </thead>
              <tbody>
                {target.columns.map((col) => (
                  <tr key={col.name}>
                    <td>
                      <code style={{ color: "var(--accent-purple)", fontWeight: 600 }}>{col.name}</code>
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
                  <div style={{ display: "flex", gap: 8, flexShrink: 0 }}>
                    <select
                      className="select-control"
                      style={{ padding: "6px 8px", width: 100 }}
                      value={authMethod}
                      onChange={(e) => {
                        setAuthMethod(e.target.value);
                        setToken(null);
                      }}
                      aria-label="인증 방식"
                    >
                      {AUTH_METHODS.map((m) => (
                        <option key={m} value={m}>
                          {m}
                        </option>
                      ))}
                    </select>
                    <button className="btn btn-secondary" style={{ padding: "6px 12px" }} onClick={handleIssueToken}>
                      <i className="fa-solid fa-fingerprint"></i> 토큰 발급
                    </button>
                  </div>
                </div>
                {authMethod === "OAuth2" && (
                  <div style={{ marginTop: 8, fontSize: 11, color: "var(--text-muted)", lineHeight: 1.5 }}>
                    <i className="fa-solid fa-circle-info" style={{ marginRight: 4 }}></i>
                    OAuth2 Authorization Code Grant: 인가코드 발급 → access_token(JWT) 교환 흐름을 시뮬레이션합니다.
                  </div>
                )}
                {token && (
                  <code className="token-line">Authorization: Bearer {token}</code>
                )}
              </div>
            </div>

            {/* 3-2 요청 구성 */}
            <div className="builder-section">
              <div className="builder-section-label">3-2 · 요청 구성 (CRUD · 필터 · 정렬 · 페이징)</div>
              <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
                <select
                  className="select-control"
                  style={{ flex: "0 0 110px" }}
                  value={method}
                  onChange={(e) => setMethod(e.target.value)}
                  aria-label="HTTP 메서드"
                >
                  {HTTP_METHODS.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
                <select
                  className="select-control"
                  value={sourceId}
                  onChange={(e) => handleSelectSource(e.target.value)}
                  aria-label="대상 데이터 소스"
                >
                  {sources.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.label} — /api/v3/dataops/{s.id}
                    </option>
                  ))}
                </select>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                <input
                  className="input-control"
                  placeholder="filter (예: in_flow_count > 100)"
                  value={filterText}
                  onChange={(e) => setFilterText(e.target.value)}
                  aria-label="필터 조건"
                />
                <select
                  className="select-control"
                  value={sortCol}
                  onChange={(e) => setSortCol(e.target.value)}
                  aria-label="정렬 컬럼"
                >
                  <option value="">정렬 없음</option>
                  {target.columns.map((c) => (
                    <option key={c.name} value={c.name}>
                      sort: {c.name}
                    </option>
                  ))}
                </select>
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
                    value={pageSize}
                    onChange={(e) => setPageSize(Math.max(1, parseInt(e.target.value, 10) || 1))}
                    aria-label="페이지 크기"
                  />
                </label>
              </div>
            </div>

            {/* 3-3 가상화 라우팅 + SQL */}
            <div className="builder-section">
              <div className="builder-section-label">3-3 · 메타데이터 가상화 라우팅</div>
              <RoutingFlow method={method} source={target} adapter={adapter} />
              <pre className="sql-preview">{generatedSql}</pre>
            </div>

            <div className="builder-action-row">
              <button className="btn btn-secondary" onClick={handleBuildApi}>
                <i className="fa-solid fa-hammer"></i> API 빌드·등록
              </button>
              <button className="btn btn-primary" onClick={handleRunApi}>
                <i className="fa-solid fa-paper-plane"></i> {method} /api/v3/dataops/{target.id} 전송
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
            <pre className="api-response">{responseText}</pre>
          </Card>
        </div>

        {/* 빌드된 API 자산 목록 — API생성기의 생성·관리·기록 흐름 */}
        <Card
          title="발급된 API 목록"
          icon="fa-list-check"
          className="dataops-built-card"
          headerRight={
            <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
              {builtApis.length}건 · 브라우저에 보존
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
              <table>
                <thead>
                  <tr>
                    <th>메서드</th>
                    <th>엔드포인트</th>
                    <th>대상 소스</th>
                    <th>쿼리</th>
                    <th>인증</th>
                    <th>빌드 일시</th>
                    <th>상태</th>
                    <th style={{ textAlign: "right" }}>동작</th>
                  </tr>
                </thead>
                <tbody>
                  {builtApis.map((api) => {
                    const rgb = METHOD_COLORS[api.method] ?? METHOD_COLORS.GET;
                    const isInvoked = builtResult?.apiId === api.id;
                    const querySummary =
                      [
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
                              color: `rgb(${rgb})`,
                              backgroundColor: `rgba(${rgb}, 0.1)`
                            }}
                          >
                            {api.method}
                          </span>
                        </td>
                        <td>
                          <code style={{ fontSize: 11, color: "var(--accent-purple)" }}>{api.endpoint}</code>
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
                              color: "var(--accent-teal)",
                              backgroundColor: "rgba(16, 185, 129, 0.1)"
                            }}
                          >
                            Active
                          </span>
                        </td>
                        <td style={{ textAlign: "right", whiteSpace: "nowrap" }}>
                          <button
                            className="btn btn-secondary"
                            style={{ padding: "4px 10px", fontSize: 11 }}
                            onClick={() => handleInvokeBuilt(api)}
                            title="등록된 구성으로 즉시 호출"
                          >
                            <i className="fa-solid fa-play"></i> 호출
                          </button>
                          <button
                            className="btn btn-secondary"
                            style={{ padding: "4px 8px", fontSize: 11, marginLeft: 6 }}
                            onClick={() => handleDeleteBuilt(api.id)}
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
            </div>
          )}
        </Card>
      </CollapsibleStage>
    </>
  );
}
