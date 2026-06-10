import { useState } from "react";
import Card from "../components/Card.jsx";
import PerfBadge from "../components/PerfBadge.jsx";
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

  const [activeSource, setActiveSource] = useState(sources[0].id);
  const [endpoint, setEndpoint] = useState(sources[0].id);
  const [method, setMethod] = useState("GET");
  const [filterText, setFilterText] = useState("");
  const [sortCol, setSortCol] = useState("");
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);
  const [token, setToken] = useState(null);
  const [authMethod, setAuthMethod] = useState("JWT");
  const [responseText, setResponseText] = useState(READY_RESPONSE);
  const [apiMs, setApiMs] = useState(null);

  const activeSchema = sources.find((s) => s.id === activeSource) ?? sources[0];
  const target = sources.find((s) => s.id === endpoint) ?? sources[0];
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

  const handleRunApi = () => {
    const start = performance.now();
    setApiMs(null);
    setResponseText(`Sending ${method} request...`);

    setTimeout(() => {
      const elapsed = performance.now() - start;
      setApiMs(elapsed);

      if (!token) {
        setResponseText(JSON.stringify(buildUnauthorized(target.id), null, 2));
        addConsoleLog(`WARN: 미인증 ${method} 요청 거부 (401) - 토큰 미발급`, false, true);
        return;
      }

      const payload = decodeJwtPayload(token);
      const response = buildApiResponse({
        method,
        schema: target,
        adapter,
        sql: generatedSql,
        payload,
        filter: filterText.trim(),
        sort: sortCol,
        page,
        pageSize
      });
      setResponseText(JSON.stringify(response, null, 2));
      addConsoleLog(
        `INFO: DataOps ${method} 성공 (${response.status}) - /api/v3/dataops/${endpoint} (${elapsed.toFixed(0)}ms)`
      );
    }, 420);
  };

  return (
    <div className="grid-details-split">
      <Card title="연계 데이터 소스 카탈로그" icon="fa-layer-group">
        <p style={{ fontSize: 13, color: "var(--text-secondary)", marginBottom: 16 }}>
          다기종 데이터를 메타데이터로 가상화하여, 물리 저장소를 직접 노출하지 않고 단일 API로 연계
          제공합니다. 데이터 소스를 선택하면 연결된 스키마를 확인할 수 있습니다.
        </p>

        <div className="table-container">
          <table className="catalog-table">
            <thead>
              <tr>
                <th>데이터 소스</th>
                <th>저장소 유형</th>
                <th>데이터 객체</th>
                <th style={{ textAlign: "right" }}>컬럼</th>
              </tr>
            </thead>
            <tbody>
              {sources.map((s) => {
                const isActive = s.id === activeSource;
                return (
                  <tr
                    key={s.id}
                    onClick={() => setActiveSource(s.id)}
                    className={isActive ? "catalog-row-active" : ""}
                    style={{ cursor: "pointer" }}
                    title={`${s.label} 스키마 보기`}
                  >
                    <td>
                      <strong>{s.label}</strong>
                    </td>
                    <td style={{ fontSize: 12, color: "var(--text-secondary)" }}>{s.source}</td>
                    <td>
                      <code style={{ fontSize: 11, color: "var(--accent-purple)" }}>{s.object}</code>
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

        <div
          style={{
            marginTop: 18,
            display: "flex",
            alignItems: "baseline",
            gap: 8,
            flexWrap: "wrap"
          }}
        >
          <h4 style={{ color: "var(--accent-blue)", margin: 0 }}>
            <i className="fa-solid fa-table-columns" style={{ marginRight: 6 }}></i>
            {activeSchema.label}
          </h4>
          <span style={{ fontSize: 12, color: "var(--text-muted)" }}>
            {activeSchema.source} · {activeSchema.object}
          </span>
        </div>
        <p style={{ fontSize: 12, color: "var(--text-secondary)", margin: "6px 0 12px", fontStyle: "italic" }}>
          {activeSchema.description}
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
              {activeSchema.columns.map((col) => (
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
      </Card>

      <Card
        title="Data API 빌더 — REST 처리"
        icon="fa-code"
        headerRight={<PerfBadge ms={apiMs} label="API 응답" />}
      >
        <p style={{ fontSize: 13, color: "var(--text-secondary)", marginBottom: 16 }}>
          표준 SQL 설정 기반 CRUD·필터·정렬·페이징을 In-Memory로 처리하여 표준 REST로 응답합니다.
          JWT/OAuth2 인증으로 저장소를 직접 노출하지 않습니다.
        </p>

        {/* ① 인증 */}
        <div className="builder-section">
          <div className="builder-section-label">① 인증</div>
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

        {/* ② 요청 구성 */}
        <div className="builder-section">
          <div className="builder-section-label">② 요청 구성 (CRUD · 필터 · 정렬 · 페이징)</div>
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
              value={endpoint}
              onChange={(e) => setEndpoint(e.target.value)}
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

        {/* ③ 가상화 라우팅 + SQL */}
        <div className="builder-section">
          <div className="builder-section-label">③ 메타데이터 가상화 라우팅</div>
          <RoutingFlow method={method} source={target} adapter={adapter} />
          <pre className="sql-preview">{generatedSql}</pre>
        </div>

        <button className="btn btn-primary" style={{ width: "100%", marginBottom: 16 }} onClick={handleRunApi}>
          <i className="fa-solid fa-paper-plane"></i> {method} Request 전송
        </button>

        {/* ④ 응답 */}
        <div className="builder-section-label">④ REST API JSON 응답</div>
        <pre className="api-response">{responseText}</pre>
      </Card>
    </div>
  );
}
