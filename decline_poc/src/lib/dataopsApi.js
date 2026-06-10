// DataOps Data API Builder 시뮬레이션 헬퍼 — CRUD/필터/정렬/페이징 + JWT 인증 + DB Adapter/SQL 생성.
// 무분별한 저장소 직접 접근을 막고 API+메타데이터로 추상화한다는 Notion 명세를 클라이언트에서 재현.

export const HTTP_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE"];

function base64url(obj) {
  return btoa(unescape(encodeURIComponent(JSON.stringify(obj))))
    .replace(/=/g, "")
    .replace(/\+/g, "-")
    .replace(/\//g, "_");
}

/** Mock JWT(HS256) 발급 — 실제 서명 대신 데모용 서명 세그먼트를 부여. */
export function issueMockJwt(sourceId) {
  const header = { alg: "HS256", typ: "JWT" };
  const iat = Math.floor(Date.now() / 1000);
  const payload = {
    sub: "rnd-dataops-client",
    scope: "data:read data:write",
    source: sourceId,
    iat,
    exp: iat + 3600
  };
  const sig = base64url({ s: (iat % 99991).toString(16) }).slice(0, 22);
  return `${base64url(header)}.${base64url(payload)}.${sig}`;
}

export const AUTH_METHODS = ["JWT", "OAuth2"];

/**
 * OAuth2 Authorization Code Grant 시뮬레이션.
 * 인가코드(code)를 access_token(JWT 형식)으로 교환하는 표준 흐름을 재현.
 * (access_token은 RFC 9068처럼 JWT 형식이라 기존 검증 로직과 호환)
 */
export function issueMockOAuth2(sourceId) {
  const iat = Math.floor(Date.now() / 1000);
  const code = base64url({ c: sourceId, t: iat }).slice(0, 16);
  return {
    grant_type: "authorization_code",
    authorization_code: code,
    token_type: "Bearer",
    expires_in: 3600,
    scope: "data:read data:write",
    access_token: issueMockJwt(sourceId)
  };
}

export function decodeJwtPayload(token) {
  try {
    const part = token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/");
    return JSON.parse(decodeURIComponent(escape(atob(part))));
  } catch {
    return null;
  }
}

/** 데이터 소스 종류에 맞는 DB Adapter 선택 (공간정보는 PostGIS). */
export function pickAdapter(sourceId) {
  if (sourceId.includes("spatial")) return "PostGISAdapter (EPSG:4326)";
  if (sourceId.includes("smartfarm")) return "TimescaleDBAdapter (시계열)";
  if (sourceId.includes("welfare") || sourceId.includes("industrial") || sourceId.includes("facility")) {
    return "PostgreSQLAdapter";
  }
  return "PostgreSQLAdapter (In-Memory Cache)";
}

/** 메서드/필터/정렬/페이징으로부터 표준 SQL 문을 생성. */
export function buildSql({ method, table, columns, filter, sort, page, pageSize }) {
  const colList = columns.map((c) => c.name).join(", ");
  const where = filter ? ` WHERE ${filter}` : "";
  const order = sort ? ` ORDER BY ${sort} DESC` : "";
  const offset = (page - 1) * pageSize;
  const limit = ` LIMIT ${pageSize} OFFSET ${offset}`;

  switch (method) {
    case "POST":
      return `INSERT INTO ${table} (${colList})\n  VALUES (${columns.map(() => "?").join(", ")});`;
    case "PUT":
      return `UPDATE ${table}\n  SET ${columns.map((c) => `${c.name} = ?`).join(", ")}${where};`;
    case "PATCH":
      return `UPDATE ${table}\n  SET ${columns[0].name} = ?${where};`;
    case "DELETE":
      return `DELETE FROM ${table}${where};`;
    case "GET":
    default:
      return `SELECT ${colList}\n  FROM ${table}${where}${order}${limit};`;
  }
}

/** 메서드별 표준 REST 응답 본문(JSON 직렬화 대상)을 생성. */
export function buildApiResponse({ method, schema, adapter, sql, payload, filter, sort, page, pageSize }) {
  const base = {
    status: method === "POST" ? 201 : 200,
    method,
    endpoint: `/api/v3/dataops/${schema.id}`,
    dataops_version: "3.0.0-R3",
    auth: { authenticated: true, sub: payload.sub, scope: payload.scope },
    db_adapter: adapter,
    generated_sql: sql
  };

  if (method === "GET") {
    const total = 1248;
    return {
      ...base,
      query: { filter: filter || null, sort: sort || null },
      pagination: { page, page_size: pageSize, total, total_pages: Math.ceil(total / pageSize) },
      result_rows: Math.min(pageSize, total - (page - 1) * pageSize),
      sample: schema.columns.reduce((acc, c) => {
        acc[c.name] = `<${c.type}>`;
        return acc;
      }, {})
    };
  }
  if (method === "DELETE") {
    return { ...base, affected_rows: filter ? 1 : 0, message: "Row(s) deleted via virtualized API." };
  }
  // POST/PUT/PATCH
  return {
    ...base,
    affected_rows: 1,
    message: `${method} processed through Data API Builder (storage abstracted).`
  };
}

/* ------------------------------------------------------------------ *
 * 자동 리포팅 ↔ Data source API 바인딩
 *   Notion: "API 형태로 Data source를 자동 리포팅 양식에 연결하여
 *            고정된 지표지만 데이터 갱신되는 부분 자동 업데이트"
 * ------------------------------------------------------------------ */

/** 리포트 양식에 바인딩되는 지표를 Data API Builder GET 형태로 산출. */
export function buildReportIndicators(region, driftInjected, refreshCount = 0) {
  const table = "report_indicators";
  const adapter = pickAdapter("welfare"); // In-Memory Cache 경유
  const sql =
    `SELECT metric, value, collected_at\n  FROM ${table}\n` +
    `  WHERE region_id = '${region.id}'\n  ORDER BY collected_at DESC LIMIT 50;`;

  // 갱신 시마다 신규 수집행이 누적되고 정확도가 미세 변동 → "자동 업데이트" 가시화.
  const rows = 1240 + refreshCount * 7;
  const jitter = ((refreshCount * 13) % 7) / 1000; // 0.000~0.006 결정적 변동
  const accuracy = Number(((driftInjected ? 0.872 : 0.892) + jitter).toFixed(3));

  return {
    source: `/api/v3/dataops/${table}`,
    adapter,
    sql,
    collected_rows: rows,
    indicators: {
      accuracy,
      psi: driftInjected ? 0.384 : 0.045,
      outliers: driftInjected ? 3 : 0,
      population: region.population,
      birthRate: region.birthRate
    }
  };
}

/**
 * In-memory Data API Builder를 통한 비동기 GET 시뮬레이션.
 * 표준 REST로 리포트 지표를 응답하여 양식에 자동 연결한다.
 * @returns {Promise<ReturnType<typeof buildReportIndicators>>}
 */
export function fetchReportData(region, driftInjected, refreshCount = 0) {
  return new Promise((resolve) => {
    setTimeout(() => resolve(buildReportIndicators(region, driftInjected, refreshCount)), 240);
  });
}

/** 401 미인증 응답. */
export function buildUnauthorized(sourceId) {
  return {
    status: 401,
    error: "Unauthorized",
    endpoint: `/api/v3/dataops/${sourceId}`,
    message: "JWT 토큰이 필요합니다. [JWT 토큰 발급] 후 다시 시도하세요.",
    hint: "Authorization: Bearer <token>"
  };
}
