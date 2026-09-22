// DataOps Data API Builder 시뮬레이션 헬퍼 — CRUD/필터/정렬/페이징 + JWT 인증 + DB Adapter/SQL 생성.
// 무분별한 저장소 직접 접근을 막고 API+메타데이터로 추상화한다는 Notion 명세를 클라이언트에서 재현.

import { splitFilterConditions } from "./filterExpression.js";

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

/** 데이터 소스 종류에 맞는 DB Adapter 선택 (공간정보는 PostGIS, 비정형 문서는 Mongo). */
export function pickAdapter(sourceId) {
  if (sourceId.includes("complaints")) return "MongoAdapter (Document Store)";
  if (sourceId.includes("spatial")) return "PostGISAdapter (EPSG:4326)";
  if (sourceId.includes("smartfarm")) return "TimescaleDBAdapter (시계열)";
  if (sourceId.includes("welfare") || sourceId.includes("industrial") || sourceId.includes("facility")) {
    return "PostgreSQLAdapter";
  }
  return "PostgreSQLAdapter";
}

/** 문서형(NoSQL) 저장소 여부 — Adapter가 SQL 대신 MQL(Mongo Query Language)을 생성한다. */
export function isDocumentStore(schema) {
  return (schema.source ?? "").includes("MongoDB");
}

/**
 * 스키마의 저장소 유형 문자열로 Adapter 결정 — 사용자 등록 소스처럼
 * id 규칙이 없는 스키마도 처리하고, 기본 소스는 기존 id 휴리스틱으로 위임.
 */
export function adapterOf(schema) {
  const src = schema.source ?? "";
  if (src.includes("MongoDB")) return "MongoAdapter (Document Store)";
  if (src.includes("PostGIS")) return "PostGISAdapter (EPSG:4326)";
  if (src.includes("TimescaleDB")) return "TimescaleDBAdapter (시계열)";
  return pickAdapter(schema.id);
}

const fmtSqlValue = (v) => (typeof v === "number" ? v : `'${v}'`);

/** 메타데이터 적재 범위(range) → SQL BETWEEN 조건. 사용자 filter와 AND로 결합. */
function sqlWhere(range, filter) {
  const parts = [];
  if (range) parts.push(`${range.column} BETWEEN ${fmtSqlValue(range.from)} AND ${fmtSqlValue(range.to)}`);
  if (filter) parts.push(filter);
  return parts.length ? ` WHERE ${parts.join(" AND ")}` : "";
}

/** 메서드/필터/정렬/페이징으로부터 표준 SQL 문을 생성 (메타데이터 range 자동 주입). */
export function buildSql({ method, table, columns, range, filter, sort, page, pageSize }) {
  const colList = columns.map((c) => c.name).join(", ");
  const where = sqlWhere(range, filter);
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

/** 사용자 filter(` AND ` 로 결합된 `col > 100` 조건들)를 MQL 조각으로. 해석 불가 시 주석으로 보존.
 *  분리 규칙은 백엔드 safety.split_filter_conditions 와 같다. */
function mongoFilterParts(filter) {
  return splitFilterConditions(filter).map((condition) => {
    const m = condition.match(/^(\w+)\s*(>=|<=|!=|=|>|<)\s*(.+)$/);
    if (!m) return `/* 미해석 조건: ${condition} */`;
    const [, col, op, rawVal] = m;
    const num = Number(rawVal);
    const val = Number.isFinite(num) ? num : `"${rawVal.replace(/^['"]|['"]$/g, "")}"`;
    const OPS = { ">": "$gt", ">=": "$gte", "<": "$lt", "<=": "$lte", "!=": "$ne" };
    return op === "=" ? `${col}: ${val}` : `${col}: { ${OPS[op]}: ${val} }`;
  });
}

/** 메타데이터 range + filter → MQL match 식 (이미지의 db.obj1.find(seq:{$gt..,$lt..}) 재현). */
function mongoMatch(range, filter) {
  const parts = [];
  if (range) parts.push(`${range.column}: { $gte: ${JSON.stringify(range.from)}, $lte: ${JSON.stringify(range.to)} }`);
  parts.push(...mongoFilterParts(filter));
  return `{ ${parts.join(", ")} }`;
}

/** 문서형 저장소용 MQL 문 생성 — 동일 요청 구성이 저장소에 따라 다른 쿼리 언어로 변환됨을 보인다. */
export function buildMql({ method, collection, columns, range, filter, sort, page, pageSize }) {
  const match = mongoMatch(range, filter);
  const docBody = `{ ${columns.map((c) => `${c.name}: <${c.type}>`).join(", ")} }`;
  const sortSeg = sort ? `.sort({ ${sort}: -1 })` : "";
  const skip = (page - 1) * pageSize;

  switch (method) {
    case "POST":
      return `db.${collection}.insertOne(\n  ${docBody}\n);`;
    case "PUT":
      return `db.${collection}.updateMany(\n  ${match},\n  { $set: ${docBody} }\n);`;
    case "PATCH":
      return `db.${collection}.updateMany(\n  ${match},\n  { $set: { ${columns[0].name}: <${columns[0].type}> } }\n);`;
    case "DELETE":
      return `db.${collection}.deleteMany(${match});`;
    case "GET":
    default:
      return `db.${collection}.find(\n  ${match}\n)${sortSeg}.skip(${skip}).limit(${pageSize});`;
  }
}

/**
 * 메타데이터 가상화 라우팅의 쿼리 생성 단계 — 저장소 유형에 맞춰 SQL 또는 MQL을 산출.
 * @returns {{ lang: "SQL" | "MQL", text: string }}
 */
export function buildQuery({ method, schema, filter, sort, page, pageSize }) {
  const common = { method, columns: schema.columns, range: schema.range, filter, sort, page, pageSize };
  if (isDocumentStore(schema)) {
    return { lang: "MQL", text: buildMql({ ...common, collection: schema.object }) };
  }
  return { lang: "SQL", text: buildSql({ ...common, table: schema.object }) };
}

/** 메서드별 표준 REST 응답 본문(JSON 직렬화 대상)을 생성. */
export function buildApiResponse({ method, schema, adapter, query, payload, filter, sort, page, pageSize }) {
  const base = {
    status: method === "POST" ? 201 : 200,
    method,
    endpoint: `/api/v3/dataops/${schema.id}`,
    dataops_version: "3.0.0-R3",
    auth: { authenticated: true, sub: payload.sub, scope: payload.scope },
    db_adapter: adapter,
    // 빅데이터 관리 아카이빙 — 응답에 아카이브 스토리지 메타를 동봉해 접근 이력·보존 정책을 노출
    archive_meta: schema.archive
      ? {
          storage_tier: schema.archive.tier,
          retention: schema.archive.retention,
          loaded_at: schema.archive.loaded_at
        }
      : null,
    // 메타데이터 적재 범위 — Adapter가 쿼리에 자동 주입한 스코프를 응답에서 추적 가능하게 노출
    range_scope: schema.range
      ? { column: schema.range.column, from: schema.range.from, to: schema.range.to }
      : null,
    query_language: query.lang,
    generated_query: query.text
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
 * 자동 리포팅 ↔ 실데이터 API 바인딩
 *   리포트 지표는 합성하지 않는다 — 활성 모델의 evaluation(WAPE·MAE)과
 *   drift(PSI)를 실제로 호출해 채우고, 채울 값이 없으면 빈 상태로 둔다.
 * ------------------------------------------------------------------ */

const REALDATA_BASE = "/api/v3/realdata";

/** 활성 버전이 있는 첫 모델 — 모델 ID는 하드코딩하지 않는다. */
export function pickActiveModel(models = []) {
  return models.find((m) => m.active_version) ?? null;
}

/** 활성 모델의 최신 스냅샷 — `/realdata/datasets` 는 created_at 내림차순이라 첫 건이 최신이다. */
export function pickModelDataset(datasets = [], model) {
  if (!model) return null;
  return datasets.find((d) => d?.spec?.model_id === model.model_id) ?? null;
}

/**
 * 설명 대상 행 선택 — 최신 base_ym(`observed_to`)에서 `local_visitors` 가 최대인 행정동.
 * rows 가 절단됐거나(`message` 비어있지 않음) `local_visitors` 컬럼이 없으면 null(빈 상태).
 * 대체 컬럼으로 "생활인구 최대"를 흉내내지 않는다.
 * @returns {{ baseYm: number, dongCode: string } | null}
 */
export function pickExplainTarget(datasetResponse, observedTo) {
  if (datasetResponse?.status !== "ok" || datasetResponse.message) return null;
  const rows = datasetResponse.data?.rows;
  if (!Array.isArray(rows) || typeof observedTo !== "number") return null;

  const latest = rows.filter((r) => r?.base_ym === observedTo);
  // 소비 모델 스냅샷에는 local_visitors 가 없다 — 그 경우 선택 자체를 포기한다.
  const withPopulation = latest.filter((r) => typeof r.local_visitors === "number");
  if (withPopulation.length === 0) return null;

  const top = withPopulation.reduce((a, b) => (b.local_visitors > a.local_visitors ? b : a));
  if (typeof top.dong_code !== "string") return null;
  return { baseYm: observedTo, dongCode: top.dong_code };
}

/** 모델 타깃 컬럼 → 리포트 표기명. 매핑에 없으면 타깃 문자열 그대로 쓴다(이름을 지어내지 않는다). */
const TARGET_LABELS = {
  nonlocal_visitors: "외지인 방문객",
  observed_sales_krw: "소비 매출 추정액"
};

// 피처명 → 리포트 표기명. `y_*` 는 모델 타깃의 랙이라 타깃에 따라 문구가 달라진다(생활인구가 아니다).
// has_yoy 는 전년동월 결측 여부를 가르는 0/1 플래그다(features.py 의 y_yoy 대체 규칙).
// month_sin/month_cos 는 항상 함께 상위에 올 수 있어 사인·코사인을 구분해 둔다.
const FEATURE_LABELS = {
  y_lag1: (y) => y && `직전월 ${y}`,
  y_lag2: (y) => y && `2개월 전 ${y}`,
  y_lag3: (y) => y && `3개월 전 ${y}`,
  y_yoy: (y) => y && `전년 동월 ${y}`,
  has_yoy: () => "전년 동월 값 유무",
  month_sin: () => "계절성(월, 사인)",
  month_cos: () => "계절성(월, 코사인)"
};

/**
 * 모델 피처명 → 지자체 담당자가 읽는 리포트 표기명. 리포트 본문·표·엑셀이 이 한 함수만 쓴다.
 * 매핑에 없는 피처, 대응표에 없는 행정동 코드, 타깃을 모르는 `y_*` 는 원본 피처명 그대로 둔다.
 * @param {string} feature 모델 피처명
 * @param {string|null|undefined} modelTarget `/realdata/models` 의 모델 `target` 컬럼명
 * @param {Array<{dong_code?: string, dong_name?: string}>} dongEntries `/realdata/dong-map` 의 entries
 */
export function featureLabel(feature, modelTarget, dongEntries = []) {
  const build = FEATURE_LABELS[feature];
  if (build) return build(TARGET_LABELS[modelTarget] ?? modelTarget) || feature;
  if (typeof feature === "string" && feature.startsWith("dong_")) {
    const code = feature.slice("dong_".length);
    const name = dongEntries.find((e) => e?.dong_code === code)?.dong_name;
    return name ? `행정동: ${name}` : feature;
  }
  return feature;
}

/**
 * explain 응답 → `{ baseYm, dongCode, dongName, contributions }`. `ok` 가 아니면 null(빈 상태).
 * 기여도는 `|phi|` 내림차순으로 정렬한다. `note` 는 내부 설명 문구라 옮기지 않는다.
 * 각 기여도에는 표기명 `label` 을 함께 싣고 원본 `feature` 도 추적용으로 남긴다.
 */
export function toExplainBinding(explainResponse, target, dongMapResponse, modelTarget) {
  const data = explainResponse?.status === "ok" ? explainResponse.data : null;
  const raw = data?.contributions;
  if (!target || !Array.isArray(raw) || raw.length === 0) return null;

  const entries = dongMapResponse?.status === "ok" ? dongMapResponse.data?.entries ?? [] : [];
  const match = entries.find((e) => e?.dong_code === target.dongCode);

  return {
    baseYm: target.baseYm,
    dongCode: target.dongCode,
    dongName: match?.dong_name ?? null,
    contributions: [...raw]
      .map((c) => ({ feature: c.feature, label: featureLabel(c.feature, modelTarget, entries), value: c.value, phi: c.phi }))
      .sort((a, b) => Math.abs(b.phi) - Math.abs(a.phi))
  };
}

/**
 * drift 응답 → `[{ feature, label, psi }]`. 집계하지 않고 피처·타깃 실측값을 그대로 옮긴다.
 * 표기명은 SHAP 기여도와 같은 `featureLabel` 을 쓴다 — 한 리포트에 내부 피처명이 섞이지 않게.
 *
 * 행정동 원핫(`dong_*`)은 PSI 목록에서 제외한다. dong-map 을 이 경로에서 받지 않아(추가 호출 금지)
 * 코드가 그대로 노출되는 데다, 행정동 수만큼 줄이 늘어 담당자가 읽을 지표가 묻힌다.
 * 기준을 값(`psi === 0`)이 아니라 피처명 접두사로 둔 이유: 값 기준은 0 이 아닌 원핫이 하나만 생겨도
 * 다시 노이즈가 들어오고, 의미 있는 피처가 우연히 0 일 때 사라진다. 접두사 기준은 "행정동 원핫은
 * 리포트 지표가 아니다"는 의도를 직접 표현하고 값 분포에 의존하지 않는다.
 * SHAP 기여도(`toExplainBinding`)의 `dong_*` 는 dong-map 으로 행정동명이 붙어 의미가 있으므로 남긴다.
 */
function psiEntries(driftResponse, modelTarget) {
  const data = driftResponse?.status === "ok" ? driftResponse.data : null;
  if (!data || data.status !== "ok") return [];
  const entries = (data.features ?? []).filter((f) => !String(f?.feature).startsWith("dong_")).map((f) => ({
    feature: f.feature,
    label: featureLabel(f.feature, modelTarget, []),
    psi: f.psi
  }));
  // 타깃 행은 내부 기호 `y` 를 노출하지 않는다 — 타깃 표기명, 없으면 타깃 문자열, 그마저 없으면 "타깃".
  if (data.target) {
    entries.push({ feature: "y", label: TARGET_LABELS[modelTarget] ?? modelTarget ?? "타깃", psi: data.target.psi });
  }
  return entries;
}

/**
 * 실호출 응답 → 리포트 바인딩 모델. 활성 모델이나 검증지표가 없으면 null(빈 상태).
 * population·birthRate는 대응 실데이터 소스가 없어 지자체 시드 값을 그대로 옮긴다.
 */
export function toReportIndicators(region, model, evaluationResponse, driftResponse, explain = null) {
  const validation = evaluationResponse?.status === "ok" ? evaluationResponse.data?.validation : null;
  if (!model || !validation) return null;

  return {
    explain,
    source: `${REALDATA_BASE}/models/${model.model_id}/evaluation`,
    driftSource: `${REALDATA_BASE}/models/${model.model_id}/drift`,
    modelId: model.model_id,
    version: model.active_version,
    indicators: {
      wape: validation.metrics?.wape ?? null,
      mae: validation.metrics?.mae ?? null,
      baselineMae: validation.baseline?.mae ?? null,
      psi: psiEntries(driftResponse, model.target),
      population: region.population,
      birthRate: region.birthRate
    }
  };
}

/**
 * 리포트 지표 실호출 — 활성 모델의 평가·드리프트를 읽어 바인딩 모델로 돌려준다.
 * @returns {Promise<ReturnType<typeof toReportIndicators>>} 활성 모델·지표가 없으면 null
 */
export async function fetchReportData(region) {
  // ponytail: realdataClient 는 .jsx 라 node 테스트가 정적 import 를 읽지 못한다 → 호출 시점 import.
  // 정적 import 로 되돌리면 reportIndicators.test.mjs 가 깨진다(빌드의 혼합 import 경고는 무해 —
  // 다른 화면이 이미 정적으로 물고 있어 같은 청크에 남는다).
  const { getRealdataToken, getModels, getDatasets, getEvaluation, getDrift, getExplain } =
    await import("../components/realdata/realdataClient.jsx");
  // api.js 도 호출 시점 import — 최상위 import.meta.env 가 node 테스트에서 터진다.
  const { apiGet } = await import("./api.js");
  const token = await getRealdataToken();
  const model = pickActiveModel((await getModels(token)).data ?? []);
  if (!model) return null;

  const [evaluation, drift, explain] = await Promise.all([
    getEvaluation(token, model.model_id, model.active_version),
    getDrift(token, model.model_id, model.active_version),
    fetchExplainBinding(token, model, { getDatasets, getExplain, apiGet })
  ]);
  return toReportIndicators(region, model, evaluation, drift, explain);
}

/**
 * SHAP 기여도 실호출 — 최신 스냅샷의 `observed_to` 월에서 생활인구가 가장 많은 행정동을 고른다.
 * 어느 단계든 조건을 못 채우면 null(빈 상태). 사유 문구는 만들지 않는다.
 */
async function fetchExplainBinding(token, model, { getDatasets, getExplain, apiGet }) {
  try {
    const dataset = pickModelDataset((await getDatasets(token)).data?.datasets ?? [], model);
    if (!dataset) return null;

    // include_rows 는 5,000행 상한이 있다 — 절단되면 최대값을 믿을 수 없어 pickExplainTarget 이 버린다.
    const withRows = await apiGet(`${REALDATA_BASE}/datasets/${dataset.dataset_id}`, {
      token,
      params: { include_rows: true }
    });
    const target = pickExplainTarget(withRows, dataset.observed_to);
    if (!target) return null;

    const [explain, dongMap] = await Promise.all([
      getExplain(token, model.model_id, model.active_version, target.baseYm, target.dongCode),
      apiGet(`${REALDATA_BASE}/dong-map`, { token })
    ]);
    return toExplainBinding(explain, target, dongMap, model.target);
  } catch {
    // 조회 실패도 빈 상태 — 평가·드리프트 바인딩까지 같이 버리지 않는다.
    return null;
  }
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
