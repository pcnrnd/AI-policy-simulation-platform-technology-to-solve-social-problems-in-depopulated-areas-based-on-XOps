// 실데이터 진단(남원) 패널 — Simulator 하단에 추가되는 신규 패널(기존 데모 경로는 불변).
// R5: 현황(current) · 진단(diagnosis, rules_v1) · 예측(forecast) · 대응 후보(responses).
// 정책 효과 수치·RICE·policyStrategies는 쓰지 않는다(계약 R5 금지) — 대응은 항상 "후보"까지만.
import { useEffect, useState } from "react";
import Card from "../Card.jsx";
import { apiGet } from "../../lib/api.js";
import { issueRealdataToken } from "./realdataClient.jsx";

const MODEL_LABELS = {
  "namwon-nonlocal-visitors-next-month": "남원 타지역 방문객 (익월)",
  "namwon-observed-sales-next-month": "남원 관측 소비 (익월)"
};

const FIELD_LABELS = {
  nonlocal_visitors: "외지인 방문객",
  local_visitors: "동일지역 방문객",
  foreign_visitors: "외국인 방문객",
  observed_sales_krw: "관측 소비(원)"
};

const STATUS_LABELS = {
  ok: "정상",
  empty: "값 없음",
  model_required: "활성 모델 필요",
  insufficient_data: "예측 표본 부족",
  insufficient_evidence: "근거 부족",
  error: "오류"
};

const orDash = (v) => (v === null || v === undefined || v === "" ? "–" : v);
const fmtNum = (v, digits = 0) => (typeof v === "number" ? v.toLocaleString(undefined, { maximumFractionDigits: digits }) : "–");
const fmtPct = (v) => (typeof v === "number" ? `${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)}%` : "–");

function addMonth(baseYm, delta) {
  const year = Math.floor(baseYm / 100);
  const month = baseYm % 100;
  const total = year * 12 + (month - 1) + delta;
  const y = Math.floor(total / 12);
  const m = total - y * 12;
  return y * 100 + m + 1;
}

function StatusNote({ status, message }) {
  if (!status || status === "ok") return null;
  return (
    <p className="realdata-hint mock-data-output">
      {STATUS_LABELS[status] ?? status}
      {message ? ` — ${message}` : ""}
    </p>
  );
}

function CurrentMetric({ metric }) {
  if (!metric) return null;
  if (metric.status !== "ok") {
    return (
      <div className="realdata-metric">
        <span className="realdata-metric-label">{FIELD_LABELS[metric.field] ?? metric.field}</span>
        <span className="realdata-hint mock-data-output">
          {metric.message}
          {metric.observed_range && ` (관측: ${metric.observed_range.from}~${metric.observed_range.to})`}
        </span>
      </div>
    );
  }
  return (
    <div className="realdata-metric mock-data-output">
      <span className="realdata-metric-label">{FIELD_LABELS[metric.field] ?? metric.field}</span>
      <span className="realdata-metric-value">{fmtNum(metric.value)}</span>
      <span className="realdata-metric-sub">
        전년동월 {metric.yoy ? fmtPct(metric.yoy.pct) : "–"} · 직전월 {metric.mom ? fmtPct(metric.mom.pct) : "–"}
      </span>
    </div>
  );
}

function Finding({ finding }) {
  return (
    <li className="realdata-finding mock-data-output">
      <strong>{finding.summary}</strong>
      <div className="realdata-hint">
        {finding.evidence.dong_code} · {finding.evidence.months.join(", ")} · 값 {finding.evidence.values.map((v) => fmtNum(v, 2)).join(" → ")}
      </div>
      {finding.limitations?.length > 0 && (
        <ul className="realdata-limitations">
          {finding.limitations.map((l) => (
            <li key={l}>{l}</li>
          ))}
        </ul>
      )}
    </li>
  );
}

export default function RealdataAnalysisPanel() {
  const [token, setToken] = useState(null);
  const [authError, setAuthError] = useState(null);

  const [dongOptions, setDongOptions] = useState([]);
  const [dongCode, setDongCode] = useState("all");

  const [baseYmOptions, setBaseYmOptions] = useState([]);
  const [baseYm, setBaseYm] = useState("");

  const [modelId, setModelId] = useState("");

  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    issueRealdataToken()
      .then(setToken)
      .catch((err) => setAuthError(err.message || "인증 토큰 발급에 실패했습니다."));
  }, []);

  // 행정동 select 옵션(dong-map API) — 최초 1회.
  useEffect(() => {
    if (!token) return;
    apiGet("/api/v3/realdata/dong-map", { token })
      .then((res) => setDongOptions(res.data?.entries ?? []))
      .catch(() => setDongOptions([]));
  }, [token]);

  // 기준월 select 범위(스냅샷 관측 범위) — 생성된 데이터셋 목록에서 계산.
  useEffect(() => {
    if (!token) return;
    apiGet("/api/v3/realdata/datasets", { token })
      .then((res) => {
        const datasets = res.data?.datasets ?? [];
        const froms = datasets.map((d) => d.observed_from).filter(Boolean);
        const tos = datasets.map((d) => d.observed_to).filter(Boolean);
        if (froms.length === 0 || tos.length === 0) return;
        const from = Math.min(...froms);
        const to = Math.max(...tos);
        const months = [];
        for (let m = to; m >= from; m = addMonth(m, -1)) months.push(m);
        setBaseYmOptions(months);
        setBaseYm((current) => current || String(to));
      })
      .catch(() => {});
  }, [token]);

  const runAnalysis = () => {
    if (!token || !baseYm) return;
    setLoading(true);
    setError(null);
    apiGet("/api/v3/realdata/analysis", {
      token,
      params: { region: "namwon", dong_code: dongCode, base_ym: baseYm, model_id: modelId || undefined }
    })
      .then((res) => setResult(res))
      .catch((err) => setError(err.message || "분석 조회에 실패했습니다."))
      .finally(() => setLoading(false));
  };

  const data = result?.data;

  return (
    <Card
      title="실데이터 진단 (남원)"
      icon="fa-magnifying-glass-chart"
      className="page-section realdata-panel"
      dataSource="api"
      headerRight={<span style={{ fontSize: 11, color: "var(--text-muted)" }}>현황 · 진단(rules_v1) · 예측 · 대응 후보</span>}
    >
      {authError && <p className="realdata-error">{authError}</p>}

      <div className="realdata-controls">
        <label className="realdata-field">
          <span>행정동</span>
          <select value={dongCode} onChange={(e) => setDongCode(e.target.value)}>
            <option value="all">전체(남원)</option>
            {dongOptions.map((d) => (
              <option key={d.dong_code} value={d.dong_code}>
                {d.dong_name}
              </option>
            ))}
          </select>
        </label>
        <label className="realdata-field">
          <span>기준월</span>
          <select value={baseYm} onChange={(e) => setBaseYm(e.target.value)} disabled={baseYmOptions.length === 0}>
            {baseYmOptions.map((ym) => (
              <option key={ym} value={ym}>
                {ym}
              </option>
            ))}
          </select>
        </label>
        <label className="realdata-field">
          <span>모델(선택)</span>
          <select value={modelId} onChange={(e) => setModelId(e.target.value)}>
            <option value="">전체</option>
            {Object.entries(MODEL_LABELS).map(([id, label]) => (
              <option key={id} value={id}>
                {label}
              </option>
            ))}
          </select>
        </label>
        <button className="btn btn-primary" onClick={runAnalysis} aria-disabled={!token || !baseYm || loading}>
          <i className="fa-solid fa-play"></i> 분석
        </button>
      </div>

      {error && <p className="realdata-error">{error}</p>}
      <StatusNote status={result?.status} message={result?.message} />

      {data && (
        <div className="realdata-analysis-grid">
          <section className="realdata-quadrant">
            <h4>현황</h4>
            <StatusNote status={data.current.status === "empty" ? "empty" : null} />
            {Object.values(data.current.metrics).map((m) => (
              <CurrentMetric key={m.field} metric={m} />
            ))}
          </section>

          <section className="realdata-quadrant">
            <h4>진단 ({data.diagnosis.rules_version})</h4>
            {data.diagnosis.findings.length === 0 ? (
              <p className="realdata-hint mock-data-output">
                진단 근거가 없습니다.
                {data.diagnosis.skipped.map((s) => (
                  <span key={s.target}> {s.reason}</span>
                ))}
              </p>
            ) : (
              <ul className="realdata-findings">
                {data.diagnosis.findings.map((f) => (
                  <Finding key={f.rule_id} finding={f} />
                ))}
              </ul>
            )}
          </section>

          <section className="realdata-quadrant">
            <h4>예측</h4>
            {Object.entries(data.forecast.models).map(([id, forecast]) => (
              <div key={id} className="realdata-forecast-model">
                <strong>{MODEL_LABELS[id] ?? id}</strong>
                {forecast.status !== "ok" ? (
                  <StatusNote status={forecast.status} message={forecast.message} />
                ) : (
                  <div className="mock-data-output">
                    <p>
                      {forecast.forecast_month} 예측 <strong>{fmtNum(forecast.prediction, 1)}</strong> · 기준선({forecast.baseline?.name}){" "}
                      {fmtNum(forecast.baseline?.value, 1)}
                    </p>
                    <p className="realdata-hint">
                      검증 MAE {fmtNum(forecast.validation?.metrics?.mae, 3)} (기준선 {fmtNum(forecast.validation?.baseline?.mae, 3)})
                    </p>
                  </div>
                )}
              </div>
            ))}
          </section>

          <section className="realdata-quadrant">
            <h4>대응 후보</h4>
            {data.responses.candidates.length === 0 ? (
              <StatusNote status={data.responses.status} message="근거가 부족해 대응 후보를 제시하지 않습니다." />
            ) : (
              <ul className="realdata-candidates">
                {data.responses.candidates.map((c) => (
                  <li key={c.rule_id} className="realdata-candidate mock-data-output">
                    <span className="realdata-badge">{c.category}</span>
                    <p>{c.text}</p>
                    <p className="realdata-hint">
                      근거: {c.based_on.dong_code} · {c.based_on.months.join(", ")}
                    </p>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </div>
      )}

      {data?.provenance && (
        <p className="realdata-hint mock-data-output" style={{ marginTop: 12 }}>
          데이터셋 {orDash(data.provenance.dataset_ids?.nonlocal_visitors)} / {orDash(data.provenance.dataset_ids?.observed_sales_krw)} ·
          규칙 {data.provenance.rules_version} · 계산 시각 {result?.provenance?.computed_at}
        </p>
      )}
    </Card>
  );
}
