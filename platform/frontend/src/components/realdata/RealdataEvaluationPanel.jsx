// 실데이터 모델 평가(남원) 패널 — Monitor 하단에 추가되는 신규 패널(기존 데모 경로는 불변).
// R4: evaluation(validation/operational) · explain(선형 SHAP, 기존 Bar 차트 재사용) · drift(PSI).
import { useCallback, useEffect, useRef, useState } from "react";
import { Bar } from "react-chartjs-2";
import Card from "../Card.jsx";
import {
  getCandidates,
  getDrift,
  getEvaluation,
  getExplain,
  getModels,
  issueRealdataToken
} from "./realdataClient.jsx";

const MODEL_LABELS = {
  "namwon-nonlocal-visitors-next-month": "남원 타지역 방문객 (익월)",
  "namwon-observed-sales-next-month": "남원 관측 소비 (익월)"
};

const orDash = (v) => (v === null || v === undefined || v === "" ? "–" : v);
const fmtNum = (v, digits = 3) => (typeof v === "number" ? v.toFixed(digits) : "–");

const STATUS_LABELS = {
  ok: "정상",
  empty: "값 없음",
  pending: "정답 대기",
  model_required: "활성 모델 필요",
  insufficient_data: "데이터 부족",
  error: "오류"
};

function StatusNote({ status, message }) {
  if (status === "ok") return null;
  return <p className="realdata-hint mock-data-output">{STATUS_LABELS[status] ?? status}{message ? ` — ${message}` : ""}</p>;
}

export default function RealdataEvaluationPanel() {
  const [token, setToken] = useState(null);
  const [authError, setAuthError] = useState(null);

  const [models, setModels] = useState([]);
  const [selectedModelId, setSelectedModelId] = useState("");

  const [candidates, setCandidates] = useState([]);
  const [selectedVersion, setSelectedVersion] = useState("");

  const [evaluation, setEvaluation] = useState(null);
  const [drift, setDrift] = useState(null);
  const [explain, setExplain] = useState(null);
  const [explainStatus, setExplainStatus] = useState(null);
  const [baseYm, setBaseYm] = useState("");
  const [dongCode, setDongCode] = useState("");

  const requestRef = useRef(0);

  useEffect(() => {
    issueRealdataToken()
      .then(setToken)
      .catch((err) => setAuthError(err.message || "인증 토큰 발급에 실패했습니다."));
  }, []);

  useEffect(() => {
    if (!token) return;
    getModels(token).then((res) => {
      const list = res.data ?? [];
      setModels(list);
      setSelectedModelId((current) => current || list[0]?.model_id || "");
    });
  }, [token]);

  useEffect(() => {
    if (!token || !selectedModelId) return;
    getCandidates(token, selectedModelId).then((res) => {
      const list = res.data ?? [];
      setCandidates(list);
      const active = models.find((m) => m.model_id === selectedModelId)?.active_version;
      setSelectedVersion((current) => {
        if (current && list.some((c) => c.version === current)) return current;
        return active ?? list[0]?.version ?? "";
      });
    });
    // models는 selectedModelId가 바뀔 때 이미 최신이므로 의존성에서 제외해 매 렌더 재조회를 막는다.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token, selectedModelId]);

  useEffect(() => {
    if (!token || !selectedModelId || !selectedVersion) return;
    const requestId = ++requestRef.current;
    Promise.all([
      getEvaluation(token, selectedModelId, selectedVersion),
      getDrift(token, selectedModelId, selectedVersion)
    ]).then(([evalRes, driftRes]) => {
      if (requestId !== requestRef.current) return;
      setEvaluation(evalRes);
      setDrift(driftRes);
    });
    setExplain(null);
    setExplainStatus(null);
  }, [token, selectedModelId, selectedVersion]);

  const runExplain = useCallback(() => {
    if (!token || !selectedModelId || !selectedVersion || !baseYm || !dongCode) return;
    getExplain(token, selectedModelId, selectedVersion, Number(baseYm), dongCode).then((res) => {
      setExplain(res.data);
      setExplainStatus(res.status);
    });
  }, [token, selectedModelId, selectedVersion, baseYm, dongCode]);

  const validation = evaluation?.data?.validation;
  const operational = evaluation?.data?.operational;
  const driftData = drift?.data;

  const chartData = explain
    ? {
        labels: explain.contributions.map((c) => c.feature),
        datasets: [
          {
            label: "기여도(phi)",
            data: explain.contributions.map((c) => c.phi),
            backgroundColor: explain.contributions.map((c) => (c.phi >= 0 ? "rgba(59,130,246,0.6)" : "rgba(239,68,68,0.6)"))
          }
        ]
      }
    : null;

  return (
    <Card
      title="실데이터 모델 평가 (남원)"
      icon="fa-chart-line"
      className="page-section realdata-panel"
      dataSource="api"
      headerRight={
        <span style={{ fontSize: 11, color: "var(--text-muted)" }}>검증·운영 평가 · 선형 SHAP · 드리프트(PSI)</span>
      }
    >
      {authError && <p className="realdata-error">{authError}</p>}

      <div className="realdata-controls">
        <label className="realdata-field">
          <span>모델</span>
          <select value={selectedModelId} onChange={(e) => setSelectedModelId(e.target.value)}>
            {models.map((m) => (
              <option key={m.model_id} value={m.model_id}>
                {MODEL_LABELS[m.model_id] ?? m.model_id}
              </option>
            ))}
          </select>
        </label>
        <label className="realdata-field">
          <span>버전</span>
          <select value={selectedVersion} onChange={(e) => setSelectedVersion(e.target.value)} disabled={candidates.length === 0}>
            {candidates.map((c) => (
              <option key={c.version} value={c.version}>
                {c.version} ({c.status})
              </option>
            ))}
          </select>
        </label>
      </div>

      <StatusNote status={evaluation?.status} message={evaluation?.message} />

      {validation && (
        <div className="table-container mock-data-output">
          <table>
            <caption className="sr-only">검증·운영 평가 지표</caption>
            <thead>
              <tr>
                <th scope="col">구분</th>
                <th scope="col" className="cell-num">MAE</th>
                <th scope="col" className="cell-num">RMSE</th>
                <th scope="col" className="cell-num">WAPE</th>
                <th scope="col">비고</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>검증(validation)</td>
                <td className="cell-num">{fmtNum(validation.metrics?.mae)}</td>
                <td className="cell-num">{fmtNum(validation.metrics?.rmse)}</td>
                <td className="cell-num">{fmtNum(validation.metrics?.wape)}</td>
                <td style={{ fontSize: 11, color: "var(--text-muted)" }}>
                  기준선 MAE {fmtNum(validation.baseline?.mae)}
                </td>
              </tr>
              <tr>
                <td>운영(operational)</td>
                {operational?.kind === "operational" ? (
                  <>
                    <td className="cell-num">{fmtNum(operational.metrics?.mae)}</td>
                    <td className="cell-num">{fmtNum(operational.metrics?.rmse)}</td>
                    <td className="cell-num">{fmtNum(operational.metrics?.wape)}</td>
                    <td style={{ fontSize: 11, color: "var(--text-muted)" }}>n={operational.n}</td>
                  </>
                ) : (
                  <td colSpan={4} style={{ color: "var(--text-muted)" }}>
                    {operational?.kind === "pending"
                      ? `정답 대기 — ${orDash(operational.pending_months?.join(", "))}`
                      : orDash(operational?.message)}
                  </td>
                )}
              </tr>
            </tbody>
          </table>
        </div>
      )}

      <div className="realdata-controls">
        <label className="realdata-field">
          <span>기준월(YYYYMM)</span>
          <input type="number" value={baseYm} onChange={(e) => setBaseYm(e.target.value)} placeholder="202310" />
        </label>
        <label className="realdata-field">
          <span>행정동 코드</span>
          <input type="text" value={dongCode} onChange={(e) => setDongCode(e.target.value)} placeholder="45190250" />
        </label>
        <button className="btn btn-secondary" onClick={runExplain} aria-disabled={!token || !baseYm || !dongCode}>
          <i className="fa-solid fa-magnifying-glass-chart"></i> 기여도 조회
        </button>
      </div>

      <StatusNote status={explainStatus} message={explain?.message} />

      {chartData && (
        <div className="realdata-shap mock-data-output">
          <Bar
            data={chartData}
            options={{ indexAxis: "y", responsive: true, plugins: { legend: { display: false } } }}
            height={160}
          />
          <p className="realdata-hint">
            기준값 {fmtNum(explain.base_value)} · 예측값 {fmtNum(explain.prediction)} · 재구성 오차{" "}
            {fmtNum(explain.reconstruction_check, 8)} · {explain.note}
          </p>
        </div>
      )}

      <StatusNote status={drift?.status} message={drift?.message} />
      {driftData && driftData.status === "none" && (
        <p className="realdata-hint mock-data-output">드리프트 비교 구간이 아직 없습니다(관측·검증 구간 부족).</p>
      )}
      {driftData && driftData.status === "ok" && (
        <div className="table-container mock-data-output">
          <table>
            <caption className="sr-only">피처·타깃 드리프트(PSI)</caption>
            <thead>
              <tr>
                <th scope="col">항목</th>
                <th scope="col" className="cell-num">PSI</th>
                <th scope="col">구간 종류</th>
                <th scope="col" className="cell-num">표본(기준/현재)</th>
              </tr>
            </thead>
            <tbody>
              {driftData.features.map((f) => (
                <tr key={f.feature}>
                  <td>{f.feature}</td>
                  <td className="cell-num">{fmtNum(f.psi, 4)}</td>
                  <td>{driftData.kind}</td>
                  <td className="cell-num">{f.n_reference} / {f.n_current}</td>
                </tr>
              ))}
              {driftData.target && (
                <tr>
                  <td>타깃(y)</td>
                  <td className="cell-num">{fmtNum(driftData.target.psi, 4)}</td>
                  <td>{driftData.kind}</td>
                  <td className="cell-num">{driftData.target.n_reference} / {driftData.target.n_current}</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
