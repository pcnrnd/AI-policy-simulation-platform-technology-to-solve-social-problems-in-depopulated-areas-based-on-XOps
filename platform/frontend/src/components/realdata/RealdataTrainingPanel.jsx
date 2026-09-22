// 실데이터 학습(남원) 패널 — Orchestrator 하단에 추가되는 신규 패널(기존 데모 경로는 불변).
// R3: 스냅샷 생성 → 학습 실행(job 폴링) → 후보 표 → 반영/복원.
import { useCallback, useEffect, useRef, useState } from "react";
import Card from "../Card.jsx";
import {
  applyCandidate,
  createDataset,
  getCandidates,
  getDatasets,
  getModels,
  getTrainingRun,
  getRealdataToken,
  restoreCandidate,
  startTrainingRun
} from "./realdataClient.jsx";

const MODEL_LABELS = {
  "namwon-nonlocal-visitors-next-month": "남원 타지역 방문객 (익월)",
  "namwon-observed-sales-next-month": "남원 관측 소비 (익월)"
};

const JOB_POLL_MS = 3000;
const TERMINAL_STATES = new Set(["saved", "failed", "cancelled"]);

const orDash = (v) => (v === null || v === undefined || v === "" ? "–" : v);
const fmtNum = (v) => (typeof v === "number" ? v.toLocaleString(undefined, { maximumFractionDigits: 3 }) : "–");

export default function RealdataTrainingPanel() {
  const [token, setToken] = useState(null);
  const [authError, setAuthError] = useState(null);

  const [models, setModels] = useState([]);
  const [modelsError, setModelsError] = useState(null);
  const [selectedModelId, setSelectedModelId] = useState("");

  const [datasets, setDatasets] = useState([]);
  const [datasetsError, setDatasetsError] = useState(null);
  const [selectedDatasetId, setSelectedDatasetId] = useState("");

  const [candidates, setCandidates] = useState([]);
  const [candidatesStatus, setCandidatesStatus] = useState(null);

  const [job, setJob] = useState(null);
  const [jobError, setJobError] = useState(null);
  const [snapshotBusy, setSnapshotBusy] = useState(false);
  const [decisionBusy, setDecisionBusy] = useState(null); // version 문자열 — 반영/복원 진행 중 표시

  const pollRef = useRef(null);
  const candidatesRequestRef = useRef(0);

  // ── 토큰 발급(자동, 1회) ─────────────────────────────────
  useEffect(() => {
    getRealdataToken()
      .then(setToken)
      .catch((err) => setAuthError(err.message || "인증 토큰 발급에 실패했습니다."));
  }, []);

  // ── 모델 목록 ────────────────────────────────────────────
  const loadModels = useCallback(() => {
    if (!token) return;
    getModels(token)
      .then((res) => {
        const list = res.data ?? [];
        setModels(list);
        setModelsError(res.status !== "ok" ? res.message ?? "모델 목록을 불러오지 못했습니다." : null);
        setSelectedModelId((current) => current || list[0]?.model_id || "");
      })
      .catch((err) => setModelsError(err.message || "모델 목록 조회 실패"));
  }, [token]);

  useEffect(() => {
    loadModels();
  }, [loadModels]);

  // ── 데이터셋 목록(A 라우트 GET /realdata/datasets → data:{datasets:[...]}) ─────────
  useEffect(() => {
    if (!token) return;
    getDatasets(token)
      .then((res) => {
        const list = res.data?.datasets ?? [];
        setDatasets(list);
        setDatasetsError(res.status !== "ok" ? res.message ?? "등록된 데이터셋이 없습니다." : null);
      })
      .catch((err) => setDatasetsError(err.message || "데이터셋 조회 라우트를 사용할 수 없습니다 (작성자 A 담당)."));
  }, [token]);

  // ── 후보 표(선택 모델 변경 시 재조회) ─────────────────────
  const loadCandidates = useCallback(() => {
    if (!token || !selectedModelId) return;
    const requestId = ++candidatesRequestRef.current;
    getCandidates(token, selectedModelId)
      .then((res) => {
        if (requestId !== candidatesRequestRef.current) return;
        setCandidates(res.data ?? []);
        setCandidatesStatus(res.status);
      })
      .catch((err) => {
        if (requestId !== candidatesRequestRef.current) return;
        setCandidatesStatus("error");
        setJobError(err.message || "후보 조회 실패");
      });
  }, [token, selectedModelId]);

  useEffect(() => {
    loadCandidates();
  }, [loadCandidates]);

  // ── job 폴링 ─────────────────────────────────────────────
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const pollJob = useCallback(
    (jobId) => {
      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = setInterval(() => {
        getTrainingRun(token, jobId)
          .then((res) => {
            setJob(res.data);
            if (res.data && TERMINAL_STATES.has(res.data.state)) {
              clearInterval(pollRef.current);
              pollRef.current = null;
              loadModels();
              loadCandidates();
            }
          })
          .catch((err) => setJobError(err.message || "작업 상태 조회 실패"));
      }, JOB_POLL_MS);
    },
    [token, loadModels, loadCandidates]
  );

  const handleCreateSnapshot = () => {
    if (!token || !selectedModelId || snapshotBusy) return;
    setSnapshotBusy(true);
    setDatasetsError(null);
    createDataset(token, selectedModelId)
      .then((res) => {
        if (res.status !== "ok") {
          setDatasetsError(res.message ?? "스냅샷 생성에 실패했습니다.");
          return;
        }
        setDatasets((prev) => [res.data, ...prev]);
        setSelectedDatasetId(res.data?.dataset_id ?? "");
      })
      .catch((err) => setDatasetsError(err.message || "스냅샷 생성 라우트를 사용할 수 없습니다 (작성자 A 담당)."))
      .finally(() => setSnapshotBusy(false));
  };

  const handleStartTraining = () => {
    if (!token || !selectedModelId || !selectedDatasetId) return;
    setJobError(null);
    setJob(null);
    startTrainingRun(token, selectedModelId, selectedDatasetId)
      .then((res) => {
        setJob(res.data);
        pollJob(res.data.job_id);
      })
      .catch((err) => setJobError(err.status === 409 ? "이미 활성 학습 작업이 있습니다." : err.message || "학습 실행 요청 실패"));
  };

  const handleApply = (version) => {
    if (!token || decisionBusy) return;
    if (!window.confirm(`후보 ${version}을(를) 반영하시겠습니까? 현재 활성 모델을 교체합니다.`)) return;
    setDecisionBusy(version);
    applyCandidate(token, selectedModelId, version)
      .then(() => {
        loadModels();
        loadCandidates();
      })
      .catch((err) => {
        const reasons = err.body?.detail?.reasons;
        setJobError(reasons ? `반영 조건 미충족: ${reasons.join(" / ")}` : err.message || "반영 실패");
      })
      .finally(() => setDecisionBusy(null));
  };

  const handleRestore = (version) => {
    if (!token || decisionBusy) return;
    if (!window.confirm(`후보 ${version}을(를) 복원하시겠습니까? 현재 활성 모델을 대체합니다.`)) return;
    setDecisionBusy(version);
    restoreCandidate(token, selectedModelId, version)
      .then(() => {
        loadModels();
        loadCandidates();
      })
      .catch((err) => setJobError(err.message || "복원 실패"))
      .finally(() => setDecisionBusy(null));
  };

  const selectedModel = models.find((m) => m.model_id === selectedModelId);
  const jobDone = !job || TERMINAL_STATES.has(job.state);

  return (
    <Card
      title="실데이터 학습"
      icon="fa-flask-vial"
      className="page-section realdata-panel"
      dataSource="api"
      headerRight={
        <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
          남원 23동 실데이터 · Ridge 공동 학습 · 후보 승인 후 반영
        </span>
      }
    >
      {authError && <p className="realdata-error">{authError}</p>}

      <div className="realdata-controls">
        <label className="realdata-field">
          <span>대상 모델</span>
          <select value={selectedModelId} onChange={(e) => setSelectedModelId(e.target.value)}>
            {models.map((m) => (
              <option key={m.model_id} value={m.model_id}>
                {MODEL_LABELS[m.model_id] ?? m.model_id}
              </option>
            ))}
          </select>
        </label>

        <label className="realdata-field">
          <span>데이터셋</span>
          <select
            value={selectedDatasetId}
            onChange={(e) => setSelectedDatasetId(e.target.value)}
            disabled={datasets.length === 0}
          >
            <option value="">선택하세요</option>
            {datasets.map((d) => (
              <option key={d.dataset_id} value={d.dataset_id}>
                {d.dataset_id} ({orDash(d.observed_from)}~{orDash(d.observed_to)})
              </option>
            ))}
          </select>
        </label>

        <button className="btn btn-secondary" onClick={handleCreateSnapshot} aria-disabled={!token || snapshotBusy}>
          <i className={`fa-solid ${snapshotBusy ? "fa-spinner fa-spin" : "fa-camera"}`}></i> 스냅샷 생성
        </button>
        <button
          className="btn btn-primary"
          onClick={handleStartTraining}
          aria-disabled={!token || !selectedModelId || !selectedDatasetId || !jobDone}
          title={!jobDone ? "현재 실행 중인 학습이 끝난 뒤 실행할 수 있습니다" : "선택한 데이터셋으로 학습을 실행합니다"}
        >
          <i className="fa-solid fa-play"></i> 학습 실행
        </button>
      </div>

      {datasetsError && <p className="realdata-hint mock-data-output">{datasetsError}</p>}

      {selectedModel && (
        <p className="realdata-hint mock-data-output">
          활성 버전: <code>{orDash(selectedModel.active_version)}</code>
          {selectedModel.retrain_needed && <span className="realdata-badge realdata-badge-warn">재학습 필요</span>}
        </p>
      )}

      {job && (
        <p className="realdata-hint mock-data-output">
          작업 <code>{job.job_id}</code> — 상태 <strong>{job.state}</strong>
          {job.error && ` (${job.error})`}
        </p>
      )}
      {jobError && <p className="realdata-error mock-data-output">{jobError}</p>}

      <div className="table-container mock-data-output">
        <table>
          <caption className="sr-only">모델 후보 버전과 지표, 반영·복원 동작</caption>
          <thead>
            <tr>
              <th scope="col">버전</th>
              <th scope="col" className="cell-num">MAE / RMSE / WAPE</th>
              <th scope="col" className="cell-num">기준선 MAE</th>
              <th scope="col">상태</th>
              <th scope="col" className="cell-actions">동작</th>
            </tr>
          </thead>
          <tbody>
            {candidatesStatus === "empty" && (
              <tr>
                <td colSpan={5} style={{ color: "var(--text-muted)" }}>등록된 후보가 없습니다.</td>
              </tr>
            )}
            {candidates.map((c) => (
              <tr key={c.version}>
                <td><code>{c.version}</code></td>
                <td className="cell-num">
                  {fmtNum(c.metrics?.mae)} / {fmtNum(c.metrics?.rmse)} / {fmtNum(c.metrics?.wape)}
                </td>
                <td className="cell-num">{fmtNum(c.baseline?.mae)}</td>
                <td>
                  <span className="system-status" style={{ padding: "1px 8px", fontSize: 10 }}>
                    {orDash(c.status)}
                  </span>
                </td>
                <td className="cell-actions">
                  <button
                    className="btn btn-secondary"
                    style={{ padding: "4px 10px", fontSize: 11 }}
                    onClick={() => handleApply(c.version)}
                    aria-disabled={c.status !== "candidate" || Boolean(decisionBusy)}
                  >
                    반영
                  </button>{" "}
                  <button
                    className="btn btn-secondary"
                    style={{ padding: "4px 10px", fontSize: 11 }}
                    onClick={() => handleRestore(c.version)}
                    aria-disabled={!["applied", "superseded"].includes(c.status) || Boolean(decisionBusy)}
                  >
                    복원
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
