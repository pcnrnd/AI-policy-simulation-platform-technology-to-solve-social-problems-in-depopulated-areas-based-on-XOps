import { useEffect, useMemo, useState } from "react";
import { Bar, Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from "chart.js";
import Card from "../components/Card.jsx";
import PerfBadge from "../components/PerfBadge.jsx";
import GaugeChart from "../components/GaugeChart.jsx";
import { useAppState } from "../context/AppStateContext.jsx";
import { useChartTheme } from "../hooks/useChartTheme.js";
import { useRenderTiming } from "../lib/perf.js";
import { MODEL_REGISTRY } from "../constants/models.js";
import InfoTip from "../components/InfoTip.jsx";
import { apiGet } from "../lib/api.js";

// 드리프트 자동 재학습 대상 — 인구이동 예측 모델(백엔드 오케스트레이션 시드 id)
const DRIFT_MODEL_ID = "population-forecast";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const fmtTime = (d) =>
  `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}:${String(d.getSeconds()).padStart(2, "0")}`;

const fmtHM = (d) => `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;

const COLLECT_REFRESH_MS = 30000;

// 모니터링 옵션 — 조회 구간 (대상 모델은 운영 모델 레지스트리에서 선택)
const WINDOW_OPTIONS = [3, 6, 10];

export default function MonitorPage() {
  const { appData, driftInjected, accuracyOverride, f1Override, pipelineRunning, injectDrift, addConsoleLog } =
    useAppState();
  const ct = useChartTheme();

  // 백엔드(/api/v3/monitoring) 실데이터 — 없으면 mock으로 폴백해 로딩 중에도 렌더.
  const [metricsResp, setMetricsResp] = useState(null);
  const [shapResp, setShapResp] = useState(null);
  const [driftResp, setDriftResp] = useState(null);

  useEffect(() => {
    let alive = true;
    Promise.all([apiGet("/api/v3/monitoring/metrics"), apiGet("/api/v3/monitoring/explain")])
      .then(([m, s]) => {
        if (!alive) return;
        setMetricsResp(m);
        setShapResp(s);
      })
      .catch((err) => addConsoleLog(`ERROR: 모니터링 지표 로드 실패 — ${err.message}`, false, true));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 드리프트 상태 변경 시 백엔드 PSI/KL 판정을 재조회(표시 전용).
  // 재학습 발화는 injectDrift → 오케스트레이션 이벤트 경로가 단독 담당(중복 트리거 방지).
  useEffect(() => {
    let alive = true;
    apiGet("/api/v3/monitoring/drift", { params: { drifted: driftInjected, model_id: DRIFT_MODEL_ID } })
      .then((d) => alive && setDriftResp(d))
      .catch((err) => addConsoleLog(`ERROR: 드리프트 조회 실패 — ${err.message}`, false, true));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [driftInjected]);

  // 마지막 수집 시각 — 탭 진입·드리프트 상태 변경 시 갱신, 30초 주기 자동 갱신
  const [lastCollected, setLastCollected] = useState(() => new Date());
  useEffect(() => {
    setLastCollected(new Date());
  }, [driftInjected]);
  useEffect(() => {
    const t = setInterval(() => setLastCollected(new Date()), COLLECT_REFRESH_MS);
    return () => clearInterval(t);
  }, []);

  // 지표 추이 x축 — 접속 시각 기준 최근 10시간(정시 라벨)
  const hourlyLabels = useMemo(() => {
    const now = new Date();
    return Array.from({ length: 10 }, (_, i) => {
      const d = new Date(now.getTime() - (9 - i) * 3600000);
      return `${String(d.getHours()).padStart(2, "0")}:00`;
    });
  }, []);

  // 모니터링 옵션 — 운영 모델 레지스트리에서 대상 모델 선택 + 조회 구간. 지표 추이 차트에 반영된다.
  const [modelTarget, setModelTarget] = useState(MODEL_REGISTRY[0].id);
  const [windowHours, setWindowHours] = useState(10);
  const targetModel = MODEL_REGISTRY.find((m) => m.id === modelTarget) ?? MODEL_REGISTRY[0];
  const modelLabel = `${targetModel.name} ${targetModel.version}`;

  const AXIS_OPTS = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: { grid: { color: ct.grid }, ticks: { color: ct.tick } },
      x: { grid: { display: false }, ticks: { color: ct.tick } }
    },
    plugins: { legend: { labels: { color: ct.legend } } }
  };

  const driftData = useMemo(() => {
    // 백엔드 응답(reference/current/buckets) 우선, 로딩 중엔 mock 폴백
    const buckets = driftResp?.buckets ?? appData.drift_distribution.buckets;
    const reference = driftResp?.reference ?? appData.drift_distribution.reference;
    const current =
      driftResp?.current ??
      (driftInjected
        ? appData.drift_distribution.current_drifted
        : appData.drift_distribution.current_normal);
    return {
      labels: buckets,
      datasets: [
        {
          label: "참조 분포 (Reference Dataset)",
          data: reference,
          backgroundColor: "rgba(59, 130, 246, 0.4)",
          borderColor: "rgba(59, 130, 246, 1)",
          borderWidth: 1.5,
          borderRadius: 4
        },
        {
          label: "실시간 유입 데이터 분포",
          data: current,
          backgroundColor: driftInjected ? "rgba(239, 68, 68, 0.65)" : "rgba(16, 185, 129, 0.5)",
          borderColor: driftInjected ? "rgba(239, 68, 68, 1)" : "rgba(16, 185, 129, 1)",
          borderWidth: 1.5,
          borderRadius: 4
        }
      ]
    };
  }, [appData, driftInjected, driftResp]);

  const metricsData = useMemo(() => {
    // 구간 슬라이스 + 모델별 결정적 오프셋(accDelta/errRatio)을 모든 지표 계열에 일괄 적용
    const tune = (arr, isError) =>
      arr.map((v) =>
        isError
          ? Number((v * targetModel.errRatio).toFixed(3))
          : Math.min(0.99, Math.max(0, Number((v + targetModel.accDelta).toFixed(3))))
      );
    const hist = metricsResp?.history ?? appData.metrics_history;
    const series = (key, isError = false) => tune(hist[key], isError).slice(-windowHours);

    return {
      labels: hourlyLabels.slice(-windowHours),
      datasets: [
        {
          label: "Accuracy",
          data: series("accuracy"),
          borderColor: "rgba(16, 185, 129, 1)",
          backgroundColor: "rgba(16, 185, 129, 0.05)",
          fill: true,
          tension: 0.35,
          borderWidth: 2
        },
        {
          label: "F1-Score",
          data: series("f1"),
          borderColor: "rgba(59, 130, 246, 1)",
          borderWidth: 2,
          pointStyle: "circle",
          tension: 0.35
        },
        {
          label: "Precision",
          data: series("precision"),
          borderColor: "rgba(34, 211, 238, 1)",
          borderWidth: 1.5,
          pointStyle: "triangle",
          tension: 0.35
        },
        {
          label: "Recall",
          data: series("recall"),
          borderColor: "rgba(168, 85, 247, 1)",
          borderWidth: 1.5,
          pointStyle: "rect",
          tension: 0.35
        },
        {
          label: "MSE",
          data: series("mse", true),
          borderColor: "rgba(239, 68, 68, 1)",
          borderWidth: 1.5,
          borderDash: [5, 5],
          tension: 0.35
        },
        {
          label: "MAE",
          data: series("mae", true),
          borderColor: "rgba(251, 146, 60, 1)",
          borderWidth: 1.5,
          borderDash: [2, 3],
          tension: 0.35
        }
      ]
    };
  }, [appData, hourlyLabels, targetModel, windowHours, metricsResp]);

  const shapData = useMemo(() => {
    const feats = shapResp?.features ?? appData.shap_features;
    return {
      labels: feats.map((f) => f.feature.split(" (")[0]),
      datasets: [
        {
          label: "SHAP 기여도 기여 지수 (음수일수록 유출 기여)",
          data: feats.map((f) => f.value),
          backgroundColor: feats.map((f) =>
            f.value > 0 ? "rgba(16, 185, 129, 0.65)" : "rgba(239, 68, 68, 0.65)"
          ),
          borderColor: feats.map((f) => (f.value > 0 ? "rgba(16, 185, 129, 1)" : "rgba(239, 68, 68, 1)")),
          borderWidth: 1.5,
          borderRadius: 4
        }
      ]
    };
  }, [appData, shapResp]);

  const shapOpts = {
    indexAxis: "y",
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: { grid: { color: ct.grid }, ticks: { color: ct.tick } },
      y: { grid: { display: false }, ticks: { color: ct.tickStrong, font: { size: 10 } } }
    },
    plugins: { legend: { display: false } }
  };

  const vizMs = useRenderTiming([driftData, metricsData, shapData]);

  // 승급 오버라이드(파이프라인 완료) 우선, 없으면 백엔드 최신 지표, 그다음 mock 폴백
  const latest = metricsResp?.latest;
  const accVal =
    accuracyOverride !== null ? accuracyOverride.toFixed(3) : (latest?.accuracy ?? 0.892).toFixed(3);
  const f1Val = f1Override !== null ? f1Override.toFixed(3) : (latest?.f1 ?? 0.884).toFixed(3);

  // PSI·드리프트 판정은 백엔드 실계산값(driftResp) 사용
  const drifted = driftResp?.drifted ?? driftInjected;
  const psiVal = driftResp ? driftResp.psi.toFixed(3) : driftInjected ? "0.384" : "0.045";
  const psiColor = drifted ? "var(--accent-red)" : "var(--accent-teal)";
  const driftLabel = drifted ? "위험 (Drift)" : "정상";
  const driftLabelStyle = drifted
    ? { backgroundColor: "rgba(239, 68, 68, 0.15)", color: "var(--accent-red)" }
    : { backgroundColor: "rgba(16, 185, 129, 0.1)", color: "var(--accent-teal)" };

  // 이상값 로그 시각 — 현재 시각 기준 상대 시각으로 산출(정적 mock 노출 방지)
  const ago = (minutes) => fmtHM(new Date(Date.now() - minutes * 60000));
  const outlierRows = driftInjected
    ? [
        { time: ago(1), target: "남원시 데이터 (스마트팜 소득)", z: "3.45", outlier: true },
        { time: ago(2), target: "신안군 데이터 (임대주택 활용도)", z: "2.89", outlier: true },
        { time: ago(29), target: "남원시 데이터", z: "1.24", outlier: false }
      ]
    : [
        { time: ago(26), target: "남원시 데이터", z: "1.24", outlier: false },
        { time: ago(100), target: "신안군 데이터", z: "0.98", outlier: false },
        { time: ago(175), target: "남원시 데이터", z: "1.67", outlier: false }
      ];

  const outlierCount = driftInjected ? "3건" : "0건";
  const outlierCountColor = driftInjected ? "var(--accent-red)" : "var(--text-primary)";

  return (
    <>
      {/* 운영 툴바 — 수집 상태 + 이상 시나리오 재현(드리프트 → 자동 재학습 검증) */}
      <div className="monitor-toolbar">
        <span className="monitor-collected">
          <i className="fa-solid fa-satellite-dish" aria-hidden="true"></i> 마지막 수집:{" "}
          {fmtTime(lastCollected)} · 6개 연계 데이터 소스 정상 수신
        </span>
        <div className="monitor-options">
          <select
            className="select-control"
            style={{ padding: "6px 8px", fontSize: 12, width: "auto" }}
            value={modelTarget}
            onChange={(e) => setModelTarget(e.target.value)}
            aria-label="모니터링 대상 모델"
          >
            {MODEL_REGISTRY.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name} {m.version}
              </option>
            ))}
          </select>
          <select
            className="select-control"
            style={{ padding: "6px 8px", fontSize: 12, width: "auto" }}
            value={windowHours}
            onChange={(e) => setWindowHours(Number(e.target.value))}
            aria-label="조회 구간"
          >
            {WINDOW_OPTIONS.map((h) => (
              <option key={h} value={h}>
                최근 {h}시간
              </option>
            ))}
          </select>
        </div>
        <button
          type="button"
          className="btn btn-secondary"
          style={{ padding: "6px 14px", fontSize: 12 }}
          onClick={injectDrift}
          disabled={pipelineRunning || driftInjected}
          title="드리프트 유입 상황을 재현하여 감지 → 알림 → 자동 재학습 파이프라인을 검증합니다"
        >
          <i className="fa-solid fa-vial-circle-check" aria-hidden="true"></i>{" "}
          {driftInjected ? "드리프트 대응 진행 중..." : "이상 시나리오 재현 (드리프트 시뮬레이션)"}
        </button>
      </div>

      <div className="grid-cols-3" style={{ marginBottom: 24 }}>
        <div className="card" style={{ padding: 18 }}>
          <div className="stat-label">
            Model Accuracy / F1-Score
            <InfoTip text="운영 중인 모델의 정확도(Accuracy)와 정밀도·재현율의 조화평균(F1-Score). 재학습 승급 시 두 값이 함께 갱신됩니다." />
          </div>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "baseline",
              marginTop: 10
            }}
          >
            <span className="stat-value" style={{ fontSize: 32 }}>
              {accVal}
            </span>
            <span className="trend-up">F1: {f1Val}</span>
          </div>
          <p style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 8 }}>
            분산 지표 통합 관리로 사일로(Silo) 제거 — 6대 평가지표 실시간 자동 집계
          </p>
        </div>

        <div className={"card" + (driftInjected ? " glow-red" : "")} style={{ padding: 18 }}>
          <div className="stat-label">
            Data Drift Status (PSI)
            <InfoTip text="PSI(Population Stability Index)는 원본 학습 분포와 실시간 유입 분포의 차이를 측정합니다. 0.2를 초과하면 데이터 드리프트로 판정해 자동 재학습을 트리거합니다." />
          </div>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "baseline",
              marginTop: 10
            }}
          >
            <span className="stat-value" style={{ fontSize: 32, color: psiColor }}>
              {psiVal}
            </span>
            <span
              className="system-status"
              style={{ padding: "2px 8px", fontSize: 11, ...driftLabelStyle }}
            >
              {driftLabel}
            </span>
          </div>
          <p style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 8 }}>
            임계치 PSI {">"} 0.2 초과 시 자동 Alert 트리거
          </p>
        </div>

        <div className="card" style={{ padding: 18 }}>
          <div className="stat-label">
            Outlier Detection (Z-Score)
            <InfoTip text="유입 데이터의 Z-score(평균 대비 표준편차 거리)가 임계치를 넘는 이상치 건수입니다. 학습데이터 품질 저하를 사전에 차단합니다." />
          </div>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "baseline",
              marginTop: 10
            }}
          >
            <span className="stat-value" style={{ fontSize: 32, color: outlierCountColor }}>
              {outlierCount}
            </span>
            <span
              className="system-status"
              style={{
                padding: "2px 8px",
                fontSize: 11,
                backgroundColor: driftInjected ? "rgba(239, 68, 68, 0.1)" : "rgba(16, 185, 129, 0.1)",
                color: driftInjected ? "var(--accent-red)" : "var(--accent-teal)"
              }}
            >
              {driftInjected ? "경고" : "정상"}
            </span>
          </div>
          <p style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 8 }}>
            IQR 및 Z-score 기반 다차원 이상치 필터링
          </p>
        </div>
      </div>

      <div className="grid-details-split">
        <Card
          title="데이터 분포 변화 시각화 (참조 vs 최근유입)"
          icon="fa-chart-area"
          headerRight={<PerfBadge ms={vizMs} />}
        >
          <div style={{ position: "relative", height: 320, width: "100%" }}>
            <Bar data={driftData} options={AXIS_OPTS} />
          </div>
        </Card>

        <Card title="이상값 검출 로그" icon="fa-filter">
          <div style={{ maxHeight: 320, overflowY: "auto" }}>
            <table style={{ width: "100%" }}>
              <thead>
                <tr>
                  <th>시간</th>
                  <th>지자체</th>
                  <th>Z-score</th>
                  <th>상태</th>
                </tr>
              </thead>
              <tbody>
                {outlierRows.map((row, idx) => (
                  <tr key={idx}>
                    <td
                      style={{
                        color: row.outlier ? "var(--accent-red)" : "var(--text-secondary)"
                      }}
                    >
                      {row.time}
                    </td>
                    <td>{row.target}</td>
                    <td>{row.z}</td>
                    <td>
                      {row.outlier ? (
                        <span className="outlier-tag">Outlier</span>
                      ) : (
                        <span
                          className="system-status"
                          style={{
                            padding: "1px 6px",
                            fontSize: 10,
                            backgroundColor: "rgba(16, 185, 129, 0.1)",
                            color: "var(--accent-teal)"
                          }}
                        >
                          Normal
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>

      <div className="grid-cols-2">
        <Card
          title="MLOps 6대 핵심 평가지표 실시간 모니터링"
          icon="fa-chart-column"
          headerRight={
            <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
              {modelLabel} · 최근 {windowHours}시간
            </span>
          }
        >
          <div style={{ position: "relative", height: 280, width: "100%" }}>
            <Line data={metricsData} options={AXIS_OPTS} />
          </div>
        </Card>

        <Card title="SHAP 기반 인구 유출 기여 특징 중요도 분석" icon="fa-brain">
          <div style={{ position: "relative", height: 280, width: "100%" }}>
            <Bar data={shapData} options={shapOpts} />
          </div>
        </Card>
      </div>

      {/* 상단 stat 카드(Accuracy·F1)·6대 지표 차트와 중복되지 않는 지표만 게이지로 표시 */}
      <Card
        title={
          <>
            모델 신뢰도 게이지 (실시간)
            <InfoTip text="운영 모델의 Precision·Recall과 예측 지연(latency)을 표시합니다. 예측 지연이 자동 롤백 임계 200ms를 초과하면 직전 버전으로 자동 롤백됩니다." />
          </>
        }
        icon="fa-gauge-high"
      >
        <div className="grid-cols-3" style={{ marginBottom: 0 }}>
          <GaugeChart value={driftInjected ? 0.803 : latest?.precision ?? 0.891} label="Precision" />
          <GaugeChart value={driftInjected ? 0.788 : latest?.recall ?? 0.878} label="Recall" />
          <GaugeChart
            value={(driftInjected ? 178 : 120) / 200}
            displayText={`${driftInjected ? 178 : 120}ms`}
            label="예측 지연"
            goodThreshold={0.75}
            lowerIsBetter
          />
        </div>
      </Card>
    </>
  );
}
