import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
import NextStepBanner from "../components/NextStepBanner.jsx";
import PerfBadge from "../components/PerfBadge.jsx";
import GaugeChart from "../components/GaugeChart.jsx";
import { useAppState } from "../context/AppStateContext.jsx";
import { useChartTheme } from "../hooks/useChartTheme.js";
import { useRenderTiming } from "../lib/perf.js";
import { MODEL_REGISTRY, RETRAIN_PIPELINES } from "../constants/models.js";
import InfoTip from "../components/InfoTip.jsx";
import RealdataEvaluationPanel from "../components/realdata/RealdataEvaluationPanel.jsx";
import { apiGet, apiSend } from "../lib/api.js";

// 드리프트 자동 재학습 대상 — 인구이동 예측 모델(백엔드 오케스트레이션 시드 id).
// PSI·이상값·예측 지연 값 자체는 이 모델 하나만 관측한다(대상 모델 선택과 무관, 값 확장은 별도 결정 사항).
const DRIFT_MODEL_ID = "population-forecast";

// 예측 지연 게이지 — 자동 롤백 임계(200ms) 대비 비율로 표시한다.
const LATENCY_ROLLBACK_MS = 200;
// 데모 시드 지연. 실측(학습 시 측정한 1건 추론 지연)이 없고 데모 표시가 켜져 있을 때만 쓴다.
const SEED_LATENCY_NORMAL_MS = 120;
const SEED_LATENCY_DRIFTED_MS = 178;
// 데모 시드 PSI — 위와 같은 조건에서만 쓴다.
const SEED_PSI_DRIFTED = "0.384";
const SEED_PSI_NORMAL = "0.045";

// 값이 없을 때 쓰는 표기 — 요소를 감추지 않고 자리만 비운다(§10 UI-04).
const NO_VALUE = "–";
const EMPTY_HINT = "실데이터가 아직 없습니다";

// 차트 자리를 비우되 캔버스는 그대로 둔다 — 요소를 숨기지 않고 "값이 없다"만 알린다(§10 UI-04).
function ChartEmptyNote({ children }) {
  return (
    <p className="monitor-empty-note" role="status">
      {children}
    </p>
  );
}

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

// 백엔드 성능 계약의 6대 지표 계열(값은 오차 지표 여부 — errRatio 보정 대상).
// 파생 계열과 수집 상태 판정이 같은 목록을 봐야 "정상 수신" 주장과 실제 폴백이 어긋나지 않는다.
const METRIC_SERIES = {
  accuracy: false,
  f1: false,
  precision: false,
  recall: false,
  mse: true,
  mae: true
};

// 모니터링 옵션 — 조회 구간 (대상 모델은 운영 모델 레지스트리에서 선택)
const WINDOW_OPTIONS = [3, 6, 10];

export default function MonitorPage() {
  const {
    appData,
    driftInjected,
    metricOverrides,
    modelStore,
    pipelineRunning,
    pipelineScheduled,
    pipelineRun,
    pipelineResult,
    injectDrift,
    startPipeline,
    navigateToTab,
    addConsoleLog,
    setMonitorCollectStatus,
    modelCandidates,
    mockDataVisible
  } = useAppState();
  const ct = useChartTheme();

  // 데모 표시 토글은 화면 구조가 아니라 **데이터 유무**만 바꾼다. OFF에서는 시드(mock_data.json)
  // 유래 값이 실값 자리에 들어가지 않도록 여기 데이터 계층에서 차단하고, 빈 자리는 "–"로 남긴다.
  const allowSeed = mockDataVisible;

  // 모니터링 옵션 — 운영 모델 레지스트리에서 대상 모델 선택 + 조회 구간. 지표 추이 차트에 반영된다.
  // 대상 모델은 실측 지표 조회 키이기도 하므로 로더보다 먼저 선언한다.
  const [modelTarget, setModelTarget] = useState(MODEL_REGISTRY[0].id);
  const [windowHours, setWindowHours] = useState(10);

  // 백엔드(/api/v3/monitoring) 실데이터 — 없으면 mock으로 폴백해 로딩 중에도 렌더.
  const [metricsResp, setMetricsResp] = useState(null);
  const [shapResp, setShapResp] = useState(null);
  const [driftResp, setDriftResp] = useState(null);
  const [monitoringLoading, setMonitoringLoading] = useState(true);
  const [monitoringError, setMonitoringError] = useState(null);
  // 드리프트 판정 수집 상태("pending" | "ok" | "error") — 요청이 끝났는지를 payload 진위와 분리한다
  // (빈 본문/`null` 200도 "ok"이지만 쓸 값은 없다). 문구는 성능·설명 API 상태와 함께
  // 아래 collectPhase에서 배너 1건으로 합친다(배너 2건은 예약 높이를 넘겨 아래 카드를 밀어냄).
  // 마운트 즉시 요청하므로 첫 렌더부터 "pending" — 한 프레임도 "정상 수신"을 주장하지 않는다.
  const [driftStatus, setDriftStatus] = useState("pending");
  // 실측 지표 계열에 대한 백엔드 이상치 판정(POST /monitoring/outliers) — 실계산 API를 화면에 연결한다.
  // { result, labels, values } 또는 null(실측 계열이 없거나 판정 실패).
  const [outlierResp, setOutlierResp] = useState(null);
  // 마지막 수집 시각 — 성능·설명 API 수집에 성공한 시점에만 갱신한다(수집 사실과 표시를 일치시킴)
  const [lastCollected, setLastCollected] = useState(null);
  const [retrying, setRetrying] = useState(false);
  const monitoringRequestRef = useRef(0);
  const driftRequestRef = useRef(0);
  const retryRef = useRef(null);
  const retryBusyRef = useRef(false);
  const driftBusyRef = useRef(false);
  const collectRef = useRef(null);

  const loadMonitoringData = useCallback(() => {
    const requestId = ++monitoringRequestRef.current;
    setMonitoringLoading(true);
    setMonitoringError(null);
    // model_id 를 함께 보내 백엔드가 실측(SQLite 실행 이력·학습 아티팩트)을 먼저 찾게 한다.
    // 실측이 없을 때의 시드 폴백은 데모 표시 ON 에서만 받는다 — OFF 에서는 시드 수치가
    // 네트워크 응답에도 남지 않게 서버가 빈 응답(`source: null`)을 내려준다.
    // include_seed 는 호출마다 적어 둔다(공유 변수로 감추면 새 호출에서 빠져도 드러나지 않는다).
    return Promise.all([
      apiGet("/api/v3/monitoring/metrics", { params: { model_id: modelTarget, include_seed: mockDataVisible } }),
      apiGet("/api/v3/monitoring/explain", { params: { model_id: modelTarget, include_seed: mockDataVisible } })
    ])
      .then(([m, s]) => {
        if (requestId !== monitoringRequestRef.current) return;
        setMetricsResp(m);
        setShapResp(s);
        setLastCollected(new Date());
      })
      .catch((err) => {
        if (requestId !== monitoringRequestRef.current) return;
        setMetricsResp(null);
        setShapResp(null);
        setMonitoringError(err?.message ?? "모니터링 서버에 연결할 수 없습니다.");
        addConsoleLog(`ERROR: 모니터링 지표 로드 실패 — ${err?.message ?? "알 수 없는 오류"}`);
      })
      .finally(() => {
        if (requestId === monitoringRequestRef.current) setMonitoringLoading(false);
      });
  }, [addConsoleLog, modelTarget, mockDataVisible]);

  // 백엔드 PSI/KL 판정 재조회(표시 전용) — driftInjected 변경 효과와 [다시 시도] 버튼이 같은 경로를 쓴다.
  // 재학습 발화는 injectDrift → 오케스트레이션 이벤트 경로가 단독 담당(중복 트리거 방지).
  const loadDriftData = useCallback(() => {
    const requestId = ++driftRequestRef.current;
    setDriftResp(null);
    setDriftStatus("pending");
    return apiGet("/api/v3/monitoring/drift", {
      // 시드 분포 판정도 데모 표시 ON 전용 — 실 추론 입력 수집 경로가 아직 없다.
      params: { drifted: driftInjected, model_id: DRIFT_MODEL_ID, include_seed: mockDataVisible }
    })
      .then((d) => {
        if (requestId !== driftRequestRef.current) return;
        setDriftResp(d);
        setDriftStatus("ok");
      })
      .catch((err) => {
        if (requestId !== driftRequestRef.current) return;
        setDriftResp(null);
        setDriftStatus("error");
        addConsoleLog(`ERROR: 드리프트 조회 실패 — ${err.message}`);
      });
  }, [driftInjected, addConsoleLog, mockDataVisible]);

  useEffect(() => {
    loadMonitoringData();
    return () => {
      monitoringRequestRef.current += 1;
    };
  }, [loadMonitoringData]);

  useEffect(() => {
    loadDriftData();
    return () => {
      driftRequestRef.current += 1;
    };
  }, [loadDriftData]);

  // 재시도 버튼은 재시도 중에도 마운트를 유지해야 키보드 초점이 유지된다.
  // disabled를 걸면 브라우저가 초점을 body로 떨어뜨리므로 aria-disabled + 핸들러 가드로 잠근다.
  // ref 가드: 같은 틱의 연타는 state 갱신 전이라 retryLocked로 막을 수 없다(중복 요청 방지).
  // 배너가 알리는 실패는 성능·설명·드리프트 어느 쪽이든이므로 세 요청을 함께 다시 보낸다.
  const retryLocked = monitoringLoading || retrying;
  const handleRetry = () => {
    if (retryBusyRef.current || retryLocked) return;
    retryBusyRef.current = true;
    const hadFocus = document.activeElement === retryRef.current;
    setRetrying(true);
    // 두 로더 모두 자체 catch로 흡수하므로 이 finally는 항상 실행된다(잠금이 남지 않는다).
    Promise.all([loadMonitoringData(), loadDriftData()]).finally(() => {
      retryBusyRef.current = false;
      setRetrying(false);
      if (!hadFocus) return;
      // 성공해서 배너가 사라지면 초점이 body로 떨어지므로 수집 상태 문구로 옮겨준다(A11Y-05).
      requestAnimationFrame(() => {
        if (document.activeElement === document.body) collectRef.current?.focus();
      });
    });
  };

  // 지표 추이 x축 — 접속 시각 기준 최근 10시간(정시 라벨)
  const hourlyLabels = useMemo(() => {
    const now = new Date();
    return Array.from({ length: 10 }, (_, i) => {
      const d = new Date(now.getTime() - (9 - i) * 3600000);
      return `${String(d.getHours()).padStart(2, "0")}:00`;
    });
  }, []);

  const targetModel = MODEL_REGISTRY.find((m) => m.id === modelTarget) ?? MODEL_REGISTRY[0];

  // 데모 표시 OFF의 데이터 계층 차단 — 응답이 실측(source="measured")일 때만 값으로 받아들인다.
  // 백엔드는 토글을 모르고 항상 같은 응답을 주므로, 시드 유래 응답을 걸러내는 곳은 여기 한 군데다.
  const usable = (resp) => (allowSeed || resp?.source === "measured" ? resp : null);
  const metricsUsable = usable(metricsResp);
  const shapUsable = usable(shapResp);
  const driftUsable = usable(driftResp);
  const metricsMeasured = metricsResp?.source === "measured";
  const measuredLabels = Array.isArray(metricsResp?.labels) ? metricsResp.labels : [];
  // 시드 계열은 데모 표시가 켜져 있을 때만 폴백으로 쓴다.
  const seedSeries = (key) => (allowSeed ? appData.metrics_history?.[key] : null);

  // 이상치 판정은 백엔드 실계산 API에 맡긴다 — 실측 Accuracy 계열이 2개 이상일 때만 의미가 있다.
  // 시드 계열은 보내지 않는다(시드에 대한 실계산은 실측이 아니다).
  const measuredAccuracy = metricsMeasured ? metricsResp?.history?.accuracy : null;
  useEffect(() => {
    const values = Array.isArray(measuredAccuracy) ? measuredAccuracy.filter(Number.isFinite) : [];
    if (values.length < 2) {
      setOutlierResp(null);
      return undefined;
    }
    let alive = true;
    apiSend("POST", "/api/v3/monitoring/outliers", { body: { values } })
      .then((result) => {
        if (alive) setOutlierResp({ result, labels: measuredLabels, values });
      })
      .catch((err) => {
        if (!alive) return;
        setOutlierResp(null);
        addConsoleLog(`ERROR: 이상치 판정 실패 — ${err?.message ?? "알 수 없는 오류"}`);
      });
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- measuredLabels는 같은 응답에서 온다
  }, [measuredAccuracy, addConsoleLog]);
  // 표시 버전은 백엔드 레지스트리의 현행 운영 버전을 따르고, 응답 전에만 Model Store·상수로 폴백한다
  // (승급 후 프런트 상수와 어긋나지 않게).
  const servingVersionOf = (modelId, fallback) =>
    modelCandidates?.[modelId]?.version ??
    modelStore?.find((m) => m.modelId === modelId && m.status === "운영")?.version ??
    fallback;

  // 대상 모델 선택지 — 데모 OFF에서는 백엔드가 내려준 모델(시드 지표 제외)만 남긴다.
  // 상수 레지스트리(MODEL_REGISTRY)는 시드 모델명·버전이라 옵션 텍스트로도 내보내지 않는다.
  const NO_DEMO_DATA = "데모 데이터 없음";
  const selectableModels = allowSeed
    ? MODEL_REGISTRY
    : MODEL_REGISTRY.filter((m) => modelCandidates?.[m.id]);

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
    // 백엔드 응답(reference/current/buckets) 우선. 시드 폴백은 데모 표시가 켜져 있을 때만 쓰고,
    // OFF에서 실측 분포가 없으면 빈 배열 → 축만 남은 빈 차트가 된다(요소는 감추지 않는다).
    const seed = allowSeed ? appData.drift_distribution : null;
    const buckets = driftUsable?.buckets ?? seed?.buckets ?? [];
    const reference = driftUsable?.reference ?? seed?.reference ?? [];
    const current =
      driftUsable?.current ??
      (driftInjected ? seed?.current_drifted : seed?.current_normal) ??
      [];
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
  }, [appData, allowSeed, driftInjected, driftUsable]);

  // 대상 모델 파생 지표 계열 — 요약 카드·게이지·6대 지표 추이가 모두 이 계열 하나에서 값을 읽는다.
  // 모델별 결정적 오프셋(accDelta/errRatio)을 새 배열로 만들어 원본(appData·API 응답)은 변경하지 않는다.
  const modelSeries = useMemo(() => {
    // 응답 계열이 없거나 비어 있으면 시드로 폴백하되, 데모 표시 OFF에서는 폴백하지 않고
    // 빈 배열로 둔다 — 없는 값을 시드로 메우면 시드가 실측치 자리를 차지한다.
    const source = (key) => {
      const fromResp = metricsUsable?.history?.[key];
      if (Array.isArray(fromResp) && fromResp.length > 0) return fromResp;
      const seed = seedSeries(key);
      return Array.isArray(seed) ? seed : [];
    };
    // 모델별 오프셋(accDelta·errRatio)은 시드 한 벌로 세 모델을 구분해 보이려는 **데모 장치**다.
    // 실측 계열(metricsUsable)에 적용하면 백엔드가 측정한 값이 시드 상수로 왜곡되므로 적용하지 않는다.
    const measured = Array.isArray(metricsUsable?.history?.accuracy) && metricsUsable.history.accuracy.length > 0;
    // 원시값을 그대로 검사한다 — Number(null)·Number("")·Number(false)는 0이라,
    // 강제 변환을 거치면 값이 없는 표본이 0.000짜리 실측치로 둔갑해 화면에 남는다.
    const tune = (key, isError) => {
      const values = source(key).filter((v) => Number.isFinite(v));
      if (measured) return values.map((v) => Number(v.toFixed(3)));
      return values.map((v) =>
        isError
          ? Number((v * targetModel.errRatio).toFixed(3))
          : Math.min(0.99, Math.max(0, Number((v + targetModel.accDelta).toFixed(3))))
      );
    };
    return Object.fromEntries(
      Object.entries(METRIC_SERIES).map(([key, isError]) => [key, tune(key, isError)])
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps -- seedSeries는 appData·allowSeed에서만 파생된다
  }, [appData, allowSeed, metricsUsable, targetModel]);

  // 화면에 표시되는 조회 구간 — 추이 차트·요약 카드·게이지·요약 문장이 모두 이 슬라이스를 본다.
  const windowSeries = useMemo(
    () =>
      Object.fromEntries(
        Object.entries(modelSeries).map(([key, values]) => [key, values.slice(-windowHours)])
      ),
    [modelSeries, windowHours]
  );

  // 추이 x축 — 실측이면 백엔드가 준 실행 ID 라벨을 쓴다. 실측 지표는 학습 실행 단위로 생기므로
  // 시각 라벨을 붙이면 없는 관측 주기를 있는 것처럼 보이게 한다.
  const axisLabels = useMemo(() => {
    const measured = metricsUsable?.labels;
    if (Array.isArray(measured) && measured.length > 0) return measured.slice(-windowHours);
    // 데모 OFF에는 그릴 계열이 없다 — 시각 라벨만 남기면 관측 주기가 있는 것처럼 보인다(빈 축).
    if (!allowSeed) return [];
    return hourlyLabels.slice(-windowHours);
  }, [metricsUsable, hourlyLabels, windowHours, allowSeed]);

  const metricsData = useMemo(() => {
    const series = (key) => windowSeries[key];

    return {
      labels: axisLabels,
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
          data: series("mse"),
          borderColor: "rgba(239, 68, 68, 1)",
          borderWidth: 1.5,
          borderDash: [5, 5],
          tension: 0.35
        },
        {
          label: "MAE",
          data: series("mae"),
          borderColor: "rgba(251, 146, 60, 1)",
          borderWidth: 1.5,
          borderDash: [2, 3],
          tension: 0.35
        }
      ]
    };
  }, [axisLabels, windowSeries]);

  const shapData = useMemo(() => {
    // features가 없거나 배열이 아니면 데모 시드로 폴백하되, 데모 표시 OFF에서는 폴백하지 않는다.
    const source = Array.isArray(shapUsable?.features)
      ? shapUsable.features
      : allowSeed
        ? appData.shap_features
        : [];
    // 값이 유효한 수치인 특징만 남겨 라벨·값·색을 같은 배열에서 만든다 — 짝이 어긋나지 않고,
    // Number(null)·Number("0.3") 같은 강제 변환으로 없는 값이 0.000짜리 실측치로 둔갑하지 않는다.
    // 0과 음수는 실제 기여도이므로 그대로 남긴다.
    const feats = source.filter((f) => Number.isFinite(f?.value));
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
  }, [appData, allowSeed, shapUsable]);

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

  // 차트 대체 텍스트(§9 A11Y-07)의 수치는 상단 KPI·게이지와 같은 소수 3자리로 표기한다.
  const fmt3 = (value) => Number(value).toFixed(3);
  // 분포 변화 요약문(driftSummary)과 그 파생 계산은 UI 피드백 #7로 화면에서 제거했다.
  const latestMetrics = metricsData.datasets
    .map((dataset) => {
      const first = dataset.data[0];
      const last = dataset.data.at(-1);
      if (first === undefined || last === undefined) return `${dataset.label} –`;
      const difference = Number(last) - Number(first);
      const direction = difference > 0 ? "상승" : difference < 0 ? "하락" : "변화 없음";
      return `${dataset.label} ${fmt3(first)}→${fmt3(last)} (${direction} ${difference > 0 ? "+" : ""}${fmt3(difference)})`;
    })
    .join(", ");
  const topShapIndex = shapData.datasets[0].data.reduce(
    (best, value, index, values) => (Math.abs(value) > Math.abs(values[best]) ? index : best),
    0
  );
  const topShapValue = shapData.datasets[0].data[topShapIndex];
  // 유효한 특징이 하나도 없으면 요약할 값이 없다(빈·전부 무효 features 응답) — 라벨·값 자리에
  // undefined를 넣지 않고, 강제 변환으로 없는 값을 0.000처럼 단정하지도 않는다.
  const shapSummary = Number.isFinite(topShapValue)
    ? `절대 기여도가 가장 큰 특징은 ${shapData.labels[topShapIndex]}이며 SHAP 값은 ${fmt3(topShapValue)}입니다. 양수는 인구 유출 완화 기여, 음수는 인구 유출 기여를 뜻합니다. 전체 값: ${shapData.labels
        .map((label, index) => `${label} ${fmt3(shapData.datasets[0].data[index])}`)
        .join(", ")}.`
    : "표시할 특징 기여도 데이터가 없어 요약할 수 없습니다.";

  const vizMs = useRenderTiming([driftData, metricsData, shapData]);

  // 현재 지표 = 승급 오버라이드(파이프라인 완료) 우선, 없으면 조회 구간의 가장 최근 값.
  // 값이 없으면 null을 돌려 대시("–", §10 UI-04)로 표기한다(0·NaN으로 단정하지 않음).
  const selectedMetricOverride = metricOverrides?.[modelTarget] ?? {};
  const metricValue = (key) => {
    const override = selectedMetricOverride[key];
    if (Number.isFinite(override)) return override;
    const values = windowSeries[key] ?? [];
    const latest = values.length > 0 ? values[values.length - 1] : null;
    return Number.isFinite(latest) ? latest : null;
  };
  const fmtMetric = (value) => (value === null ? NO_VALUE : fmt3(value));
  const accVal = fmtMetric(metricValue("accuracy"));
  const f1Val = fmtMetric(metricValue("f1"));

  // PSI·드리프트 판정은 백엔드 실계산값(driftResp) 사용.
  // psi 필드가 빠진 200 응답에도 화면이 깨지지 않도록 유한수일 때만 사용한다.
  const psiUsable = Number.isFinite(driftUsable?.psi);
  // 판정할 값 자체가 없으면(데모 OFF · 실 추론 입력 수집 경로 없음) 정상이라고 단정하지 않는다.
  const driftKnown = psiUsable || allowSeed;
  const drifted = driftKnown && (driftUsable?.drifted ?? driftInjected);
  const psiVal = psiUsable
    ? driftUsable.psi.toFixed(3)
    : allowSeed
      ? driftInjected
        ? SEED_PSI_DRIFTED
        : SEED_PSI_NORMAL
      : NO_VALUE;
  const psiColor = !driftKnown
    ? "var(--text-muted)"
    : drifted
      ? "var(--accent-red)"
      : "var(--accent-teal)";
  const driftLabel = !driftKnown ? "판정 없음" : drifted ? "위험 (Drift)" : "정상";
  // 정상(teal)은 .system-status 기본 스타일과 같으므로 위험 상태만 덮어쓴다.
  const driftLabelStyle = drifted
    ? { backgroundColor: "rgba(var(--accent-red-rgb), 0.02)", color: "var(--accent-red)" }
    : null;

  // 이상값 로그 시각 — 현재 시각 기준 상대 시각으로 산출(정적 mock 노출 방지)
  const ago = (minutes) => fmtHM(new Date(Date.now() - minutes * 60000));
  const seedOutlierRows = driftInjected
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

  // 실측 이상치 — 백엔드 실계산 API(POST /monitoring/outliers)가 돌려준 결과만 쓴다.
  const measuredOutlierRows = outlierResp?.result?.outliers?.map((o) => ({
    time: outlierResp.labels[o.index] ?? `#${o.index + 1}`,
    target: `${targetModel.name} 학습 실행 Accuracy ${fmt3(outlierResp.values[o.index])}`,
    z: Number.isFinite(o.z) ? Number(o.z).toFixed(2) : NO_VALUE,
    outlier: true
  }));
  const outlierRows = measuredOutlierRows ?? (allowSeed ? seedOutlierRows : []);
  const outlierMeasured = measuredOutlierRows != null;
  const outlierCount = outlierMeasured
    ? `${measuredOutlierRows.length}건`
    : allowSeed
      ? driftInjected
        ? "3건"
        : "0건"
      : NO_VALUE;
  const outlierAlerting = outlierMeasured ? measuredOutlierRows.length > 0 : allowSeed && driftInjected;
  const outlierCountColor = outlierAlerting ? "var(--accent-red)" : "var(--text-primary)";
  // 예측 지연 — 학습 시 실측한 1건 추론 지연(백엔드 training.latency_ms)을 우선한다.
  // 실측이 없고 데모 표시가 켜져 있을 때만 시드 상수로 내려가고, OFF에서는 "측정 없음"으로 둔다.
  // 드리프트는 인구이동 예측에만 주입되므로 다른 모델에 드리프트 지연을 붙이지 않는다.
  const measuredLatency = metricsMeasured ? metricsResp?.latency_ms : null;
  const latencyMs = Number.isFinite(measuredLatency)
    ? measuredLatency
    : allowSeed
      ? driftInjected && modelTarget === DRIFT_MODEL_ID
        ? SEED_LATENCY_DRIFTED_MS
        : SEED_LATENCY_NORMAL_MS
      : null;
  // 실측 지연은 1ms 미만이라 정수로 반올림하면 전부 0ms가 된다.
  const latencyText =
    latencyMs === null ? "측정 없음" : `${latencyMs >= 1 ? latencyMs.toFixed(0) : fmt3(latencyMs)}ms`;

  // 수집 상태 — 성능·설명 API 결과와 문구를 일치시킨다(실패 시 데이터 소스 정상 수신을 주장하지 않음).
  // 색상 외 아이콘 형태로도 상태를 구분한다(§9 A11Y-06 · §10 UI-01).
  // 드리프트 액션 — 파이프라인 진행/종료 상태를 따른다. driftInjected만으로 '진행 중'을 주장하면
  // 실패·롤백·반려·조정으로 종료된 뒤에도 버튼이 잠긴 채 남는다.
  const TERMINAL_RESULT_LABELS = {
    failed: "실행 실패",
    rolled_back: "자동 롤백",
    rejected: "승급 반려",
    debounced: "실행 조정"
  };
  // 파이프라인 상태는 전역 1개뿐이므로 대상 모델을 확인한다.
  // 다른 모델(생활인구·정주여건) 실행 결과가 이 화면의 드리프트 종료 배너·재시도를 가로채면 안 된다.
  const driftPipeline = RETRAIN_PIPELINES.find((p) => p.model === DRIFT_MODEL_ID) ?? RETRAIN_PIPELINES[0];
  const isDriftRun = pipelineRun?.model === DRIFT_MODEL_ID;
  // 예약(pipelineScheduled)은 injectDrift만 세우므로 항상 드리프트 대응이다(이때 pipelineRun은 아직 이전 실행일 수 있음).
  const driftInFlight = pipelineScheduled || (pipelineRunning && isDriftRun);
  const otherRunInFlight = pipelineRunning && !driftInFlight;
  const terminalState =
    !pipelineRunning && !pipelineScheduled && isDriftRun && pipelineResult && pipelineResult.state !== "succeeded"
      ? pipelineResult.state
      : null;
  const terminalLabel = terminalState
    ? TERMINAL_RESULT_LABELS[terminalState] ?? `미승급(${terminalState})`
    : null;
  const terminalReason = terminalState
    ? pipelineResult?.reason ?? pipelineResult?.deploy?.reason ?? pipelineResult?.evaluation?.reason ?? ""
    : "";
  // 승급 후에는 운영 버전이 등록 후보와 같아져 재학습이 시작되지 않는다(startPipeline 조기 반환).
  // 이 경우 실행되지 않을 버튼 대신 사유와 다음 행동(오케스트레이터의 후보 등록)을 안내한다(§10 UI-04).
  const driftServingVersion = servingVersionOf(DRIFT_MODEL_ID, driftPipeline.baseVersion);
  // 후보 버전은 백엔드 레지스트리(next_version)가 정한다. 프런트 상수(candidateVersion)는 승급
  // 한 번이면 운영 버전과 같아져 버튼이 영구 비활성으로 굳는다 — 응답 전에만 상수로 폴백한다.
  const driftCandidateVersion =
    modelCandidates?.[DRIFT_MODEL_ID]?.nextVersion ?? driftPipeline.candidateVersion;
  const candidateAvailable = driftServingVersion !== driftCandidateVersion;
  // 재학습 이력의 트리거는 실제 사유와 일치해야 한다. 드리프트 미감지 상태의 재시도까지
  // 드리프트/PSI로 기록하면 백엔드 트리거 분류(drift/manual)와 콘솔 로그가 모두 거짓이 된다.
  const retryLabel = driftInjected ? "드리프트 대응 재학습 재시도" : "수동 재학습 재시도";
  const retryRetrain = () =>
    startPipeline(
      driftInjected ? "드리프트 자동 감지 (PSI 임계 0.20 초과) 재시도" : "모니터 화면 수동 재시도",
      driftPipeline
    );
  const buildDriftAction = () => {
    // 드리프트 시뮬레이션은 시드 분포(PSI 0.384)·시드 파이프라인을 재현하는 **데모 장치**다.
    // 데모 표시 OFF에서 눌리면 PSI 카드는 "판정 없음"인데 배너·로그·벨 알림만 드리프트를
    // 단언하는 모순이 생기므로 잠근다. 실데이터 드리프트는 아래 실데이터 평가 패널이 담당한다.
    if (!allowSeed) {
      return {
        icon: "fa-vial-circle-check",
        label: "이상 시나리오 재현 (데모 전용)",
        locked: true,
        run: undefined,
        title: "드리프트 시뮬레이션은 시드 분포를 재현하는 데모 기능입니다. 설정에서 데모 데이터 표시를 켜면 실행할 수 있습니다."
      };
    }
    if (driftInFlight) {
      return {
        icon: "fa-hourglass-half",
        label: "드리프트 대응 진행 중...",
        locked: true,
        run: undefined,
        title: "재학습 파이프라인이 실행 중입니다. 오케스트레이터 탭에서 단계를 확인할 수 있습니다."
      };
    }
    // 다른 모델 실행 중에는 드리프트 시뮬레이션이 시작되지 않는다(injectDrift가 조기 반환) — 사유를 밝힌다.
    if (otherRunInFlight) {
      return {
        icon: "fa-hourglass-half",
        label: "다른 재학습 실행 중...",
        locked: true,
        run: undefined,
        title: `${pipelineRun?.pipelineName ?? "다른 모델"} 파이프라인이 실행 중입니다. 완료 후 드리프트 시뮬레이션을 실행할 수 있습니다.`
      };
    }
    if (!candidateAvailable) {
      return {
        icon: "fa-circle-info",
        label: "오케스트레이터에서 후보 확인",
        locked: false,
        run: () => navigateToTab("tab-mlops-orch"),
        title: `${driftPipeline.name}은 현재 운영 버전(${driftServingVersion})과 백엔드 후보 버전(${driftCandidateVersion})이 같아 실행할 수 없습니다. 오케스트레이터 탭으로 이동해 파이프라인·Model Store 상태를 확인합니다.`
      };
    }
    if (terminalState || driftInjected) {
      return {
        icon: "fa-rotate-right",
        label: retryLabel,
        locked: false,
        run: retryRetrain,
        title: driftInjected
          ? `${terminalLabel ? `직전 재학습 종료 상태: ${terminalLabel}. ` : ""}감지된 드리프트(PSI 임계 초과)에 대한 재학습을 다시 실행합니다.`
          : `직전 재학습 종료 상태: ${terminalLabel}. 드리프트가 감지되지 않은 상태이므로 수동 실행으로 기록됩니다.`
      };
    }
    return {
      icon: "fa-vial-circle-check",
      label: "이상 시나리오 재현 (드리프트 시뮬레이션)",
      locked: false,
      run: injectDrift,
      title: "드리프트 유입 상황을 재현하여 감지 → 알림 → 자동 재학습 파이프라인을 검증합니다"
    };
  };
  const driftAction = buildDriftAction();

  // 진행 중에도 마운트를 유지해야 키보드 초점이 버튼에 남는다(native disabled는 초점을 body로 떨어뜨림).
  // ref 가드: 같은 틱의 연타는 state 갱신 전이라 파이프라인 상태로 막을 수 없다(중복 오케스트레이션 요청 방지).
  const handleDriftAction = () => {
    if (driftAction.locked || driftBusyRef.current) return;
    driftBusyRef.current = true;
    Promise.resolve(driftAction.run()).finally(() => {
      driftBusyRef.current = false;
    });
  };

  // HTTP 200이라도 쓸 수 있는 값이 없으면 화면은 데모 시드로 폴백한다(modelSeries·shapData·psiVal).
  // 그 상태에서 '정상 수신'을 주장하지 않도록, 폴백이 실제로 일어난 소스만 골라낸다.
  // 판정 시작점은 payload 진위가 아니라 '요청이 끝났는가'다(lastCollected·driftStatus) —
  // 빈 본문/`null` 200은 apiGet이 null로 매핑하므로 payload만 보면 '퇴화 아님'으로 새어나간다.
  // 성능 계열은 6대 지표가 모두 있어야 통과시킨다(일부만 온 응답은 나머지가 데모 시드로 채워짐).
  // 실제 백엔드 계약(history 6계열·features 배열·필수 psi float)은 그대로 통과한다.
  // modelSeries가 계열 단위로 폴백하므로(위 source(key)) 판정도 계열 단위로 둔다.
  // 판정은 데이터 계층이 실제로 받아들인 응답(metricsUsable)을 본다 — 데모 표시 OFF에서 시드
  // 응답을 차단했는데 여기서는 "정상 수신"이라고 하면 화면의 빈 자리와 문구가 어긋난다.
  const seriesUsable = (key) => {
    const values = metricsUsable?.history?.[key];
    return Array.isArray(values) && values.some((v) => Number.isFinite(v));
  };
  const emptySeries = Object.keys(METRIC_SERIES).filter((key) => !seriesUsable(key));
  const allSeriesUsable = emptySeries.length === 0;
  const hasUsableFeatures = (resp) =>
    Array.isArray(resp?.features) && resp.features.some((f) => Number.isFinite(f?.value));
  const featuresUsable = hasUsableFeatures(shapUsable);
  const emptySources = [
    // 6계열 중 일부만 비어도 "성능 지표 전체가 데모"로 읽히지 않게 결손 계열을 밝힌다.
    lastCollected && !allSeriesUsable ? `성능 지표(${emptySeries.join("·")})` : null,
    lastCollected && !featuresUsable ? "특징 기여도" : null,
    driftStatus === "ok" && !psiUsable ? "드리프트 판정(PSI)" : null
  ].filter(Boolean);
  const failedSources = [
    monitoringError ? "실시간 성능·설명" : null,
    driftStatus === "error" ? "드리프트 판정" : null
  ].filter(Boolean);

  // 수집 판정 단일 출처 — 툴바 문구와 상태 배너가 같은 값을 읽어야 서로 모순되지 않는다
  // (드리프트만 실패해도 툴바가 '6개 정상 수신'을 주장하면 옆 role="alert"와 반대말이 된다).
  // 우선순위: 수집 실패 > 확인 중 > 값 없는 200(퇴화) > 정상.
  // 재조회 중에는 두 로더가 실패 플래그를 먼저 지우므로 '확인 중'이 직전 실패를 덮지 않는다.
  const collectPhase =
    failedSources.length > 0
      ? "failed"
      : monitoringLoading || driftStatus === "pending"
        ? "pending"
        : emptySources.length > 0
          ? "empty"
          : "healthy";

  // 헤더의 활성 문구("모델 모니터링 활성")는 이 판정을 모른다 — 같은 collectPhase를 전역에 공유해
  // 헤더 칩이 이 화면의 배너·툴바 문구와 어긋나지 않게 한다(§QA-2026-09-11 #5).
  useEffect(() => {
    setMonitorCollectStatus(
      collectPhase === "healthy" ? "ok" : collectPhase === "pending" ? "unknown" : "fail"
    );
  }, [collectPhase, setMonitorCollectStatus]);

  const collectedAt = lastCollected ? fmtTime(lastCollected) : NO_VALUE;
  // 데모 표시 OFF에서는 시드로 메우지 않으므로 "대체 값 표시 중"이라고 말하면 안 된다.
  const fallbackPhrase = allowSeed ? "대체 값 표시 중" : "표시할 값 없음";
  const collectStatus =
    collectPhase === "failed"
      ? {
          icon: "fa-triangle-exclamation",
          color: "var(--accent-red)",
          text: `수집 검증 실패 — ${fallbackPhrase} · 마지막 수집 성공: ${collectedAt}`
        }
      : collectPhase === "pending"
        ? {
            icon: "fa-satellite-dish",
            color: "var(--accent-blue)",
            text: "연계 데이터 소스 수집 상태 확인 중..."
          }
        : collectPhase === "empty"
          ? {
              // 어떤 소스가 비었는지는 아래 상태 배너가 밝힌다. 툴바 문구는 다른 분기와 길이를 맞춰
              // 한 줄을 유지한다(길어지면 툴바가 줄바꿈되며 아래 콘텐츠가 밀린다).
              icon: "fa-circle-exclamation",
              color: "var(--accent-orange)",
              text: `수집 응답에 값 없음 — ${fallbackPhrase} · 마지막 응답: ${collectedAt}`
            }
          : {
              icon: "fa-satellite-dish",
              color: "var(--accent-teal)",
              text: `마지막 수집: ${collectedAt} · 6개 연계 데이터 소스 정상 수신`
            };

  // 공용 상태 슬롯의 실시간 수집 배너 — 성능·설명 API와 드리프트 판정을 한 건으로 합친다.
  // 각각 배너를 띄우면 첫 진입(둘 다 pending)에 2건이 겹쳐 예약 높이(한 줄)를 넘긴다(§11 상태 전환).
  const liveStatus =
    collectPhase === "failed"
      ? {
          tone: "error",
          message: `${failedSources.join("·")} 수집에 실패했습니다. ${allowSeed ? "대체 값을 표시합니다." : "실데이터가 없어 해당 값은 비워 둡니다."}${monitoringError ? ` ${monitoringError}` : ""}`
        }
      : collectPhase === "pending"
        ? {
            tone: "pending",
            message: `실시간 성능·설명 API와 드리프트 판정을 확인하는 동안 ${allowSeed ? "대체 값을 표시합니다." : "값을 비워 둡니다."}`
          }
        : collectPhase === "empty"
          ? {
              tone: "warn",
              message: `${emptySources.join("·")}에 사용할 실측 값이 없습니다. ${allowSeed ? "대체 값을 표시합니다." : "해당 값은 비워 두며, 재학습을 실행하면 실측 지표가 쌓입니다."}`
            }
          : null;

  // API 수신과 폴백 경로를 기록한다. 표시 OFF는 전달 경로와 무관하게 모든 데이터에 적용한다.
  // 판정은 위 수집 검증(seriesUsable·hasUsableFeatures·psiUsable)과 같은 함수를 재사용해야
  // "정상 수신" 문구와 화면에 남는 영역이 어긋나지 않는다.
  // 판정 단위는 소비자가 실제로 읽는 값이다 — 6계열 공통 판정을 쓰면 한 계열만 비어도 실응답으로
  // 채워진 KPI 카드·게이지까지 함께 가려진다. metricValue가 조회 계열보다 먼저 쓰는 승급 오버라이드
  // (백엔드 응답 PipelineRun.candidate_metrics)도 응답 유래이므로 같이 본다.
  // 6대 지표 추이 차트만 전 계열을 요구한다 — 값 하나가 아니라 계열 전체를 그리기 때문이다.
  const srcOf = (fromApi) => (fromApi ? "api" : "mock");
  const metricFromApi = (key) => seriesUsable(key) || Number.isFinite(selectedMetricOverride[key]);
  const accF1Src = srcOf(metricFromApi("accuracy") && metricFromApi("f1"));
  const precisionSrc = srcOf(metricFromApi("precision"));
  const recallSrc = srcOf(metricFromApi("recall"));
  const metricsChartSrc = srcOf(allSeriesUsable);
  // 소비자가 실제로 읽는 값(shapUsable) 기준이어야 한다 — 원본 응답(shapResp)으로 판정하면
  // 시드 응답이 왔을 때 빈 카드에 "api" 표기가 붙는다.
  const shapSrc = srcOf(hasUsableFeatures(shapUsable));
  // 분포 차트는 buckets·reference·current를 한 응답에서 모두 받아야 실측이다(일부만 오면 mock 폴백이 섞인다).
  const driftChartSrc = srcOf(
    Array.isArray(driftUsable?.buckets) &&
      driftUsable.buckets.length > 0 &&
      Array.isArray(driftUsable?.reference) &&
      Array.isArray(driftUsable?.current)
  );
  const psiSrc = srcOf(psiUsable);

  return (
    <>
      {/* 운영 툴바 — 수집 상태 + 이상 시나리오 재현(드리프트 → 자동 재학습 검증) */}
      <div className="monitor-toolbar">
        <span className="monitor-collected" ref={collectRef} tabIndex={-1}>
          <i
            className={`fa-solid ${collectStatus.icon}`}
            style={{ color: collectStatus.color }}
            aria-hidden="true"
          ></i>{" "}
          {collectStatus.text}
        </span>
        <div className="monitor-options">
          <label className="compact-select-field">
            <span>대상 모델</span>
            <select className="select-control mock-data-output" value={modelTarget} onChange={(e) => setModelTarget(e.target.value)}>
              {/* 데모 OFF에서는 백엔드가 확인해 준 모델(실측 지표 보유)만 고를 수 있다. 상수 레지스트리
                  이름·버전은 시드라 옵션 텍스트로도 내보내지 않는다. */}
              {selectableModels.length === 0 ? (
                <option value={modelTarget}>{NO_DEMO_DATA} — 확인된 운영 모델 없음</option>
              ) : (
                selectableModels.map((m) => (
                  <option key={m.id} value={m.id}>{m.name} {servingVersionOf(m.id, m.version)}</option>
                ))
              )}
            </select>
          </label>
          <label className="compact-select-field">
            <span>조회 구간</span>
            <select className="select-control" value={windowHours} onChange={(e) => setWindowHours(Number(e.target.value))}>
              {WINDOW_OPTIONS.map((h) => (
                <option key={h} value={h}>최근 {h}시간</option>
              ))}
            </select>
          </label>
        </div>
        {/* 조작 버튼이라 데모 OFF에서도 눌러야 한다 — mock-data-output(가림 대상)에서 제외한다. */}
        <button
          type="button"
          className="btn btn-secondary"
          style={{ padding: "6px 14px", fontSize: 12 }}
          onClick={handleDriftAction}
          aria-disabled={driftAction.locked}
          aria-busy={driftInFlight}
          title={driftAction.title}
        >
          <i className={`fa-solid ${driftAction.icon}`} aria-hidden="true"></i> {driftAction.label}
        </button>
      </div>
      {/* 상태 안내 슬롯 — 배너가 붙고 떨어져도 아래 지표 카드가 위아래로 밀리지 않도록 높이를 예약한다(§11 상태 전환) */}
      <div className="monitor-state-slot">
        {terminalLabel && (
          <p
            className={`async-feedback is-${terminalState === "failed" || terminalState === "rolled_back" ? "error" : "pending"}`}
            role={terminalState === "failed" || terminalState === "rolled_back" ? "alert" : "status"}
            aria-live={terminalState === "failed" || terminalState === "rolled_back" ? "assertive" : "polite"}
          >
            재학습 파이프라인이 종료되었습니다 — 결과: {terminalLabel}
            {terminalReason ? ` · ${terminalReason}` : ""}.{" "}
            {driftInjected
              ? `드리프트 감지 상태는 유지됩니다. 상단 [${retryLabel}] 버튼에서 다시 실행할 수 있습니다.`
              : `드리프트가 감지되지 않은 실행이므로 상단 [${retryLabel}] 버튼은 수동 실행으로 기록됩니다.`}
          </p>
        )}
        {/* 후보 부재 안내문은 화면에서 제거했다(UI 피드백 #1). 같은 내용은 상단 실행 버튼의 title에
            남아 있고, 이 슬롯의 종료 안내·liveStatus가 role=status/alert 알림 자리를 계속 맡는다. */}
        {liveStatus && (
          <div
            className={`async-feedback is-${liveStatus.tone}`}
            role={liveStatus.tone === "error" ? "alert" : "status"}
            aria-live={liveStatus.tone === "error" ? "assertive" : "polite"}
          >
            <span>{liveStatus.message}</span>
            {/* retrying 중에도 버튼을 유지해야 키보드 초점이 버튼에 남는다(§9 A11Y-01) */}
            {(liveStatus.tone !== "pending" || retrying) && (
              <button
                ref={retryRef}
                type="button"
                className="btn btn-secondary"
                onClick={handleRetry}
                aria-disabled={retryLocked}
                aria-busy={retryLocked}
              >
                {retryLocked ? "다시 시도 중..." : "다시 시도"}
              </button>
            )}
          </div>
        )}
      </div>

      {/* 드리프트 이후의 다음 단계 — 재학습 진행/결과는 오케스트레이터 탭에서 확인한다.
          문구는 위 드리프트 판정(driftInFlight/otherRunInFlight)을 그대로 따른다. 다른 모델의 수동 실행까지
          '드리프트 대응 자동 재학습'으로 부르면 안 되므로 그때는 중립 문구를 쓴다.
          상태 안내 슬롯(.monitor-state-slot)은 목업 가림 대상이라 그 밖에 둔다. */}
      {(driftInjected || driftInFlight || otherRunInFlight) && (
        <NextStepBanner
          tone={driftInjected || driftInFlight ? "warn" : "info"}
          message={
            driftInFlight
              ? "드리프트 대응 자동 재학습이 진행 중입니다"
              : driftInjected
                ? "드리프트 감지 상태가 남아 있습니다 — 직전 재학습 결과를 확인하세요"
                : "파이프라인 실행 중 — 오케스트레이터에서 진행 확인"
          }
          actionLabel="오케스트레이터에서 보기"
          targetTab="tab-mlops-orch"
        />
      )}

      <div className="grid-cols-3">
        <div className="card" style={{ padding: "var(--space-xl)" }} data-values-source={accF1Src}>
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
        </div>

        <div
          className={"card" + (driftInjected ? " glow-red" : "")}
          style={{ padding: "var(--space-xl)" }}
          data-values-source={psiSrc}
        >
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
        </div>

        <div
          className="card"
          style={{ padding: "var(--space-xl)" }}
          data-values-source={srcOf(outlierMeasured)}
        >
          <div className="stat-label">
            Outlier Detection (Z-Score)
            <InfoTip
              text={
                outlierMeasured
                  ? `백엔드 이상치 판정 API가 실측 학습 실행 ${measuredLabels.length}건의 Accuracy 계열에서 Z-score 임계를 넘는 실행을 센 값입니다.`
                  : "Z-score(평균 대비 표준편차 거리)가 임계치를 넘는 이상치 건수입니다. 실측 지표 계열이 2건 이상 쌓이면 백엔드 판정 결과로 바뀝니다."
              }
            />
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
                // 정상(teal)은 .system-status 기본값과 같으므로 경고 상태만 덮어쓴다.
                ...(outlierAlerting
                  ? { backgroundColor: "rgba(var(--accent-red-rgb), 0.02)", color: "var(--accent-red)" }
                  : outlierCount === NO_VALUE
                    ? { color: "var(--text-muted)" }
                    : null)
              }}
            >
              {outlierCount === NO_VALUE ? "판정 없음" : outlierAlerting ? "경고" : "정상"}
            </span>
          </div>
        </div>
      </div>

      <div className="grid-details-split">
        <Card
          title="데이터 분포 변화 시각화 (참조 vs 최근유입)"
          icon="fa-chart-area"
          headerRight={<PerfBadge ms={vizMs} />}
          data-values-source={driftChartSrc}
        >
          <div style={{ position: "relative", height: 320, width: "100%" }}>
            <Bar data={driftData} options={AXIS_OPTS} />
            {driftData.labels.length === 0 && (
              <ChartEmptyNote>
                {EMPTY_HINT} — 실 추론 입력 분포 수집 경로가 아직 없어 분포 비교를 표시할 수 없습니다.
                아래 실데이터 연계 패널의 드리프트 판정을 사용하세요.
              </ChartEmptyNote>
            )}
          </div>
        </Card>

        <Card title="이상값 검출 로그" icon="fa-filter">
          <div style={{ maxHeight: 320, overflowY: "auto" }}>
            <table style={{ width: "100%" }}>
              <caption className="sr-only">최근 이상값 검출 시간, 지자체, Z-score와 판정 상태</caption>
              <thead>
                <tr>
                  <th scope="col">시간</th>
                  <th scope="col">지자체</th>
                  <th scope="col" className="cell-num">Z-score</th>
                  <th scope="col">상태</th>
                </tr>
              </thead>
              <tbody>
                {outlierRows.length === 0 && (
                  <tr>
                    <td colSpan={4} className="monitor-empty-cell">
                      {outlierMeasured
                        ? "실측 지표 계열에서 이상치가 검출되지 않았습니다."
                        : `${EMPTY_HINT} — 재학습을 실행하면 실측 지표 계열에 대한 이상치 판정이 표시됩니다.`}
                    </td>
                  </tr>
                )}
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
                    <td className="cell-num">{row.z}</td>
                    <td>
                      {row.outlier ? (
                        <span className="outlier-tag">Outlier</span>
                      ) : (
                        <span
                          className="system-status"
                          style={{ padding: "1px 6px", fontSize: 10 }}
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
        {/* 차트 요약문은 카드 제목의 InfoTip으로 옮겼다(UI 피드백 #9·#10). 값은 현재 로드된
            데이터(실데이터 또는 데모 시드)에서 읽고, 값이 없으면 '요약할 수 없습니다'로 표기한다.
            데모 토글은 데이터 유무만 바꾸므로 트리거 자체는 토글과 무관하게 항상 렌더한다. */}
        <Card
          title={
            <>
              MLOps AI 평가지표 실시간 모니터링
              <InfoTip
                text={
                  latestMetrics
                    ? `조회 구간 첫 값에서 최근 값까지의 변화: ${latestMetrics}.`
                    : "표시할 지표 데이터가 없어 구간 변화를 요약할 수 없습니다."
                }
                label="지표 변화 요약 보기"
              />
            </>
          }
          icon="fa-chart-column"
          data-values-source={metricsChartSrc}
        >
          <div style={{ position: "relative", height: 280, width: "100%" }}>
            <Line data={metricsData} options={AXIS_OPTS} />
            {!allSeriesUsable && (
              <ChartEmptyNote>
                {EMPTY_HINT} — 재학습을 실행하면 학습 실행별 실측 6대 지표가 이 자리에 쌓입니다.
              </ChartEmptyNote>
            )}
          </div>
        </Card>

        <Card
          title={
            <>
              SHAP 기반 인구 유출 기여 특징 중요도 분석
              <InfoTip
                text={
                  shapUsable?.basis
                    ? `산출 근거: ${shapUsable.basis}${shapUsable.version ? ` · 버전 ${shapUsable.version}` : ""}. ${shapSummary}`
                    : shapSummary
                }
                label="특징 기여도 요약 보기"
              />
            </>
          }
          icon="fa-brain"
          data-values-source={shapSrc}
        >
          <div style={{ position: "relative", height: 280, width: "100%" }}>
            <Bar data={shapData} options={shapOpts} />
            {!featuresUsable && (
              <ChartEmptyNote>
                {EMPTY_HINT} — 승급된 학습 아티팩트가 있어야 학습된 모델의 특징 기여도를 읽을 수 있습니다.
              </ChartEmptyNote>
            )}
          </div>
        </Card>
      </div>

      {/* 상단 stat 카드(Accuracy·F1)·6대 지표 차트와 중복되지 않는 지표만 게이지로 표시 */}
      <Card
        title={
          <>
            모델 신뢰도 게이지 (실시간)
            {/* 카드마다 제목 옆 트리거 1개만 둔다 — 기존 설명 툴팁에 표시 값 기준(피드백 #11)을 덧붙인다.
                데모 토글은 값의 유무만 바꾸므로 툴팁 자체는 토글과 무관하게 항상 같은 내용을 낸다. */}
            <InfoTip text="운영 모델의 Precision·Recall과 예측 지연(latency)을 표시합니다. 예측 지연이 자동 롤백 임계 200ms를 초과하면 직전 버전으로 자동 롤백됩니다. 표시 값: 인구이동 예측 모델 기준(예측 지연)." />
          </>
        }
        icon="fa-gauge-high"
      >
        <div className="grid-cols-3" style={{ marginBottom: 0 }}>
          {/* 값이 없으면 null 을 그대로 넘긴다 — 0으로 바꿔 넘기면 게이지가 "위험"/"정상"을 단정한다. */}
          <GaugeChart value={metricValue("precision")} label="Precision" data-values-source={precisionSrc} />
          <GaugeChart value={metricValue("recall")} label="Recall" data-values-source={recallSrc} />
          {/* 실측 지연(학습 시 측정한 1건 추론 지연)이 있으면 api, 없으면 시드 상수 또는 '측정 없음'. */}
          <GaugeChart
            value={latencyMs === null ? null : latencyMs / LATENCY_ROLLBACK_MS}
            displayText={latencyText}
            label="예측 지연"
            goodThreshold={0.75}
            lowerIsBetter
            data-values-source={srcOf(Number.isFinite(measuredLatency))}
          />
        </div>
      </Card>

      {/* 실데이터 연계(R4) — 남원 모델 검증/운영 평가·선형 SHAP·드리프트. 위 데모 지표와는 별개 경로. */}
      <RealdataEvaluationPanel />
    </>
  );
}
