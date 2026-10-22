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
import { dataModeEpoch } from "../lib/dataMode.js";
import { useChartTheme } from "../hooks/useChartTheme.js";
import { useRenderTiming } from "../lib/perf.js";
import { MODEL_REGISTRY } from "../constants/models.js";
import InfoTip from "../components/InfoTip.jsx";
import { getRealdataToken } from "../components/realdata/realdataClient.jsx";
import { apiGet, apiSend } from "../lib/api.js";
import { alertText } from "../lib/alertText.js";
import { monitorInstances, monitorActiveVersion, legacyMonitorId, regressionMetricText } from "../lib/monitorModels.js";
import { TARGET_LABELS, typeName, displayId } from "../lib/modelops.js";
import { baselineName, targetUnit } from "../lib/realSimulation.js";

// 데모 분포 표시 대상 — 인구이동 예측 모델(백엔드 시드 id).
// 등록 모델과 매핑된 이력만 조회하며, 설정의 데모 예시 판정은 별도 동작이다.
const DRIFT_MODEL_ID = "population-forecast";

// 데모 시드 PSI — 위와 같은 조건에서만 쓴다.
const SEED_PSI_DRIFTED = "0.384";
const SEED_PSI_NORMAL = "0.045";

// 값이 없을 때 쓰는 표기 — 요소를 감추지 않고 자리만 비운다(§10 UI-04).
const NO_VALUE = "–";
// 값이 비었을 때의 안내 — 상태가 아니라 값을 채우는 방법을 한 문장으로 알린다.
const EMPTY_HINT = "DataOps에서 학습데이터를 등록하고 오케스트레이터에서 모델을 학습·활성화하면 표시됩니다.";

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

// 실데이터(회귀) 계열 — 데모 표시 OFF에서 6대 분류 지표 자리에 이 계열이 들어간다.
// 백엔드 realdata_bridge가 후보 버전별 검증 MAE·RMSE·WAPE와 기준선 MAE를 같은 스키마로 내려준다.
const REALDATA_SERIES = {
  mae: true,
  rmse: true,
  wape: true,
  baseline_mae: true
};

// 추이 차트 데이터셋 정의 — 계열 키·라벨·선 스타일을 한 곳에 둔다(분류/회귀 두 벌).
const CLASSIFICATION_DATASETS = [
  { key: "accuracy", label: "Accuracy", borderColor: "rgba(16, 185, 129, 1)", backgroundColor: "rgba(16, 185, 129, 0.05)", fill: true, tension: 0.35, borderWidth: 2 },
  { key: "f1", label: "F1-Score", borderColor: "rgba(59, 130, 246, 1)", borderWidth: 2, pointStyle: "circle", tension: 0.35 },
  { key: "precision", label: "Precision", borderColor: "rgba(34, 211, 238, 1)", borderWidth: 1.5, pointStyle: "triangle", tension: 0.35 },
  { key: "recall", label: "Recall", borderColor: "rgba(168, 85, 247, 1)", borderWidth: 1.5, pointStyle: "rect", tension: 0.35 },
  { key: "mse", label: "MSE", borderColor: "rgba(239, 68, 68, 1)", borderWidth: 1.5, borderDash: [5, 5], tension: 0.35 },
  { key: "mae", label: "MAE", borderColor: "rgba(251, 146, 60, 1)", borderWidth: 1.5, borderDash: [2, 3], tension: 0.35 }
];

// WAPE(비율)는 MAE·RMSE(원 단위)와 자릿수가 달라 같은 축에 두면 선이 바닥에 깔린다 — 오른쪽 축으로 분리.
const REALDATA_DATASETS = [
  { key: "mae", label: "검증 MAE", borderColor: "rgba(239, 68, 68, 1)", backgroundColor: "rgba(239, 68, 68, 0.05)", fill: true, tension: 0.35, borderWidth: 2 },
  { key: "rmse", label: "검증 RMSE", borderColor: "rgba(59, 130, 246, 1)", borderWidth: 1.5, pointStyle: "circle", tension: 0.35 },
  { key: "baseline_mae", label: "기준선 MAE", borderColor: "rgba(148, 163, 184, 1)", borderWidth: 1.5, borderDash: [5, 5], tension: 0.35 },
  { key: "wape", label: "검증 WAPE (우축)", borderColor: "rgba(168, 85, 247, 1)", borderWidth: 1.5, pointStyle: "rect", tension: 0.35, yAxisID: "y1" }
];

// 모니터링 옵션 — 조회 구간 (대상 모델은 운영 모델 레지스트리에서 선택).
// 단위는 관측 개수다. 실측 지표는 학습 실행 단위로 생기므로 시간 라벨을 붙이지 않는다.
const WINDOW_OPTIONS = [3, 6, 10];

export default function MonitorPage() {
  const {
    appData,
    driftInjected,
    metricOverrides,
    addConsoleLog,
    setMonitorCollectStatus,
    mockDataVisible
  } = useAppState();
  const ct = useChartTheme();

  // 데모 표시 토글은 화면 구조가 아니라 **데이터 유무**만 바꾼다. OFF에서는 시드(mock_data.json)
  // 유래 값이 실값 자리에 들어가지 않도록 여기 데이터 계층에서 차단하고, 빈 자리는 "–"로 남긴다.
  const mode = mockDataVisible ? "mock" : "real";
  const [catalog, setCatalog] = useState(null);
  const [modelPick, setModelPick] = useState(null);
  const [realdataModels, setRealdataModels] = useState([]);
  const selectableModels = monitorInstances(catalog, mode);
  const selectedInstance = selectableModels.find((row) => modelPick?.mode === mode && row.id === modelPick.id)
    ?? selectableModels.find((row) => monitorActiveVersion(catalog, row, mode)) ?? selectableModels[0] ?? null;
  const selectedVersion = monitorActiveVersion(catalog, selectedInstance, mode);
  const modelTarget = legacyMonitorId(selectedInstance, realdataModels, mode) ?? "";
  // 매핑 없는 공통 모델에 다른 데모 모델의 이력을 폴백하지 않는다.
  const allowSeed = mockDataVisible && Boolean(modelTarget);

  // 모니터링 옵션 — 운영 모델 레지스트리에서 대상 모델 선택 + 조회 구간. 지표 추이 차트에 반영된다.
  // 대상 모델은 실측 지표 조회 키이기도 하므로 로더보다 먼저 선언한다.
  const [windowHours, setWindowHours] = useState(10);

  // 실데이터(rd_*) 접근 토큰 — 백엔드는 `data:read` 토큰이 있는 호출에만 실데이터를 싣는다.
  // undefined = 아직 확인 전(요청을 보내지 않는다), null = 발급 실패, 문자열 = 발급 완료.
  const [realdataToken, setRealdataToken] = useState(allowSeed ? null : undefined);
  // 데모 OFF의 대상 모델 목록 — 백엔드가 확인해 준 실데이터 모델(이름·표시 버전)만 쓴다.

  useEffect(() => {
    let alive = true;
    setRealdataToken(undefined);
    getRealdataToken()
      .then((token) => alive && setRealdataToken(token))
      .catch((err) => {
        if (!alive) return;
        setRealdataToken(null);
        addConsoleLog(`WARN: 실데이터 토큰 발급 실패 — ${alertText(err, "인증 토큰을 받지 못했습니다. 잠시 뒤 다시 시도하세요.")}`);
      });
    return () => {
      alive = false;
    };
  }, [mockDataVisible, addConsoleLog]);

  useEffect(() => {
    setCatalog(null); setRealdataModels([]);
    if (!realdataToken) return undefined;
    let alive = true;
    const epoch = dataModeEpoch();
    const current = () => alive && epoch === dataModeEpoch();
    Promise.all([
      apiGet("/api/v3/modelops/catalog", { token: realdataToken }),
      mockDataVisible ? Promise.resolve([]) : apiGet("/api/v3/orchestration/models", { token: realdataToken, params: { include_seed: false } })
    ]).then(([data, rows]) => {
      if (!current()) return;
      if (data.mode !== mode || data.source !== mode) throw new Error("모델 목록의 데이터 모드를 확인하세요.");
      setCatalog(data);
      setRealdataModels(Array.isArray(rows) ? rows.filter((row) => row.metrics_source === "realdata") : []);
    }).catch((err) => {
      if (current()) addConsoleLog(`WARN: 모델 목록 조회 실패 — ${alertText(err, "모델 목록을 불러오지 못했습니다.")}`);
    });
    return () => { alive = false; };
  }, [mockDataVisible, realdataToken, addConsoleLog]);

  // 백엔드(/api/v3/monitoring) 실데이터 — 없으면 mock으로 폴백해 로딩 중에도 렌더.
  const [metricsResult, setMetricsResp] = useState(null);
  const metricsResp = metricsResult?.model === modelTarget && metricsResult?.epoch === dataModeEpoch() ? metricsResult.data : null;
  const [shapResult, setShapResp] = useState(null);
  const shapResp = shapResult?.model === modelTarget && shapResult?.epoch === dataModeEpoch() ? shapResult.data : null;
  const [driftResult, setDriftResp] = useState(null);
  const driftResp = driftResult?.model === modelTarget && driftResult?.epoch === dataModeEpoch() ? driftResult.data : null;
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
  const collectRef = useRef(null);

  const loadMonitoringData = useCallback(() => {
    const requestId = ++monitoringRequestRef.current;
    const epoch = dataModeEpoch();
    const isCurrent = () => requestId === monitoringRequestRef.current && epoch === dataModeEpoch();
    setMonitoringLoading(true);
    setMonitoringError(null);
    setMetricsResp(null); setShapResp(null);
    if (!modelTarget) { setMonitoringLoading(false); setLastCollected(null); return Promise.resolve(); }
    // model_id 를 함께 보내 백엔드가 실측(SQLite 실행 이력·학습 아티팩트)을 먼저 찾게 한다.
    // 실측이 없을 때의 시드 폴백은 데모 표시 ON 에서만 받는다 — OFF 에서는 시드 수치가
    // 네트워크 응답에도 남지 않게 서버가 빈 응답(`source: null`)을 내려준다.
    // include_seed 는 호출마다 적어 둔다(공유 변수로 감추면 새 호출에서 빠져도 드러나지 않는다).
    return Promise.all([
      apiGet("/api/v3/monitoring/metrics", {
        token: realdataToken ?? undefined,
        params: { model_id: modelTarget, include_seed: mockDataVisible, window: windowHours }
      }),
      apiGet("/api/v3/monitoring/explain", {
        token: realdataToken ?? undefined,
        params: { model_id: modelTarget, include_seed: mockDataVisible }
      })
    ])
      .then(([m, s]) => {
        if (!isCurrent()) return;
        setMetricsResp({ model: modelTarget, epoch, data: m });
        setShapResp({ model: modelTarget, epoch, data: s });
        setLastCollected(new Date());
      })
      .catch((err) => {
        if (!isCurrent()) return;
        setMetricsResp(null);
        setShapResp(null);
        setMonitoringError(alertText(err, "모니터링 서버에 연결할 수 없습니다."));
        addConsoleLog(`ERROR: 모니터링 지표 로드 실패 — ${alertText(err, "모니터링 서버에 연결할 수 없습니다.")}`);
      })
      .finally(() => {
        if (isCurrent()) setMonitoringLoading(false);
      });
  }, [addConsoleLog, modelTarget, mockDataVisible, realdataToken, windowHours]);

  // 백엔드 PSI/KL 판정 재조회(표시 전용) — driftInjected 변경 효과와 [다시 시도] 버튼이 같은 경로를 쓴다.
  // 데모 판정은 읽기 전용이다. 등록 파이프라인 실행은 오케스트레이터에서 별도로 선택한다.
  const loadDriftData = useCallback(() => {
    const requestId = ++driftRequestRef.current;
    const epoch = dataModeEpoch();
    const isCurrent = () => requestId === driftRequestRef.current && epoch === dataModeEpoch();
    setDriftResp(null);
    setDriftStatus("pending");
    if (!modelTarget) { setDriftStatus("ok"); return Promise.resolve(); }
    return apiGet("/api/v3/monitoring/drift", {
      // 시드 분포 판정은 데모 표시 ON 전용 — 실 추론 입력 수집 경로가 아직 없다.
      // 데모 OFF에서는 선택한 실데이터 모델의 학습 구간 vs 검증/이후 구간 PSI를 받는다.
      token: realdataToken ?? undefined,
      params: {
        drifted: driftInjected,
        model_id: mockDataVisible ? DRIFT_MODEL_ID : modelTarget,
        include_seed: mockDataVisible
      }
    })
      .then((d) => {
        if (!isCurrent()) return;
        setDriftResp({ model: modelTarget, epoch, data: d });
        setDriftStatus("ok");
      })
      .catch((err) => {
        if (!isCurrent()) return;
        setDriftResp(null);
        setDriftStatus("error");
        addConsoleLog(`ERROR: 드리프트 조회 실패 — ${alertText(err, "드리프트 판정을 불러오지 못했습니다. 잠시 뒤 다시 시도하세요.")}`);
      });
  }, [driftInjected, addConsoleLog, mockDataVisible, modelTarget, realdataToken]);

  // 데모 OFF에서 토큰 확인 전에 보내면 서버가 (인증 없는) 빈 응답을 주고 곧바로 다시 그린다.
  const monitoringReady = allowSeed || realdataToken !== undefined;

  useEffect(() => {
    if (!monitoringReady) return undefined;
    loadMonitoringData();
    return () => {
      monitoringRequestRef.current += 1;
    };
  }, [loadMonitoringData, monitoringReady]);

  useEffect(() => {
    if (!monitoringReady) return undefined;
    loadDriftData();
    return () => {
      driftRequestRef.current += 1;
    };
  }, [loadDriftData, monitoringReady]);

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

  const targetModel =
    MODEL_REGISTRY.find((m) => m.id === modelTarget) ??
    // 실데이터 모델은 시드 오프셋(accDelta/errRatio)이 없다 — 이름만 쓰고 보정은 적용하지 않는다.
    { id: modelTarget, name: realdataModels.find((m) => m.model_id === modelTarget)?.name ?? modelTarget, accDelta: 0, errRatio: 1 };

  // 데모 표시 OFF의 데이터 계층 차단 — 응답이 실측(source="measured")일 때만 값으로 받아들인다.
  // 백엔드는 토글을 모르고 항상 같은 응답을 주므로, 시드 유래 응답을 걸러내는 곳은 여기 한 군데다.
  const usable = (resp) =>
    allowSeed || resp?.source === "measured" || resp?.source === "realdata" ? resp : null;
  // 실데이터 응답 여부 — 계열 정의·카드 라벨·출처 표기가 모두 이 판정을 따른다.
  const isRealdata = metricsResp?.source === "realdata";
  const activeSeries = isRealdata ? REALDATA_SERIES : METRIC_SERIES;
  const chartSeriesDefs = isRealdata ? REALDATA_DATASETS : CLASSIFICATION_DATASETS;
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
        addConsoleLog(`ERROR: 이상치 판정 실패 — ${alertText(err, "이상치 판정을 불러오지 못했습니다. 잠시 뒤 다시 시도하세요.")}`);
      });
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- measuredLabels는 같은 응답에서 온다
  }, [measuredAccuracy, addConsoleLog]);
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
    // 실데이터·실측 계열에는 데모 오프셋을 적용하지 않는다(측정값이 시드 상수로 왜곡된다).
    const measured =
      isRealdata || (Array.isArray(metricsUsable?.history?.accuracy) && metricsUsable.history.accuracy.length > 0);
    // 원시값을 그대로 검사한다 — Number(null)·Number("")·Number(false)는 0이라,
    // 강제 변환을 거치면 값이 없는 표본이 0.000짜리 실측치로 둔갑해 화면에 남는다.
    const tune = (key, isError) => {
      const values = source(key).filter((v) => Number.isFinite(v));
      // 실데이터 회귀 지표는 원 단위(수만~수억)라 소수 3자리로 깎으면 안 된다 — 원값 그대로 둔다.
      if (isRealdata) return values;
      if (measured) return values.map((v) => Number(v.toFixed(3)));
      return values.map((v) =>
        isError
          ? Number((v * targetModel.errRatio).toFixed(3))
          : Math.min(0.99, Math.max(0, Number((v + targetModel.accDelta).toFixed(3))))
      );
    };
    return Object.fromEntries(
      Object.entries(activeSeries).map(([key, isError]) => [key, tune(key, isError)])
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps -- seedSeries는 appData·allowSeed에서만 파생된다
  }, [appData, allowSeed, metricsUsable, targetModel, activeSeries, isRealdata]);

  // 화면에 표시되는 조회 구간 — 추이 차트·요약 카드·게이지·요약 문장이 모두 이 슬라이스를 본다.
  const windowSeries = useMemo(
    () =>
      Object.fromEntries(
        Object.entries(modelSeries).map(([key, values]) => [key, values.slice(-windowHours)])
      ),
    [modelSeries, windowHours]
  );
  // 실제로 그려진 관측 수. 구간보다 관측이 적으면 화면이 그대로라, 그 사실을 카드에 적는다.
  const plottedCount = Math.max(0, ...Object.values(windowSeries).map((values) => values.length));

  // 추이 x축 — 실측이면 백엔드가 준 실행 ID 라벨을 쓴다. 실측 지표는 학습 실행 단위로 생기므로
  // 시각 라벨을 붙이면 없는 관측 주기를 있는 것처럼 보이게 한다.
  const axisLabels = useMemo(() => {
    const measured = metricsUsable?.labels;
    if (Array.isArray(measured) && measured.length > 0) return measured.slice(-windowHours);
    // 데모 OFF에는 그릴 계열이 없다 — 시각 라벨만 남기면 관측 주기가 있는 것처럼 보인다(빈 축).
    if (!allowSeed) return [];
    return hourlyLabels.slice(-windowHours);
  }, [metricsUsable, hourlyLabels, windowHours, allowSeed]);

  const metricsData = useMemo(
    () => ({
      labels: axisLabels,
      datasets: chartSeriesDefs.map(({ key, ...style }) => ({ ...style, data: windowSeries[key] ?? [] }))
    }),
    [axisLabels, windowSeries, chartSeriesDefs]
  );

  // 실데이터 추이는 값의 자릿수가 달라 우축(WAPE)을 더한다. 분류 지표는 기존 축 그대로.
  const metricsChartOpts = useMemo(() => {
    if (!isRealdata) return AXIS_OPTS;
    return {
      ...AXIS_OPTS,
      scales: {
        ...AXIS_OPTS.scales,
        y1: {
          position: "right",
          grid: { display: false },
          ticks: { color: ct.tick }
        }
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- AXIS_OPTS는 ct에서만 파생된다
  }, [isRealdata, ct]);

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
      if (!Number.isFinite(first) || !Number.isFinite(last)) return null;
      const difference = Number(last) - Number(first);
      const direction = difference > 0 ? "상승" : difference < 0 ? "하락" : "변화 없음";
      return `${dataset.label} ${fmt3(first)}→${fmt3(last)} (${direction} ${difference > 0 ? "+" : ""}${fmt3(difference)})`;
    })
    .filter(Boolean)
    .join(", ");
  const topShapIndex = shapData.datasets[0].data.reduce(
    (best, value, index, values) => (Math.abs(value) > Math.abs(values[best]) ? index : best),
    0
  );
  const topShapValue = shapData.datasets[0].data[topShapIndex];
  // 유효한 특징이 하나도 없으면 요약할 값이 없다(빈·전부 무효 features 응답) — 라벨·값 자리에
  // undefined를 넣지 않고, 강제 변환으로 없는 값을 0.000처럼 단정하지도 않는다.
  // 실데이터(선형 SHAP)는 타깃이 방문객·소비액이라 부호 의미가 다르다 — 유출/완화로 읽지 않는다.
  const shapSignNote = isRealdata
    ? "양수는 예측값을 기준값보다 끌어올린 기여, 음수는 끌어내린 기여를 뜻합니다."
    : "양수는 인구 유출 완화 기여, 음수는 인구 유출 기여를 뜻합니다.";
  const shapSummary = Number.isFinite(topShapValue)
    ? `절대 기여도가 가장 큰 특징은 ${shapData.labels[topShapIndex]}이며 SHAP 값은 ${fmt3(topShapValue)}입니다. ${shapSignNote} 전체 값: ${shapData.labels
        .map((label, index) => `${label} ${fmt3(shapData.datasets[0].data[index])}`)
        .join(", ")}.`
    : EMPTY_HINT;

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
  // 실데이터 회귀 지표는 원 단위(수만~수억)라 소수 3자리 대신 천단위 구분으로 읽는다.
  const fmtAmount = (value) =>
    value === null ? NO_VALUE : Number(value).toLocaleString(undefined, { maximumFractionDigits: 1 });
  const accVal = isRealdata ? fmtAmount(metricValue("mae")) : fmtMetric(metricValue("accuracy"));
  const f1Val = isRealdata ? fmtAmount(metricValue("baseline_mae")) : fmtMetric(metricValue("f1"));
  // 카드 수·배치는 그대로 두고 라벨·설명만 회귀 지표에 맞춘다. 기준선 방법은 한글 이름, 모르면 "기준선"만.
  const baselineLabel = baselineName(metricsUsable?.baseline?.name);
  const primaryCard = isRealdata
    ? {
        label: "Model MAE / 기준선 MAE",
        tip: `선택한 실데이터 모델의 검증 MAE와 같은 구간 ${baselineLabel ? `기준선(${baselineLabel})` : "기준선"} MAE입니다. MAE가 기준선보다 낮아야 학습 모델이 이긴 것입니다.`,
        trailing: `기준선: ${f1Val}`
      }
    : {
        label: "Model Accuracy / F1-Score",
        tip: "운영 중인 모델의 정확도(Accuracy)와 정밀도·재현율의 조화평균(F1-Score). 재학습 승급 시 두 값이 함께 갱신됩니다.",
        trailing: `F1: ${f1Val}`
      };

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
    : isRealdata
      ? // 실데이터 계열에는 Z-score 판정 입력(학습 실행 Accuracy 계열)이 없다. "값 없음"이 아니라
        // 실측 계열에서 검출된 이상값이 0건이라는 뜻이므로 0건으로 적고 근거를 배지로 밝힌다.
        "0건"
      : allowSeed
        ? driftInjected
          ? "3건"
          : "0건"
        : NO_VALUE;
  const outlierAlerting = outlierMeasured ? measuredOutlierRows.length > 0 : allowSeed && driftInjected;
  const outlierCountColor = outlierAlerting ? "var(--accent-red)" : "var(--text-primary)";
  // 수집 상태 — 성능·설명 API 결과와 문구를 일치시킨다(실패 시 데이터 소스 정상 수신을 주장하지 않음).
  // 색상 외 아이콘 형태로도 상태를 구분한다(§9 A11Y-06 · §10 UI-01).
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
  const emptySeries = Object.keys(activeSeries).filter((key) => !seriesUsable(key));
  const allSeriesUsable = emptySeries.length === 0;
  const hasUsableFeatures = (resp) =>
    Array.isArray(resp?.features) && resp.features.some((f) => Number.isFinite(f?.value));
  const featuresUsable = hasUsableFeatures(shapUsable);
  const PSI_EMPTY_SOURCE = "드리프트 판정(PSI)";
  const emptySources = [
    // 6계열 중 일부만 비어도 "성능 지표 전체가 데모"로 읽히지 않게 결손 계열을 밝힌다.
    lastCollected && !allSeriesUsable ? `성능 지표(${emptySeries.join("·")})` : null,
    lastCollected && !featuresUsable ? "특징 기여도" : null,
    driftStatus === "ok" && !psiUsable ? PSI_EMPTY_SOURCE : null
  ].filter(Boolean);
  // PSI만 비어 있는 상태는 상단 배너로 계속 띄우지 않는다 — 값이 없다는 사실은 드리프트 카드
  // 안의 한 줄 안내와 배지("판정 없음")·분포 차트 빈 상태 문구가 이미 같은 자리에서 알린다.
  // 수집 판정(툴바 문구·헤더 칩)에서는 빼지 않는다: 빠뜨리면 값이 없는데 "정상 수신"이 된다.
  const psiEmptyOnly = driftStatus === "ok" && !psiUsable;
  const bannerEmptySources = emptySources.filter((source) => source !== PSI_EMPTY_SOURCE);
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
        : collectPhase === "empty" && bannerEmptySources.length > 0
          ? {
              tone: "warn",
              message: `${bannerEmptySources.join("·")}에 사용할 실측 값이 없습니다. ${allowSeed ? "대체 값을 표시합니다." : "해당 값은 비워 두며, 재학습을 실행하면 실측 지표가 쌓입니다."}`
            }
          : null;

  // API 수신과 폴백 경로를 기록한다. 표시 OFF는 전달 경로와 무관하게 모든 데이터에 적용한다.
  // 판정은 위 수집 검증(seriesUsable·hasUsableFeatures·psiUsable)과 같은 함수를 재사용해야
  // "정상 수신" 문구와 화면에 남는 영역이 어긋나지 않는다.
  // 판정 단위는 소비자가 실제로 읽는 값이다 — 6계열 공통 판정을 쓰면 한 계열만 비어도 실응답으로
  // 채워진 KPI 카드·게이지까지 함께 가려진다. metricValue가 조회 계열보다 먼저 쓰는 승급 오버라이드
  // (백엔드 응답 PipelineRun.candidate_metrics)도 응답 유래이므로 같이 본다.
  // 6대 지표 추이 차트만 전 계열을 요구한다 — 값 하나가 아니라 계열 전체를 그리기 때문이다.
  // 실데이터 응답으로 채운 자리는 "api"가 아니라 "realdata"로 표기한다 — 전달 경로가 아니라
  // 어느 저장소에서 온 값인지를 화면에서 구분할 수 있어야 한다.
  const srcOf = (fromApi, resp = metricsResp) =>
    !fromApi ? "mock" : resp?.source === "realdata" ? "realdata" : "api";
  const metricFromApi = (key) => seriesUsable(key) || Number.isFinite(selectedMetricOverride[key]);
  const primaryMetricKeys = isRealdata ? ["mae", "baseline_mae"] : ["accuracy", "f1"];
  const accF1Src = srcOf(primaryMetricKeys.every(metricFromApi));
  const metricsChartSrc = srcOf(allSeriesUsable);
  // 소비자가 실제로 읽는 값(shapUsable) 기준이어야 한다 — 원본 응답(shapResp)으로 판정하면
  // 시드 응답이 왔을 때 빈 카드에 "api" 표기가 붙는다.
  const shapSrc = srcOf(hasUsableFeatures(shapUsable), shapResp);
  // 분포 차트는 buckets·reference·current를 한 응답에서 모두 받아야 실측이다(일부만 오면 mock 폴백이 섞인다).
  const driftChartSrc = srcOf(
    Array.isArray(driftUsable?.buckets) &&
      driftUsable.buckets.length > 0 &&
      Array.isArray(driftUsable?.reference) &&
      Array.isArray(driftUsable?.current),
    driftResp
  );
  const psiSrc = srcOf(psiUsable, driftResp);

  return (
    <>
      {/* 운영 툴바 — 수집 상태 + 데모 분포의 드리프트 판정·경보 */}
      <div className="monitor-toolbar">
        <span className="monitor-collected" ref={collectRef} tabIndex={-1}>
          <i
            className={`fa-solid ${collectStatus.icon}`}
            style={{ color: collectStatus.color }}
            aria-hidden="true"
          ></i>{" "}
          {collectStatus.text}
        </span>
        <div className="monitor-actions">
        <div className="monitor-options">
          <label className="compact-select-field">
            <span>대상 모델</span>
            <select className="select-control" aria-label="대상 모델" value={selectedInstance?.id ?? ""}
              onChange={(event) => setModelPick({ mode, id: event.target.value })}>
              {!selectableModels.length && <option value="">등록 모델 없음</option>}
              {selectableModels.map((model) => <option key={model.id} value={model.id}>
                {model.name} · {appData.regions.find((row) => row.id === model.region_id)?.name ?? "지역 미확인"} · {TARGET_LABELS[model.target] ?? "학습 대상 미확인"} · {typeName(model.model_type)}
              </option>)}
            </select>
          </label>
          <label className="compact-select-field">
            <span>조회 구간</span>
            <select className="select-control" value={windowHours} onChange={(e) => setWindowHours(Number(e.target.value))}>
              {WINDOW_OPTIONS.map((h) => (
                <option key={h} value={h}>최근 {h}개 관측</option>
              ))}
            </select>
          </label>
        </div>
        </div>
      </div>
      {/* 상태 안내 슬롯 — 배너가 붙고 떨어져도 아래 지표 카드가 위아래로 밀리지 않도록 높이를 예약한다(§11 상태 전환) */}
      <div className="monitor-state-slot">
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

      {driftInjected && (
        <NextStepBanner
          tone="warn"
          message="데모 분포에서 드리프트를 감지했습니다 — 등록한 모델과 파이프라인을 확인하세요"
          actionLabel="오케스트레이터에서 보기"
          targetTab="tab-mlops-orch"
        />
      )}

      <div className="grid-cols-3">
        <div className="card" style={{ padding: "var(--space-xl)" }} data-values-source={accF1Src}>
          <div className="stat-label">
            {primaryCard.label}
            <InfoTip text={primaryCard.tip} />          </div>
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
            <span className="trend-up">{primaryCard.trailing}</span>
          </div>
        </div>

        <div
          className={"card" + (driftInjected ? " glow-red" : "")}
          style={{ padding: "var(--space-xl)" }}
          data-values-source={psiSrc}
        >
          <div className="stat-label">
            Data Drift Status (PSI)
            <InfoTip
              text={
                driftUsable?.source === "realdata"
                  ? `실데이터 대표 피처 ${driftUsable.feature}의 학습 구간 대비 ${driftUsable.kind === "operational" ? "학습 이후 관측" : "검증"} 구간 PSI입니다. 0.2를 초과하면 분포 변화로 판정합니다.`
                  : "PSI는 기준 분포와 현재 분포의 차이를 계산합니다. 데모 입력은 예시 분포이며, PSI/KL 임계 초과 경보를 확인한 뒤 등록한 파이프라인을 선택해 수동 실행합니다."
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
          {psiEmptyOnly && (
            <p className="monitor-card-note" role="status">
              드리프트 판정에 사용할 실측 분포가 없습니다 — 재학습을 실행하면 판정 값이 쌓입니다.
            </p>
          )}
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
                  : isRealdata
                    ? "실데이터 실측 계열(후보 버전별 검증 지표)에서 Z-score 임계를 넘는 값이 없습니다. 이상값 로그는 학습 실행이 더 쌓이면 표시됩니다."
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
              {outlierCount === NO_VALUE
                ? "판정 없음"
                : isRealdata && !outlierMeasured
                  ? "실측 계열"
                  : outlierAlerting
                    ? "경고"
                    : "정상"}
            </span>
          </div>
        </div>
      </div>

      <div className="grid-details-split">
        <Card
          title="데이터 분포 변화 시각화"
          icon="fa-chart-area"
          headerRight={<PerfBadge ms={vizMs} />}
          data-values-source={driftChartSrc}
        >
          <div style={{ position: "relative", height: 320, width: "100%" }}>
            <Bar data={driftData} options={AXIS_OPTS} />
            {driftData.labels.length === 0 && (
              <ChartEmptyNote>
                선택 모델에 연결된 분포 비교 데이터가 없습니다. 모델과 학습데이터를 확인하세요.
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
                      {outlierMeasured || isRealdata
                        ? "실측 계열에서 이상치가 검출되지 않았습니다."
                        : EMPTY_HINT}
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
            데이터(실데이터 또는 데모 시드)에서 읽고, 값이 없으면 값을 채우는 방법(EMPTY_HINT)을 표기한다.
            데모 토글은 데이터 유무만 바꾸므로 트리거 자체는 토글과 무관하게 항상 렌더한다. */}
        <Card
          title={
            <>
              MLOps AI 평가지표 실시간 모니터링
              <InfoTip
                text={
                  latestMetrics
                    ? `조회 구간 첫 값에서 최근 값까지의 변화: ${latestMetrics}.`
                    : EMPTY_HINT
                }
                label="지표 변화 요약 보기"
              />
            </>
          }
          icon="fa-chart-column"
          data-values-source={metricsChartSrc}
          headerRight={
            <span
              style={{ fontSize: 11, color: "var(--text-muted)", whiteSpace: "nowrap" }}
              title={`요청 구간 최근 ${windowHours}개 중 ${plottedCount}개 관측 표시`}
            >
              관측 {plottedCount}/{windowHours}개
            </span>
          }
        >
          <div style={{ position: "relative", height: 280, width: "100%" }}>
            <Line data={metricsData} options={metricsChartOpts} />
            {!allSeriesUsable && (
              <ChartEmptyNote>
                {EMPTY_HINT}
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
                {EMPTY_HINT}
              </ChartEmptyNote>
            )}
          </div>
        </Card>
      </div>

      <Card title={<>
        모델 신뢰도 게이지
        <InfoTip text="선택한 등록 모델의 현재 활성 버전 검증 MAE·RMSE·WAPE입니다. MAE와 RMSE는 예측 오차의 크기이며 낮을수록 좋습니다. WAPE는 절대 오차 비율입니다. 신뢰 확률이나 정규화 판정 기준이 아니므로 색상 판정을 하지 않습니다. 위 차트는 별도 이력 자료입니다." />
      </>} icon="fa-gauge-high">
        <p className="realdata-hint" role="status">
          {selectedInstance ? selectedVersion
            ? `${selectedInstance.name} · 활성 버전 ${displayId(selectedVersion.id)}`
            : `${selectedInstance.name} · 활성 버전이 없습니다.` : "등록 모델을 확인하세요."}
          {modelTarget ? ` · 위 차트: ${realdataModels.find((row) => row.model_id === modelTarget)?.name}의 별도 학습 이력`
            : " · 위 차트에 연결된 이력 자료가 없습니다."}
        </p>
        <div className="grid-cols-3" style={{ marginBottom: 0 }}>
          {[{ key: "mae", label: "MAE", unit: targetUnit(selectedInstance?.target) },
            { key: "rmse", label: "RMSE", unit: targetUnit(selectedInstance?.target) },
            { key: "wape", label: "WAPE", unit: "%", percent: true }].map((metric) => {
              const raw = selectedVersion?.metrics?.[metric.key];
              const value = Number.isFinite(raw) && raw >= 0 ? raw : null;
              return <GaugeChart key={metric.key} neutral value={value} label={metric.label}
                displayText={regressionMetricText(value, metric.unit, metric.percent)}
                statusText={value === null ? selectedVersion ? "측정값 없음" : "활성 버전 없음" : "정규화 판정 없음"}
                data-values-source={value === null ? "empty" : "api"} />;
            })}
        </div>
      </Card>
    </>
  );
}
