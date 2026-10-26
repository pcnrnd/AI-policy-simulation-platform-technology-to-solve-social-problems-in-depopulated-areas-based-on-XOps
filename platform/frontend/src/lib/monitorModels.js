import { compactNumber, observedFeatureLabel } from "./realSimulation.js";

// src/realdata/binding.py와 snapshot.TARGETS의 명시적인 지역·학습 대상 매핑.
// 공통 인스턴스 ID는 rd_* 이력 API의 model_id로 보내지 않는다.
const LEGACY_MODELS = {
  namwon: {
    observed_sales_krw: "namwon-observed-sales-next-month",
    nonlocal_visitors: "namwon-nonlocal-visitors-next-month"
  }
};

export function monitorInstances(catalog, mode) {
  return catalog?.mode === mode && catalog?.source === mode
    ? (catalog.instances ?? []).filter((row) => row.mode === mode) : [];
}

export function monitorActiveVersion(catalog, instance, mode) {
  if (!monitorInstances(catalog, mode).includes(instance) || !instance?.active_version) return null;
  return catalog.versions?.find((row) => row.mode === mode && row.id === instance.active_version
    && row.instance_id === instance.id) ?? null;
}

// 데모 표시의 지표·설명·분포 이력은 인구이동 예측 예시 모델 한 벌이다. 이 이력은 같은 학습 대상(인구 변화율)의
// 데모 등록 모델에만 잇는다 — 대상 모델을 바꾸면 모든 카드가 함께 바뀌고, 이력이 없는 모델은 같은 자리에 안내를 둔다.
// 등록 모델이 하나도 없으면(첫 진입·빈 카탈로그) 예시 이력을 그대로 본다(2026-10-22 데모 ON 모니터 미표시).
export const DEMO_MONITOR_ID = "population-forecast";
const DEMO_MONITOR_TARGET = "population_change_pct";

export function legacyMonitorId(instance, rows, mode) {
  if (mode === "mock") return !instance || (instance.mode === "mock" && instance.target === DEMO_MONITOR_TARGET) ? DEMO_MONITOR_ID : null;
  if (mode !== "real" || instance?.mode !== mode) return null;
  const id = LEGACY_MODELS[instance.region_id]?.[instance.target];
  return id && rows.some((row) => row.model_id === id && row.metrics_source === "realdata") ? id : null;
}

/** 기본 대상 모델 — 이력이 연결된 활성 모델 → 이력이 연결된 모델 → 활성 모델 → 첫 모델 순. */
export function defaultMonitorInstance(catalog, rows, mode) {
  const models = monitorInstances(catalog, mode);
  const linked = (row) => Boolean(legacyMonitorId(row, rows, mode));
  const active = (row) => Boolean(monitorActiveVersion(catalog, row, mode));
  return models.find((row) => linked(row) && active(row)) ?? models.find(linked) ?? models.find(active) ?? models[0] ?? null;
}

/** 오차 지표 표기 — 비율(WAPE)은 %, 1만 이상은 "1.03억 원"처럼 축약해 지표 카드와 게이지가 같은 형식을 쓴다. */
export function regressionMetricText(value, unit, percent = false) {
  if (!Number.isFinite(value) || value < 0) return "–";
  const shown = percent ? value * 100 : value;
  if (!percent && shown >= 1e4) return `${compactNumber(shown)} ${unit}`;
  return `${shown.toLocaleString("ko-KR", { maximumFractionDigits: percent || unit === "%" ? 2 : 0 })}${unit}`;
}

/** 게이지 호 — 같은 구간 기준선 오차 대비 비율(0~1로 자름). 기준선이 없으면 null. */
export function baselineRatio(value, baseline) {
  if (!Number.isFinite(value) || !Number.isFinite(baseline) || baseline <= 0) return null;
  return Math.max(0, Math.min(1, value / baseline));
}

/** 평가지표 x축 — 실행·버전 ID 대신 그 안의 날짜("09-21"), 같은 날이 겹치면 "09-21 (2)". 날짜가 없으면 "n회차". */
export function runDateLabels(labels = []) {
  const seen = new Map();
  return labels.map((label, index) => {
    const match = String(label ?? "").match(/(20\d{2})(\d{2})(\d{2})/);
    const base = match ? `${match[2]}-${match[3]}` : `${index + 1}회차`;
    const count = (seen.get(base) ?? 0) + 1;
    seen.set(base, count);
    return count > 1 ? `${base} (${count})` : base;
  });
}

// 데모 이력(지역 인구 추이) 학습 피처 — 직전 기간 변화율 랙과 지역 정적 지표.
const DEMO_FEATURES = { riskIndex: "위험지수", birthRate: "출산율", agingIndex: "고령화지수" };
/**
 * SHAP 특징 표시명 — 내부 변수명 대신 한글. 행정동 원핫은 대응표의 행정동명, 대응표에 없으면 "행정동 <코드>".
 * @param {Array<{dong_code?: string, dong_name?: string}>} dongEntries 행정동 대응표 entries
 */
export function monitorFeatureLabel(feature, target, dongEntries = []) {
  const name = String(feature ?? "").split(" (")[0];
  const lag = name.match(/^rate_t-(\d+)$/);
  if (lag) return lag[1] === "1" ? "직전 기간 인구 변화율" : `${lag[1]}기간 전 인구 변화율`;
  if (DEMO_FEATURES[name]) return DEMO_FEATURES[name];
  if (name.startsWith("dong_")) {
    const dongName = dongEntries.find((row) => row?.dong_code === name.slice(5))?.dong_name;
    return dongName ? `행정동: ${dongName}` : `행정동 ${name.slice(5)}`;
  }
  return observedFeatureLabel(name, target);
}
