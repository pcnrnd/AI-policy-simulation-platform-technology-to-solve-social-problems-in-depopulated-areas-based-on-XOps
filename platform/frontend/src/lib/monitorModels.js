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

// 데모 표시의 지표·설명·분포 이력은 인구이동 예측 예시 모델 한 벌이다. 데모 모드 등록 모델(인구 변화율)이
// 없거나 있어도 이 이력을 본다 — null 이면 화면 전체가 빈 값이 된다(2026-10-22 데모 ON 모니터 미표시).
export const DEMO_MONITOR_ID = "population-forecast";

export function legacyMonitorId(instance, rows, mode) {
  if (mode === "mock") return !instance || instance.mode === "mock" ? DEMO_MONITOR_ID : null;
  if (mode !== "real" || instance?.mode !== mode) return null;
  const id = LEGACY_MODELS[instance.region_id]?.[instance.target];
  return id && rows.some((row) => row.model_id === id && row.metrics_source === "realdata") ? id : null;
}

export function regressionMetricText(value, unit, percent = false) {
  if (!Number.isFinite(value) || value < 0) return "–";
  const shown = percent ? value * 100 : value;
  return `${shown.toLocaleString("ko-KR", { maximumFractionDigits: percent || unit === "%" ? 2 : 0 })}${unit}`;
}
