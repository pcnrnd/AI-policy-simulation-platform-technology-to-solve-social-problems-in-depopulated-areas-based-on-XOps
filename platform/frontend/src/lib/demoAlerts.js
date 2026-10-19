export function retainRealAlerts(entries = []) {
  return entries.filter((entry) => entry.source !== "demo");
}

// 모드 전환은 데이터 결과만 폐기한다. 세션·시스템 알림의 객체와 읽음 상태는 그대로 둔다.
export function retainModeIndependentAlerts(entries = []) {
  return entries.filter((entry) => entry.system === true || entry.source === "session" || entry.source === "system");
}

export function notificationFromAlert(alert) {
  return {
    severity: alert.system ? "info" : "warn",
    title: alert.title,
    message: alert.message,
    source: alert.source,
    system: alert.system === true
  };
}

export function unreadNotifications(entries = []) {
  return entries.filter((entry) => entry.unread).length;
}

// 데모 입력 분포의 백엔드 계산 결과를 현재 모드에만 반영한다.
export async function detectDemoDrift({ loadDrift, isCurrent, onDetected }) {
  if (!isCurrent()) return;
  const result = await loadDrift();
  if (!isCurrent()) return;
  if (result?.source !== "seed" || ![result.psi, result.kl, result.psi_threshold, result.kl_threshold].every(Number.isFinite)) {
    throw new Error("데모 분포 판정 결과를 확인하지 못했습니다.");
  }
  if (result.drifted === true && (result.psi >= result.psi_threshold || result.kl >= result.kl_threshold)) {
    onDetected(result, {
      source: "demo", title: "[데모 경보] 데이터 드리프트 발생",
      message: `데모 분포 PSI ${result.psi} (임계 ${result.psi_threshold}) · KL ${result.kl} (임계 ${result.kl_threshold}). 등록한 모델과 파이프라인을 확인하세요.`
    });
  }
}

// Context가 유지하는 epoch ref를 단일 잠금으로 쓴다. 이전 요청은 새 epoch의 잠금을 해제하지 않는다.
export async function runDemoDriftCheck({ requestEpochRef, epoch, isCurrent, onChecking, loadDrift, onDetected, onError }) {
  if (!isCurrent() || requestEpochRef.current !== null) return;
  requestEpochRef.current = epoch;
  onChecking(true);
  try {
    await detectDemoDrift({ loadDrift, isCurrent, onDetected });
  } catch (error) {
    if (isCurrent()) onError(error);
  } finally {
    if (requestEpochRef.current === epoch) {
      requestEpochRef.current = null;
      if (isCurrent()) onChecking(false);
    }
  }
}
