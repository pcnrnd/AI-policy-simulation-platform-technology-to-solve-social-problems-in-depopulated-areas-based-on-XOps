export function retainRealAlerts(entries = []) {
  return entries.filter((entry) => entry.source !== "demo");
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
