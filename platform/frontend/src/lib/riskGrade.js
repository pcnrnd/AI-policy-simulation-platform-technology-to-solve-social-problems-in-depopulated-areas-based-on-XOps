// 지방소멸위험지수 등급 — 지역 현황 카드·지도 마커/범례/팝업·시뮬레이터 지역 목록·리포트가 같은 기준을 쓴다.
// 기준: 한국고용정보원 지방소멸위험지수 원시자료(2025년 6월 기준) 자료설명의 소멸위험분류 6단계(2025-10-27 등급 구분 수정).
// 원시자료의 지수는 ×100 척도(양호 100 이상 … 심각 10 미만)이고, 시드 riskIndex는 ÷100 값이라 구간도 ÷100으로 적는다.
export const RISK_GRADE_SOURCE = {
  title: "한국고용정보원 지방소멸위험지수 원시자료(2025년 6월 기준) 자료설명 — 소멸위험분류(2025-10-27 수정)",
  url: "https://www.keis.or.kr/keis/ko/bbs/145/detail.do?pstSn=63580",
  period: "2025-06",
  checked: "2026-11-06"
};

// 낮을수록 위험. maxExclusive 미만이면 그 등급.
export const RISK_GRADES = [
  { label: "심각", rangeLabel: "0.1 미만", maxExclusive: 0.1, color: "var(--accent-red)", rgbToken: "--accent-red-rgb", markerColor: "#ef4444", rgb: "239, 68, 68" },
  { label: "위험", rangeLabel: "0.1 이상 0.2 미만", maxExclusive: 0.2, color: "var(--accent-orange)", rgbToken: "--accent-orange-rgb", markerColor: "#f59e0b", rgb: "245, 158, 11" },
  { label: "경계", rangeLabel: "0.2 이상 0.4 미만", maxExclusive: 0.4, color: "var(--accent-purple)", rgbToken: "--accent-purple-rgb", markerColor: "#a855f7", rgb: "168, 85, 247" },
  { label: "관리", rangeLabel: "0.4 이상 0.6 미만", maxExclusive: 0.6, color: "var(--accent-blue)", rgbToken: "--accent-blue-rgb", markerColor: "#3b82f6", rgb: "59, 130, 246" },
  { label: "보통", rangeLabel: "0.6 이상 1.0 미만", maxExclusive: 1.0, color: "var(--accent-teal)", rgbToken: "--accent-teal-rgb", markerColor: "#10b981", rgb: "16, 185, 129" },
  { label: "양호", rangeLabel: "1.0 이상", maxExclusive: Number.POSITIVE_INFINITY, color: "var(--accent-teal)", rgbToken: "--accent-teal-rgb", markerColor: "#22c55e", rgb: "34, 197, 94" }
];

/** 위험지수(÷100 척도) → 등급. */
export function riskGrade(riskIndex) {
  return RISK_GRADES.find((grade) => riskIndex < grade.maxExclusive) ?? RISK_GRADES.at(-1);
}
