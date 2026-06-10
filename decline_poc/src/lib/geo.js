// 공간정보 격자/단계구분도 헬퍼 — 공간정보 데이터 소스(ds_04, 격자·인구밀도) 개념을 시각화.
// mock_data에는 실제 폴리곤 좌표가 없어, 지자체 중심점 주변에 합성 격자(grid)를 생성한다.

// 인구밀도 단계 색상(낮음→높음). 단계구분도(choropleth) 표준 팔레트.
const DENSITY_BINS = [
  { max: 20, color: "#1e3a5f" },
  { max: 40, color: "#2563eb" },
  { max: 60, color: "#7c3aed" },
  { max: 80, color: "#db2777" },
  { max: Infinity, color: "#ef4444" }
];

export function densityColor(density) {
  return (DENSITY_BINS.find((b) => density <= b.max) ?? DENSITY_BINS[DENSITY_BINS.length - 1]).color;
}

export const DENSITY_LEGEND = [
  { label: "~20", color: "#1e3a5f" },
  { label: "20~40", color: "#2563eb" },
  { label: "40~60", color: "#7c3aed" },
  { label: "60~80", color: "#db2777" },
  { label: "80+", color: "#ef4444" }
];

/**
 * 지자체 중심점 주변에 n×n 격자 셀과 ㎢당 인구밀도 값을 합성한다.
 * @returns {{ bounds: [[number,number],[number,number]], density: number, gridId: string }[]}
 */
export function buildRegionGrid(region, size = 3, span = 0.12) {
  const cells = [];
  const step = span / size;
  const baseDensity = (region.population / 1000) * (1 - region.riskIndex) * 2.2;

  for (let i = 0; i < size; i += 1) {
    for (let j = 0; j < size; j += 1) {
      const south = region.lat - span / 2 + i * step;
      const west = region.lng - span / 2 + j * step;
      // 결정적 변동(난수 미사용): 중심에서 멀수록 밀도 감소 + 격자별 미세 변화.
      const dist = Math.abs(i - (size - 1) / 2) + Math.abs(j - (size - 1) / 2);
      const factor = 1.1 - dist * 0.18 + ((i * size + j) % 3) * 0.05;
      cells.push({
        bounds: [
          [south, west],
          [south + step, west + step]
        ],
        density: Math.max(5, Math.round(baseDensity * factor)),
        gridId: `${region.id.toUpperCase()}-G${i}${j}`
      });
    }
  }
  return cells;
}
