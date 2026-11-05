// 시설 × 활용 방안 추천 — 지역 시설 자료의 관측값과 시설 조절변수만으로 행을 만들고,
// 사용자가 고른 정렬 기준 하나로 순서를 정한다. 가중합·점수·예측 효과는 쓰지 않는다.

// 시설 유형별 활용 방안(용도 전환·복합화·운영 개선·개방·공유). 통합 검토는 넣지 않는다.
const USE_PLANS = {
  임대주택: { type: "용도 전환", text: "공실 호실 단기 체류형 전환 검토" },
  스마트팜: { type: "운영 개선", text: "운영 시간·교육 프로그램 개선 검토" },
  신재생: { type: "개방·공유", text: "견학·체험 프로그램 개방 검토" },
  정원수: { type: "복합화", text: "정원 체험·방문 기능 복합화 검토" }
};
const DEFAULT_PLAN = { type: "운영 개선", text: "이용 실적 확인 후 활용 방안 검토" };

// 정렬 기준 — 값이 없는 시설은 방향과 관계없이 뒤에 둔다.
export const SORT_CRITERIA = {
  utilization: { label: "활용도", unit: "%", ascending: true, text: "활용도 낮은 순" },
  budget: { label: "운영 예산", unit: "억 원", ascending: false, text: "운영 예산 큰 순" }
};
export const DEFAULT_CRITERION = "utilization";

export const FACILITY_EVIDENCE_ROWS = [
  ["시설 자료", "시설 현황 자료(시설명·유형·운영 예산·입주율) · 활용도 정의: 주거 시설 입주율(입주 호수 기준), 개방 시설 이용률(수용 인원 기준) · 자료 기준일과 기금 투자 여부는 시설 대장으로 확인"],
  ["생활인구 관련 관측(외지인 방문·소비)", "시설 소재 행정동의 통신·카드 관측으로, 시군구·월 단위 공식 생활인구(주민등록·체류·등록외국인)와 다릅니다. 자료 사용 승인 후 표시됩니다."],
  ["정렬 기준", "선택한 기준 하나로만 순서를 정합니다. 시설 조절변수의 임대주택 공급 호수는 입주율에 반영됩니다."]
];

const isLinked = (c) => Boolean(c.facility_id && c.field);
const valueOf = (c, controlValues) => controlValues?.[c.key] ?? c.default;

/** 시설 조절변수 중 특정 시설 값에 연결된 것만 — 이 값이 바뀔 때만 추천을 다시 도출한다. */
export function linkedControls(controls) {
  return (controls ?? []).filter(isLinked);
}

/** 추천 재도출 판정 서명 — 시설 값에 연결된 조절변수만 넣는다. */
export function recoControlSig(controls, controlValues) {
  return linkedControls(controls).map((c) => `${c.key}:${valueOf(c, controlValues)}`).join(",");
}

/** 시설 값이 없는 조절변수를 바꿨을 때 재도출 안내 자리에 둘 행동 안내(바꾸지 않았으면 null). */
export function unlinkedControlNote(controls, controlValues) {
  const changed = (controls ?? []).filter((c) => !isLinked(c) && valueOf(c, controlValues) !== c.default);
  return changed.length ? `${changed.map((c) => c.label).join("·")}: 시설 자료를 등록하면 추천에 반영됩니다.` : null;
}

// 주거 시설 입주율(%) — 공급 호수가 조절되면 현재 입주 호수 기준으로 다시 계산한다(공급보다 많이 입주할 수 없음).
function occupancyOf(facility, supply) {
  if (!Number.isFinite(facility.occupancy_rate)) return null;
  if (!Number.isFinite(facility.supply_units) || !Number.isFinite(supply) || supply <= 0) return facility.occupancy_rate;
  const occupied = (facility.occupancy_rate / 100) * facility.supply_units;
  return Math.min(100, (occupied / supply) * 100);
}

/**
 * 지역 시설 → 추천 행(정렬 전). 연결된 조절변수 값은 해당 시설의 필드를 대신한다.
 * @param {string} regionId
 * @param {object[]} facilities 시설 자료(id·name·type·region_id·operating_budget·occupancy_rate·supply_units)
 * @param {object[]} controls 시설 조절변수(facility_id·field가 있으면 그 시설 값에 연결)
 * @param {Record<string, number>} controlValues
 */
export function facilityRows(regionId, facilities, controls = [], controlValues = {}) {
  const linked = linkedControls(controls);
  return (facilities ?? []).filter((f) => f.region_id === regionId).map((f) => {
    const adjusted = { ...f };
    linked.filter((c) => c.facility_id === f.id).forEach((c) => { adjusted[c.field] = valueOf(c, controlValues); });
    const plan = USE_PLANS[f.type] ?? DEFAULT_PLAN;
    const utilization = occupancyOf(f, adjusted.supply_units);
    return {
      id: f.id,
      name: f.name,
      planType: plan.type,
      plan: plan.text,
      utilization: utilization === null ? null : Math.round(utilization),
      utilizationLabel: utilization === null ? null : "입주율",
      budget: Number.isFinite(f.operating_budget) ? f.operating_budget : null
    };
  });
}

/** 선택한 기준 하나로 정렬(행 필드 이름 = 기준 키) — 같은 값은 시설 자료 순서를 유지한다. */
export function sortFacilityRows(rows, criterionKey = DEFAULT_CRITERION) {
  const { ascending } = SORT_CRITERIA[criterionKey];
  const key = criterionKey;
  return [...rows].sort((a, b) => {
    if (a[key] === null || b[key] === null) return (a[key] === null) - (b[key] === null);
    return ascending ? a[key] - b[key] : b[key] - a[key];
  });
}

const utilizationText = (row) => (row.utilization === null ? "이용실적 등록" : `${row.utilizationLabel} ${row.utilization}%`);
const budgetText = (row) => (row.budget === null ? "예산 자료 등록" : row.budget.toLocaleString("ko-KR"));
// Top3 카드 값 줄: 큰 글씨는 숫자만, 단위·지표 이름·자료 등록 안내는 작은 글씨 칸에 둔다(기존 "3,061 점" 줄과 같은 높이).
const scoreParts = (row, criterionKey) => {
  if (criterionKey === "utilization") return row.utilization === null ? ["", "이용실적 등록"] : [String(row.utilization), `%(${row.utilizationLabel})`];
  return row.budget === null ? ["", "예산 자료 등록"] : [budgetText(row), "억 원(운영 예산)"];
};

/** 정렬된 행 → 추천 패널 표시 형식(실데이터 민감도 패널과 같은 자리·같은 구조). */
export function facilityRecommendationView(rows, criterionKey = DEFAULT_CRITERION) {
  const criterion = SORT_CRITERIA[criterionKey];
  const items = sortFacilityRows(rows, criterionKey).map((row) => {
    const value = row[criterionKey];
    const [scoreText, scoreUnit] = scoreParts(row, criterionKey);
    return {
      id: row.id,
      name: row.name,
      category: row.plan,
      score: value,
      scoreText,
      scoreUnit,
      cells: [row.plan, utilizationText(row), "사용 승인 후 확인", "투자 대장 확인", "기준일 확인"],
      scoreCell: budgetText(row),
      missing: value === null
    };
  });
  return {
    items,
    criterion: criterionKey,
    title: "시설 활용 추천",
    scoreLabel: `${criterion.label}(${criterion.unit})`,
    scoreUnit: "",
    // 표 9열 = 순위 + columns 6 + 숫자 열(scoreColumn) 1 + 검토 구분 — 기존 추천 표와 같은 열 수·같은 숫자 열 자리.
    columns: [
      { label: "시설" },
      { label: "활용 방안" },
      { label: "활용도(주거 입주율·개방 시설 이용률)", sortKey: "utilization" },
      { label: "생활인구 관련 관측(외지인 방문·소비)" },
      { label: "기금 투자 여부" },
      { label: "자료 기준일" }
    ],
    scoreColumn: { label: "운영 예산(억 원)", sortKey: "budget" },
    rankText: (i) => `${i + 1}순위`,
    statusLabel: "검토 구분",
    status: (item, i) => (item.missing ? { text: "자료 등록 후 검토", strong: false } : { text: i < 3 ? "우선 검토" : "추가 검토", strong: i < 3 }),
    caption: `시설별 활용 방안과 관측값 — 정렬 기준 ${criterion.text}`,
    basisTip: "선택한 정렬 기준 하나로 시설 순서를 정합니다. 시설 자료 출처와 기준일은 추천 근거에서 확인하세요.",
    footnote: `※ 정렬 기준: ${criterion.text}(활용도·운영 예산 열 제목으로 변경) · 운영 예산은 시설별 관측값이며 총 예산과 합산하지 않습니다.`
  };
}
