// 시설 × 활용 방안 추천 — 지역 시설 자료와 시설 조절변수로 행을 만들고, 추천 지수(0~100)로 순위를 정한다.
// 화면에는 추천 지수 값만 보이고 구성 요소·비중·계산식은 보이지 않는다(계산 방법은 이 파일과 보고서에만).

// 시설 유형별 활용 방안(용도 전환·복합화·운영 개선·개방·공유). 시설 자료에 use_plan이 있으면 그것을 쓴다. 통합 검토는 넣지 않는다.
const USE_PLANS = {
  임대주택: { type: "용도 전환", text: "공실 호실 단기 체류형 전환 검토" },
  스마트팜: { type: "운영 개선", text: "운영 시간·교육 프로그램 개선 검토" },
  신재생: { type: "개방·공유", text: "견학·체험 프로그램 개방 검토" },
  정원수: { type: "복합화", text: "정원 체험·방문 기능 복합화 검토" }
};
const DEFAULT_PLAN = { type: "운영 개선", text: "이용 실적 확인 후 활용 방안 검토" };
export const PLAN_TYPES = ["용도 전환", "복합화", "운영 개선", "개방·공유"];

// 추천 지수 = 활용 여지(room)·생활인구 유입 연계(link)·추진 여건(readiness)의 가중 평균(각 0~100).
// 주거 시설은 활용 여지를 입주율로 계산한다(100 − 입주율) — 공급 호수 조절변수가 지수와 순위에 반영된다.
// 고정 가중치 — 시설 대장 실적(이용 실적·공실·운영 시간)이 들어오면 실적 기반 지표로 바꾼다.
const INDEX_WEIGHTS = { room: 0.4, link: 0.35, readiness: 0.25 };

// 표 정렬 기준 — rank(추천 순위)가 기본. 값이 없는 시설은 방향과 관계없이 뒤에 둔다.
export const SORT_CRITERIA = {
  rank: { label: "순위", unit: "", ascending: true, text: "추천 순위" },
  utilization: { label: "활용도", unit: "%", ascending: true, text: "활용도 낮은 순" },
  budget: { label: "운영 예산", unit: "억 원", ascending: false, text: "운영 예산 큰 순" }
};
export const DEFAULT_CRITERION = "rank";

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

// 추천 지수(0~100 정수) — 구성 값이 하나라도 없으면 null.
function recommendationIndex(inputs = {}, utilization) {
  const parts = { ...inputs, room: utilization === null ? inputs.room : 100 - utilization };
  const keys = Object.keys(INDEX_WEIGHTS);
  if (!keys.every((k) => Number.isFinite(parts[k]))) return null;
  return Math.round(keys.reduce((sum, k) => sum + INDEX_WEIGHTS[k] * parts[k], 0));
}

/**
 * 지역 시설 → 추천 행(순위 포함, 추천 지수 높은 순). 연결된 조절변수 값은 해당 시설의 필드를 대신한다.
 * @param {string} regionId
 * @param {object[]} facilities 시설 자료(id·name·type·region_id·operating_budget·occupancy_rate·supply_units·use_plan·reco_inputs)
 * @param {object[]} controls 시설 조절변수(facility_id·field가 있으면 그 시설 값에 연결)
 * @param {Record<string, number>} controlValues
 */
export function facilityRows(regionId, facilities, controls = [], controlValues = {}) {
  const linked = linkedControls(controls);
  const rows = (facilities ?? []).filter((f) => f.region_id === regionId).map((f) => {
    const adjusted = { ...f };
    linked.filter((c) => c.facility_id === f.id).forEach((c) => { adjusted[c.field] = valueOf(c, controlValues); });
    const plan = f.use_plan ?? USE_PLANS[f.type] ?? DEFAULT_PLAN;
    const utilization = occupancyOf(f, adjusted.supply_units);
    return {
      id: f.id,
      name: f.name,
      type: f.type,
      planType: plan.type,
      plan: plan.text,
      index: recommendationIndex(f.reco_inputs, utilization),
      utilization: utilization === null ? null : Math.round(utilization),
      utilizationLabel: utilization === null ? null : "입주율",
      budget: Number.isFinite(f.operating_budget) ? f.operating_budget : null
    };
  });
  // 추천 순위: 지수 높은 순(값 없는 시설은 뒤, 같은 값은 시설 자료 순서).
  const byIndex = [...rows].sort((a, b) => ((a.index === null) - (b.index === null)) || (b.index ?? 0) - (a.index ?? 0));
  return byIndex.map((row, i) => ({ ...row, rank: i + 1 }));
}

/** 선택한 기준 하나로 정렬(행 필드 이름 = 기준 키) — 같은 값은 입력 순서를 유지한다. */
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

/**
 * 추천 행 → 추천 패널 표시 형식(실데이터 민감도 패널과 같은 자리·같은 구조).
 * items(Top3·주 그래프)는 항상 추천 순위, tableItems(표)는 사용자가 고른 기준으로 정렬한다.
 */
export function facilityRecommendationView(rows, criterionKey = DEFAULT_CRITERION) {
  const criterion = SORT_CRITERIA[criterionKey];
  const items = sortFacilityRows(rows, "rank").map((row) => ({
    id: row.id,
    name: row.name,
    chartLabel: `${row.name} · ${row.planType}`,
    category: row.plan,
    planType: row.planType,
    type: row.type,
    rank: row.rank,
    utilization: row.utilization,
    budget: row.budget,
    score: row.index,
    // Top3 카드 값 줄: 큰 글씨는 숫자만, 지표 이름은 작은 글씨 칸(기존 "3,061 점" 줄과 같은 높이).
    scoreText: row.index === null ? "" : String(row.index),
    scoreUnit: row.index === null ? "자료 등록 후 산정" : "추천 지수",
    cells: [row.plan, utilizationText(row), "사용 승인 후 확인", "투자 대장 확인", "기준일 확인"],
    scoreCell: budgetText(row),
    missing: row.index === null
  }));
  return {
    items,
    tableItems: sortFacilityRows(items, criterionKey),
    criterion: criterionKey,
    title: "시설 활용 추천",
    scoreLabel: "추천 지수(0~100)",
    scoreMax: 100,
    scoreUnit: "",
    // 표 9열 = 순위 + columns 6 + 숫자 열(scoreColumn) 1 + 검토 구분 — 기존 추천 표와 같은 열 수·같은 숫자 열 자리.
    rankColumn: { label: "순위", sortKey: "rank" },
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
    // 검토 구분은 표 정렬과 관계없이 추천 순위로 정한다.
    status: (item) => (item.missing ? { text: "자료 등록 후 검토", strong: false } : { text: item.rank <= 3 ? "우선 검토" : "추가 검토", strong: item.rank <= 3 }),
    caption: `시설별 활용 방안과 관측값 — 정렬 기준 ${criterion.text}`,
    basisTip: "추천 지수가 높은 시설부터 순위를 정합니다.",
    footnote: `※ 순위: 추천 지수(0~100) 높은 순 · 표 정렬 기준: ${criterion.text}(순위·활용도·운영 예산 열 제목으로 변경) · 운영 예산은 시설별 관측값이며 총 예산과 합산하지 않습니다.`
  };
}

// 추천 카드 끝 그래프 묶음 — 두 모드가 같은 세 자리를 갖는다. 값이 없으면 같은 자리에 행동 안내만.
export const EVIDENCE_CHART_SLOTS = [
  { id: "population", title: "생활인구 관측(공식 통계)", guide: "생활인구 통계를 연결하면 표시됩니다." },
  { id: "case", title: "지역 사례 성과", guide: "시설 이용실적을 등록하면 표시됩니다." },
  { id: "plans", title: "추천 활용 방안 구성", guide: "시설 자료를 등록하면 표시됩니다." }
];

/**
 * 그래프 묶음 표시 형식. regionCharts(지역 출처 수치)와 items(추천 행)가 없으면 모든 자리가 행동 안내.
 * @param {object|null} regionCharts mock_data.recommendation_charts[지역] — { population, case }
 * @param {object[]|null} items facilityRecommendationView(...).items
 */
export function evidenceChartsView(regionCharts, items) {
  const planCounts = PLAN_TYPES.map((type) => ({ label: type, value: (items ?? []).filter((i) => i.planType === type).length }))
    .filter((p) => p.value > 0);
  const derived = planCounts.length
    ? { scope: `추천 시설 ${items.length}곳 · 활용 방안별 시설 수`, unit: "곳", points: planCounts, note: "기준: 위 추천 결과" }
    : null;
  const dataOf = { population: regionCharts?.population, case: regionCharts?.case, plans: derived };
  return EVIDENCE_CHART_SLOTS.map((slot) => {
    const data = dataOf[slot.id] ?? null;
    const points = data?.points ?? [];
    return {
      id: slot.id,
      title: slot.title,
      scope: data ? `${data.scope} (${data.unit})` : "",
      unit: data?.unit ?? "",
      labels: points.map((p) => p.label),
      values: points.map((p) => p.value),
      note: data ? (data.note ?? `출처: ${data.source_text}`) : "",
      empty: points.length === 0 ? slot.guide : null
    };
  });
}
