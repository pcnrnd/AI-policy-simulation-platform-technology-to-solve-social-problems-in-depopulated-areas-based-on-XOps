import { useId, useMemo, useState } from "react";
import { Bar } from "react-chartjs-2";
import { BarElement, CategoryScale, Chart, Legend, LinearScale, Tooltip } from "chart.js";
import { TARGET_AGE_LABEL } from "../lib/ingu.js";
import { scenarioState, typeResult, usageChanges } from "../lib/scenario.js";
import { bandLabels, dongStatus, dongStatusLabels, kmText, radiusIn, radiusList, regionOf, servedPop } from "../lib/spatial.js";
import { fmt, km, signedPeople } from "./format.js";
import { CompareTable, compareRows, ratePct } from "./ScenarioTabs.jsx";

Chart.register(BarElement, CategoryScale, LinearScale, Tooltip, Legend);

function Rows({ rows }) {
  return (
    <dl className="rows">
      {rows.map(([k, v]) => (
        <div key={k}><dt>{k}</dt><dd>{v}</dd></div>
      ))}
    </dl>
  );
}

// 고를 유형: 시설 분류 그룹별(청년 지원 시설 그룹 포함), except 유형은 뺌
function KindSelect({ label, value, onChange, groups, except }) {
  const id = useId();
  return (
    <div className="kind-select">
      <label htmlFor={id}>{label}</label>
      <select id={id} value={value} onChange={(e) => onChange(e.target.value)}>
        {groups.map((g) => (
          <optgroup key={g.id} label={g.label}>
            {g.types.filter((t) => t.id !== except).map((t) => <option key={t.id} value={t.id}>{t.label}</option>)}
          </optgroup>
        ))}
      </select>
    </div>
  );
}

/**
 * 동작 하나를 더했을 때 전·후 미리 보기: 보일 유형마다 기준 거리 밖 대상 인구·부족 읍면동·평균 거리·최고 사용률(영향 시설),
 * 정주·생활인구 유입 합, ◉ 유형 거리 구간 막대그래프.
 */
function ActionPreview({ scenario, candidate, showTypes, type }) {
  const { baseFacilities, actions, units, settings, capacities, envOf, state } = scenario;
  const after = useMemo(() => scenarioState(baseFacilities, [...actions, candidate], units, settings), [baseFacilities, actions, units, settings, candidate]);
  const res = showTypes.map((t) => {
    const env = envOf(t);
    const b = typeResult(state, t, env, capacities[t.id]);
    const a = typeResult(after, t, env, capacities[t.id]);
    return { t, env, b, a, rows: compareRows(t, env, b, a, state, after) };
  });
  const head = res.find((r) => r.t.id === type?.id) ?? res[0];
  const many = res.length > 1;
  const rows = [
    ...res.flatMap((r) => [
      ...(many ? [[r.t.label, "", ""]] : []),
      ...r.rows.slice(0, 4),
      ["영향 시설", "", `${usageChanges(r.b.usage, r.a.usage).length}곳`]
    ]),
    ...head.rows.slice(4)
  ];
  const data = {
    labels: bandLabels(radiusList(head.env)),
    datasets: [
      { label: "전", data: head.b.summary.bands, backgroundColor: "#b8b2a6" },
      { label: "후", data: head.a.summary.bands, backgroundColor: head.t.color }
    ]
  };
  const options = {
    responsive: true, maintainAspectRatio: false, animation: { duration: 250 },
    plugins: { legend: { position: "bottom", labels: { boxWidth: 12 } }, tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${fmt(c.parsed.y)}명` } } },
    scales: { y: { beginAtZero: true, title: { display: true, text: "대상 인구(명)" } }, x: { title: { display: true, text: `최근접 ${head.t.label}까지 거리` } } }
  };
  return (
    <>
      <div className="chart-box"><Bar data={data} options={options} aria-label="거리 구간별 대상 인구 전후 막대그래프" /></div>
      <CompareTable head={["지표", "전", "후"]} rows={rows} />
    </>
  );
}

// 동작 줄: 카드 맨 아래에 붙어 있고(스크롤 상자 안 sticky), 상세·미리 보기는 그 위에서 스크롤한다
const ActionBar = ({ children }) => <div className="card-actions">{children}</div>;

// 보일 유형: 같은 유형은 한 번, 분석 기준이 없으면(null) 뺀다
const uniqueTypes = (list) => [...new Map(list.filter(Boolean).map((t) => [t.id, t])).values()];

// 읍면동·격자: 여기에 시설을 추가(청년 지원 시설 유형이면 정주·생활 유입도 생김).
// 기본 유형 = defaultTo(분석 기준 유형, 그룹 기준이면 시설이 가장 많은 공급 유형, 기준이 없으면 목록 첫 유형).
function AddPanel({ site, type, defaultTo, groups, scenario, onAdd }) {
  const [to, setTo] = useState(defaultTo ?? groups[0]?.types[0]?.id);
  const candidate = useMemo(() => ({ id: "preview", kind: "add", to, lat: site.lat, lng: site.lng, name: "" }), [to, site]);
  const showTypes = uniqueTypes([scenario.typeById(to), type]);
  return (
    <>
      <div className="impact"><ActionPreview scenario={scenario} candidate={candidate} showTypes={showTypes} type={type} /></div>
      <ActionBar>
        <KindSelect label="추가할 시설" value={to} onChange={setTo} groups={groups} />
        <button type="button" className="primary" onClick={() => onAdd(site, to)}>추가</button>
      </ActionBar>
    </>
  );
}

// 기존 시설: 다른 유형으로 전환 — 원래 유형·◉ 유형·새 유형 변화를 같이 보인다(같은 유형은 한 번)
function ConvertPanel({ facility, type, defaultTo, groups, scenario, onConvert }) {
  const own = scenario.typeById(facility.typeId);
  const firstOther = groups.flatMap((g) => g.types).find((t) => t.id !== own.id);
  const [to, setTo] = useState(defaultTo && own.id !== defaultTo ? defaultTo : firstOther.id);
  const candidate = useMemo(() => ({ id: "preview", kind: "convert", from: facility.id, to, name: "" }), [to, facility.id]);
  const showTypes = uniqueTypes([own, type, scenario.typeById(to)]);
  return (
    <>
      <div className="impact"><ActionPreview scenario={scenario} candidate={candidate} showTypes={showTypes} type={type} /></div>
      <ActionBar>
        <KindSelect label="전환할 종류" value={to} onChange={setTo} groups={groups} except={own.id} />
        <button type="button" className="primary" onClick={() => onConvert(facility, to)}>전환</button>
      </ActionBar>
    </>
  );
}

function DongDetail({ unit, index, type, env, summary, scenario, ...rest }) {
  const dist = summary.dist[index];
  const count = rest.typeFacilities.filter((f) => f.dong === unit.id).length;
  // 그 읍면동이 쓴 기준 거리로(시·군 시작값이 섞인 동안은 지역마다 다름)
  const status = dongStatusLabels(summary.radii[index], env.minPop)[dongStatus(summary, index, env)];
  return (
    <>
      <h2>{unit.name}</h2>
      <Rows rows={[
        ["총인구", `${fmt(unit.pop)}명`],
        ["대상 인구", `${fmt(summary.pops[index])}명`],
        [`${type.label} 수`, `${count}곳`],
        [`최근접 ${type.label}`, km(dist)],
        ["상태", status],
        // 정주인구 변화 = 이 읍면동의 인구 증감분 + 청년 지원 시설 정주 유입(App settledChange)
        ["정주인구 변화", signedPeople(scenario.settledByDong[unit.id] ?? 0)],
        ["생활인구 유입", `${fmt(scenario.state.living[unit.id] ?? 0)}명`]
      ]} />
      {unit.lat != null
        ? <AddPanel key={type.id} site={unit} type={type} scenario={scenario} {...rest} />
        : <p className="hint">지도에서 가까운 격자를 눌러 시설을 추가하세요.</p>}
    </>
  );
}

// 격자 지점: 분석 기준이 있으면 최근접 거리·상태 줄, 없으면 추가만
function CellDetail({ cell, type, env, ...rest }) {
  const r = radiusIn(env, cell.region);
  return (
    <>
      <h2>선택 지점</h2>
      {type && <Rows rows={[
        [`최근접 ${type.label}`, km(cell.dist)],
        ["상태", `${kmText(r)} ${cell.dist <= r ? "이내" : "밖"}`]
      ]} />}
      <AddPanel key={type?.id ?? "none"} site={cell} type={type} {...rest} />
    </>
  );
}

const actionButton = (item, scenario, onRemoveAction) => {
  const a = scenario.actions.find((x) => x.id === item.actionId);
  return a && <ActionBar><button type="button" className="primary" onClick={() => onRemoveAction(a.id)}>{a.kind === "add" ? "삭제" : "전환 되돌리기"}</button></ActionBar>;
};

// 시설 상세: 유형·대상 인구·사용률·배정 읍면동. 기존 시설은 전환, 추가·전환 시설은 삭제·되돌리기.
function FacilityDetail({ facility, units, typeById, population, scenario, type, defaultTo, groups, onConvert, onRemoveAction }) {
  const t = typeById(facility.typeId);
  const env = scenario.envOf(t);
  const use = useMemo(() => typeResult(scenario.state, t, env, scenario.capacities[t.id]).usage[facility.id], [scenario, t, env, facility.id]);
  const unit = units.find((u) => u.id === facility.dong);
  const dong = population?.dongs?.[facility.dong];
  return (
    <>
      <h2>{facility.name}</h2>
      <Rows rows={[
        ["유형", t.label],
        ["구분", facility.converted ? "전환 시설" : facility.added ? "추가 시설" : "기존 시설"],
        ["대상 인구", TARGET_AGE_LABEL[t.target]],
        ...(units.length ? [
          [`${kmText(radiusIn(env, regionOf(facility.dong)))} 이내 대상 인구`, `${fmt(servedPop(units, facility, t, env))}명`],
          ["사용률", ratePct(use?.rate)],
          ["배정 읍면동", `${use?.dongs.length ?? 0}곳`]
        ] : []),
        ...(unit && dong ? [
          ["소재 읍면동", unit.name],
          ["읍면동 인구", `${fmt(dong.total)}명`]
        ] : [])
      ]} />
      {facility.actionId
        ? actionButton(facility, scenario, onRemoveAction)
        : units.length > 0 && <ConvertPanel key={`${facility.id}-${type?.id}`} facility={facility} type={type} defaultTo={defaultTo} groups={groups} scenario={scenario} onConvert={onConvert} />}
    </>
  );
}

// 지도 위 선택 상세 — 무언가를 골랐을 때만 나오고, × 로 선택을 푼다.
export default function SelectionCard(props) {
  const { units, grid, facilities, typeFacilities, type, defaultTo, groups, typeById, population, env, summary, scenario, selection } = props;
  const { onAdd, onConvert, onRemoveAction, onClose } = props;
  if (!selection) return null;
  // 읍면동 상세는 분석 기준이 있을 때만(기준이 없으면 읍면동 원도 없음)
  const dongIndex = selection.kind === "dong" && summary ? units.findIndex((u) => u.id === selection.id) : -1;
  const cell = selection.kind === "cell" ? grid.find((c) => c.id === selection.id) : null;
  const facility = selection.kind === "facility" ? facilities.find((f) => f.id === selection.id) : null;
  const addProps = { typeFacilities, type, defaultTo, groups, env, scenario, onAdd };
  const hasPop = units.length > 0;

  return (
    <section className="detail selection-card" aria-label="선택 결과 상세">
      <button type="button" className="close" aria-label="선택 닫기" onClick={onClose}>×</button>
      {dongIndex >= 0 && <DongDetail unit={units[dongIndex]} index={dongIndex} summary={summary} {...addProps} />}
      {cell && hasPop && <CellDetail cell={cell} {...addProps} />}
      {facility && (
        <FacilityDetail
          facility={facility} units={units} typeById={typeById} population={population} scenario={scenario}
          type={type} defaultTo={defaultTo} groups={groups} onConvert={onConvert} onRemoveAction={onRemoveAction}
        />
      )}
      {dongIndex < 0 && !(cell && hasPop) && !facility && <p className="hint">지도에서 읍면동·격자·시설을 다시 고르세요.</p>}
    </section>
  );
}
