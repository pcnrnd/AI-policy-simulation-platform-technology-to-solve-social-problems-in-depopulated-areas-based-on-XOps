import { useEffect, useMemo, useRef, useState } from "react";
import { boundaryBounds, insideAny } from "./lib/boundary.js";
import { dbView, targetBands } from "./lib/ingu.js";
import {
  actionsWithin, basisDefaultType, basisTypeIds, capacityOf, changeVisible, exampleInflowFacilities, groupBasis, growthByDong, INFLOW_SETTINGS,
  meanRate, scenarioState, settledChange, toggleBasis, typeResult
} from "./lib/scenario.js";
import { accessGrid, assignDongs, basePopMax, dataBounds, DEFAULT_MIN_POP, dongUnits, minPopScale, padBounds, radiusText, regionStartKm, withRegions } from "./lib/spatial.js";
import ControlPanel from "./components/ControlPanel.jsx";
import MapView from "./components/MapView.jsx";
import SelectionCard from "./components/SelectionCard.jsx";
import MapControls from "./components/MapControls.jsx";
// 시설 자료가 없을 때 쓰는 빈 유형(목록에는 나오지 않음)
const NO_TYPE = { id: "none", label: "시설", target: "elderly", color: "#8a8378", radiusKm: 3 };
// 분석 기준이 없을 때 거리 슬라이더 값을 두는 자리(유형 id와 겹치지 않음)
const ANY_TYPE = "*";

// 시설·행정동 인구·이름·경계는 로컬 API(api.js)가 고른 시군구 코드들로 읽어 준다. 처음(코드 없음)은 API 기본 지역(REGION_CODE).
const EMPTY_DB = { region: null, regions: [], regionList: [], facilities: [], names: {}, population: null, boundaries: [] };
function useDbData(codes) {
  const [db, setDb] = useState({ status: "loading", ...EMPTY_DB });
  const key = codes.join(",");
  useEffect(() => {
    let alive = true;
    fetch(key ? `/api/data?regions=${key}` : "/api/data")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((d) => alive && setDb({ status: "ok", ...EMPTY_DB, ...d }))
      .catch(() => alive && setDb((prev) => ({ ...prev, status: "error" })));
    return () => { alive = false; };
  }, [key]);
  return db;
}

export default function App() {
  const [regionCodes, setRegionCodes] = useState([]);
  const db = useDbData(regionCodes);
  // 행정동 단위: 대표 지점은 API가 붙인 경계 안 중심점(spatial.js dongUnits)
  const units = useMemo(() => dongUnits(db.population, db.names), [db.population, db.names]);
  // 청년 지원 시설 그룹의 예시 시설(지역 코드 고정 시드, 실제 자료 아님) — DB·파일 시설 뒤에 붙인다
  const examples = useMemo(() => exampleInflowFacilities(units, db.region), [units, db.region]);
  const view = useMemo(() => dbView([...db.facilities, ...examples]), [db.facilities, examples]);
  const types = useMemo(() => (view.types.length ? view.types : [NO_TYPE]), [view.types]);
  // 지도 범위·격자 = 지역 경계 범위(경계가 없으면 시설 범위), 격자는 지역 안 칸만
  const regionBounds = useMemo(() => boundaryBounds(db.boundaries), [db.boundaries]);
  const bounds = useMemo(() => regionBounds ?? dataBounds(db.facilities), [regionBounds, db.facilities]);
  const inRegion = useMemo(() => insideAny(db.boundaries), [db.boundaries]);
  const typeById = useMemo(() => {
    const map = Object.fromEntries(types.map((t) => [t.id, t]));
    return (id) => map[id];
  }, [types]);
  // 분석 기준 후보: 유형 하나씩 + 그룹 전체(scenario.js groupBasis, id "g:{그룹}")
  const bases = useMemo(() => [...types, ...view.groups.map(groupBasis)], [types, view.groups]);
  const basisById = useMemo(() => {
    const map = Object.fromEntries(bases.map((b) => [b.id, b]));
    return (id) => map[id];
  }, [bases]);

  const [hidden, setHidden] = useState(() => new Set());
  const [chosenTypeId, setActiveTypeId] = useState(null);
  const [radiusEdits, setRadius] = useState({});
  // 연령 구간 id별 증감률(%) — 같은 구간을 쓰는 유형끼리 공유, 없는 구간은 0
  const [growth, setGrowth] = useState({});
  const [minPop, setMinPop] = useState(DEFAULT_MIN_POP);
  // 청년 지원 시설 1곳당 정주·생활인구·비중 = 고정 가정값(README "가정"). 정주·생활은 결과이고 입력은 인구 증감과 시설 추가·전환뿐이다.
  const settings = INFLOW_SETTINGS;
  // 추가·전환 동작 목록
  const [actions, setActions] = useState([]);
  const [selection, setSelection] = useState(null);
  const [focus, setFocus] = useState(null);
  // 조절 중인 조건("radius" | "pop" | null) — 조작하는 동안 지도에 그 값의 뜻을 그린다
  const [adjusting, setAdjusting] = useState(null);
  const addSeq = useRef(0);

  // 기준 거리 시작값: 고른 시군구마다 시·구 3km, 군 5km. 값이 하나면 그 값, 섞이면 슬라이더를 움직이기 전까지 지역별로 판정한다.
  const startKm = useMemo(() => {
    const names = Object.fromEntries(db.regionList.map((r) => [r.code, r.name]));
    return Object.fromEntries(db.regions.map((c) => [c, regionStartKm(names[c])]));
  }, [db.regions, db.regionList]);
  const starts = useMemo(() => [...new Set(Object.values(startKm))].sort((a, b) => a - b), [startKm]);
  // 분석 기준(◉): "분석" 탭에서 그룹·유형 줄을 눌러야 생기고, 같은 줄을 다시 누르면 없음(null) — 판정·요약 숫자 없음.
  // 유형 기준이면 유형 객체, 그룹 기준이면 groupBasis(members = 공급 유형). 시설 분류 탭의 표시 체크는 이 값을 바꾸지 않는다.
  const type = basisById(chosenTypeId) ?? null;
  const activeTypeId = type?.id ?? null;
  // 거리 슬라이더가 바꾸는 자리: 기준 유형, 기준이 없으면 ANY_TYPE(아직 바꾸지 않은 유형의 시작값이 됨)
  const radiusKey = activeTypeId ?? ANY_TYPE;
  const visible = useMemo(() => new Set(types.filter((t) => !hidden.has(t.id)).map((t) => t.id)), [types, hidden]);
  // 유형별 거리: 바꾼 유형은 바꾼 값, 아니면 기준 없이 바꾼 값, 아니면 시작값(섞이면 작은 값이 슬라이더 자리이고 지역별 판정)
  const envOf = useMemo(() => (t) => {
    const edited = radiusEdits[t.id] ?? radiusEdits[ANY_TYPE];
    return { radiusKm: edited ?? starts[0] ?? NO_TYPE.radiusKm, radiusByRegion: edited == null && starts.length > 1 ? startKm : null, growthPct: growth, minPop };
  }, [radiusEdits, starts, startKm, growth, minPop]);
  const env = useMemo(() => envOf({ id: radiusKey }), [envOf, radiusKey]);
  // 지도 크기·색과 최소 대상 인구 슬라이더의 기준 = 증감률 0일 때 대상 인구 최댓값(대상 유형이 바뀌면 다시 계산, 기준이 없으면 전체 인구)
  const target = type?.target ?? "all";
  const baseMax = useMemo(() => basePopMax(units, target), [units, target]);
  const popScale = useMemo(() => minPopScale(baseMax), [baseMax]);
  useEffect(() => {
    if (minPop > popScale.max) setMinPop(popScale.max);
  }, [minPop, popScale.max]);
  // 시나리오: 기존 시설 + 추가·전환 동작 → 시설·유입 배치·정주 유입을 붙인 계산 단위. 요약·판정·상세·숫자가 모두 이 상태로 계산한다.
  const baseFacilities = useMemo(() => assignDongs(view.facilities, units), [view.facilities, units]);
  // 동작은 지역을 바꿔도 남기고, 고른 지역 안 것만 계산·목록에 쓴다
  const liveActions = useMemo(() => actionsWithin(actions, baseFacilities, inRegion), [actions, baseFacilities, inRegion]);
  const state = useMemo(() => scenarioState(baseFacilities, liveActions, units, settings), [baseFacilities, liveActions, units, settings]);
  // 정주인구 변화 = 인구 증감분(모든 읍면동·구간) + 청년 지원 시설 정주 유입, 분석 기준과 관계없이
  const growthDong = useMemo(() => growthByDong(units, growth), [units, growth]);
  const settledNow = useMemo(() => settledChange(growthDong, state.settled), [growthDong, state.settled]);
  const facilities = state.facilities;
  const simUnits = state.units;
  // 1곳당 수용(사용률 분모) = 처음 상태 대상 인구 합 ÷ 처음 시설 수, 유형마다·그룹마다(그룹은 공급 유형 시설 전체 = 그룹 평균)
  const capacities = useMemo(
    () => Object.fromEntries(bases.map((b) => {
      const ids = new Set(basisTypeIds(b));
      return [b.id, capacityOf(units, baseFacilities.filter((f) => ids.has(f.typeId)), b.target)];
    })),
    [bases, units, baseFacilities]
  );
  // 분석 기준의 공급 시설(그룹이면 공급 유형 전부)
  const typeFacilities = useMemo(() => {
    const ids = new Set(basisTypeIds(type));
    return facilities.filter((f) => ids.has(f.typeId));
  }, [facilities, type]);
  // 추가·전환 목록 기본 유형: 유형 기준이면 그 유형, 그룹 기준이면 시설이 가장 많은 공급 유형
  const defaultTo = useMemo(() => basisDefaultType(type, facilities), [type, facilities]);
  // 기준이 없으면 result·base는 null(summary도 null)
  const result = useMemo(() => typeResult(state, type, env, capacities[activeTypeId]), [state, type, env, capacities, activeTypeId]);
  const summary = result?.summary ?? null;
  // 처음 상태(추가·전환 0, 증감 0, 지금 거리·부족 판정 기준) — 요약 숫자의 "처음" 비교값
  const base = useMemo(
    () => typeResult({ facilities: baseFacilities, units }, type, { ...env, growthPct: {} }, capacities[activeTypeId]),
    [baseFacilities, units, type, env, capacities, activeTypeId]
  );
  const scenario = { baseFacilities, actions: liveActions, units, settings, capacities, envOf, state, typeById, settledByDong: settledNow.byDong };
  // 지도 위 "변화" 띠: 처음 상태(추가·전환 0, 증감 0)와 다를 때만. 사용률 = 분석 기준 평균(처음은 base 평균 = 100%)
  const change = changeVisible(liveActions, growth) ? {
    rate: result ? meanRate(result.usage) : null,
    baseRate: base ? meanRate(base.usage) : null,
    settled: settledNow.total,
    living: state.livingTotal
  } : null;
  const grid = useMemo(() => {
    const cells = regionBounds
      ? accessGrid(regionBounds, typeFacilities).filter(inRegion)
      : bounds ? accessGrid(padBounds(bounds, env.radiusKm), typeFacilities) : [];
    // 여러 시군구면 칸마다 시군구를 붙여 둔다(기준 거리가 섞일 때 칸 색을 그 지역 값으로)
    return db.regions.length > 1 ? withRegions(cells, units) : cells;
  }, [regionBounds, inRegion, bounds, typeFacilities, env.radiusKm, db.regions.length, units]);

  // 지도 표시 체크(유형·그룹·전체) — 지도 표시만 바꾸고 분석 기준은 그대로
  const toggleTypes = (ids, on) => setHidden((prev) => {
    const next = new Set(prev);
    ids.forEach((id) => (on ? next.delete(id) : next.add(id)));
    return next;
  });
  // "분석" 탭 줄(그룹·유형) 누르기: 그 기준으로 분석, 같은 줄을 다시 누르면 분석 끔
  const chooseBasis = (id) => setActiveTypeId((cur) => toggleBasis(cur, id));
  const select = (kind, id, target) => {
    setSelection({ kind, id });
    if (target?.lat != null) setFocus(target);
  };
  const selectCell = (id) => select("cell", id);
  const selectDong = (id, fly = false) => select("dong", id, fly ? units.find((u) => u.id === id) : null);
  const selectFacility = (id, fly = false) => select("facility", id, fly ? facilities.find((f) => f.id === id) : null);
  // 추가(지점에 새 시설) · 전환(기존 시설을 다른 유형으로). to = 유형 id(청년 지원 시설 유형 포함). 그 유형을 켜고 만든 시설을 바로 고른다.
  const act = (action) => {
    addSeq.current += 1;
    const n = addSeq.current;
    const a = { ...action, id: n, name: `${typeById(action.to).label} ${action.kind === "add" ? "추가" : "전환"} ${n}` };
    setActions((prev) => [...prev, a]);
    toggleTypes([a.to], true);
    setSelection({ kind: "facility", id: `act-${n}` });
  };
  const addAt = (site, to) => act({ kind: "add", to, lat: site.lat, lng: site.lng });
  const convert = (facility, to) => act({ kind: "convert", from: facility.id, to });
  // 추가는 지우고, 전환은 되돌린다(원래 시설이 돌아옴)
  const removeAction = (actionId) => {
    setActions((prev) => prev.filter((a) => a.id !== actionId));
    setSelection((s) => (s?.id === `act-${actionId}` ? null : s));
  };
  // 지역을 바꾸면 고른 곳·이동은 풀고, 조건·◉·동작·옵션은 남긴다
  const changeRegions = (codes) => {
    setRegionCodes(codes);
    setSelection(null);
    setFocus(null);
  };

  return (
    <div className="app">
      <header className="topbar">
        <h1 className="brand">사회문제 해결지원 시뮬레이션 공간정보 표현</h1>
      </header>

      <main className="layout">
        <ControlPanel
          groups={view.groups}
          dbStatus={db.status}
          facilities={facilities}
          visible={visible}
          onToggleTypes={toggleTypes}
          activeTypeId={activeTypeId}
          onChooseBasis={chooseBasis}
          activeType={type}
          summary={summary}
          hasPop={units.length > 0}
          selection={selection}
          onSelectFacility={(id) => selectFacility(id, true)}
          scenario={scenario}
          result={result}
          base={base}
          regionList={db.regionList}
          regions={db.regions}
          onChangeRegions={changeRegions}
          env={env}
          onRemoveAction={removeAction}
        />
        <MapView
          bounds={bounds}
          boundaries={db.boundaries}
          grid={grid}
          units={simUnits}
          facilities={facilities}
          settled={state.settled}
          living={state.living}
          usage={result?.usage ?? {}}
          visible={visible}
          activeType={type}
          typeById={typeById}
          envOf={envOf}
          summary={summary}
          env={env}
          baseMax={baseMax}
          selection={selection}
          onSelectCell={selectCell}
          onSelectDong={selectDong}
          onSelectFacility={selectFacility}
          focus={focus}
          adjusting={adjusting}
          change={change}
        >
          <MapControls
            radiusKey={radiusKey}
            bands={type ? targetBands(type.target) : []}
            radiusKm={env.radiusKm}
            radiusLabel={radiusText(env)}
            setRadius={setRadius}
            growth={growth}
            setGrowth={setGrowth}
            minPop={minPop}
            setMinPop={setMinPop}
            popScale={popScale}
            onAdjust={setAdjusting}
          >
            <SelectionCard
              units={simUnits}
              grid={grid}
              facilities={facilities}
              typeFacilities={typeFacilities}
              groups={view.groups}
              type={type}
              defaultTo={defaultTo}
              typeById={typeById}
              population={db.population}
              env={env}
              summary={summary}
              usage={result?.usage ?? {}}
              scenario={scenario}
              selection={selection}
              onAdd={addAt}
              onConvert={convert}
              onRemoveAction={removeAction}
              onClose={() => setSelection(null)}
            />
          </MapControls>
        </MapView>
      </main>
    </div>
  );
}
