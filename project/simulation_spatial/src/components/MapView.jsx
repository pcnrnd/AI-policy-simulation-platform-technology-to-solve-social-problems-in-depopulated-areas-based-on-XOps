import { useEffect, useMemo, useState } from "react";
import L from "leaflet";
import { Circle, CircleMarker, GeoJSON, MapContainer, Marker, Polyline, Rectangle, TileLayer, Tooltip, useMap, useMapEvents } from "react-leaflet";
import { focusZoom, LABEL_H, labelWidth, placeLabels } from "../lib/labels.js";
import { basisTypeIds, isInflowKind, mapFacilities, usageLevel } from "../lib/scenario.js";
import { signedPeople } from "./format.js";
import { bandLabels, bandOf, dongStatus, dongStatusLabels, popText, radiusIn, radiusList, regionOf } from "../lib/spatial.js";

// 데이터를 불러오기 전 첫 화면(국토 전체). 데이터가 오면 그 범위로 맞춘다.
const COUNTRY_VIEW = { center: [36.4, 127.8], zoom: 7 };
// 배경지도는 항상 켠다. 타일을 못 받으면(인터넷 없음) 깨진 그림 대신 투명 칸을 둔다.
const BLANK_TILE = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7";

// 격자 = 최근접 시설까지 거리 4구간(bandOf, 영향 그래프와 같음). 원의 판정 색(빨강·초록·회색)과 겹치지 않게 파랑 한 계열, 가까울수록 연하게.
const BAND_COLOR = ["#eff3ff", "#bdd7e7", "#6baed6", "#2171b5"];
const GRID_OPACITY = 0.4;
// 읍면동 원 색 = 판정(dongStatus), 크기 = 대상 인구
const STATUS_COLOR = { shortage: "#c62828", within: "#2e7d32", outsideLow: "#9e9a92" };
// 행정동 경계선 — 얇게, 채우지 않고, 눌러도 아래 격자가 받도록 상호작용 없음
const BOUNDARY_STYLE = { color: "#3d4347", weight: 1, opacity: 0.6, fill: false };
// 여러 시군구를 고르면 경계선 색으로 시군구를 가른다(첫 시군구는 기본 색, 행정동 코드 앞 5자리 = 시군구)
const REGION_LINE = ["#3d4347", "#9a3412", "#1e40af", "#3f6212", "#86198f", "#155e75", "#92400e", "#4338ca"];
// 기준 거리 원 — 선택한 시설 하나(진하게)와 거리 조절 중 고른 유형 시설 전부(옅게, 격자·원·바탕지도가 읽히게)가 같이 쓴다
const radiusStyle = (color, faint) => ({ color, dashArray: "6 4", weight: faint ? 1.2 : 2, opacity: faint ? 0.6 : 1, fillOpacity: faint ? 0.03 : 0.06 });

// 인구 조절 중: 읍면동 원마다 대상 인구 숫자. 큰 값부터 자리를 잡고 겹치거나 범례·조건 카드에 가리는 숫자는 뺀다(labels.js).
function PopLabels({ units, pops, sizeOf }) {
  const map = useMap();
  const [, redraw] = useState(0);
  useMapEvents({ zoomend: () => redraw((n) => n + 1), moveend: () => redraw((n) => n + 1) });
  const origin = map.getContainer().getBoundingClientRect();
  const blocked = [...(map.getContainer().parentElement?.querySelectorAll(".legend, .map-controls") ?? [])].map((el) => {
    const b = el.getBoundingClientRect();
    return { x: b.left - origin.left, y: b.top - origin.top, w: b.width, h: b.height };
  });
  const items = units.flatMap((u, i) => {
    if (u.lat == null) return [];
    const p = map.latLngToContainerPoint([u.lat, u.lng]);
    const text = popText(pops[i]);
    return [{ id: u.id, x: p.x, y: p.y, r: sizeOf(pops[i]), w: labelWidth(text), h: LABEL_H, rank: pops[i], text, lat: u.lat, lng: u.lng }];
  });
  const byId = Object.fromEntries(items.map((it) => [it.id, it]));
  return placeLabels(items, blocked).map(({ id, dx, dy }) => {
    const it = byId[id];
    const icon = L.divIcon({ className: "pop-label", html: it.text, iconSize: [it.w, it.h], iconAnchor: [it.w / 2 - dx, it.h / 2 - dy] });
    return <Marker key={id} position={[it.lat, it.lng]} icon={icon} interactive={false} keyboard={false} />;
  });
}

// 유입 읍면동 고리·선택한 추가·전환의 연결선 색 — 시설 점·판정 색과 겹치지 않게 짙은 회색 한 가지
const INFLOW_COLOR = "#1f2937";
// ◉ 유형 시설 점 테두리 = 사용률 3단계(scenario.js usageLevel: 70% 미만·70~130%·130% 초과)
const USAGE_STROKE = { low: { color: "#d6d3d1", weight: 1.5 }, mid: { color: "#57534e", weight: 1.5 }, high: { color: "#111111", weight: 3.5 } };

/**
 * 시나리오 그림: 정주·생활 유입이 생긴 읍면동(원 바깥 점선 고리),
 * 고른 추가·전환 시설 → 배정된 읍면동 대표 지점까지 점선(◉가 아닌 청년 지원 시설은 유입이 생긴 그 읍면동까지).
 */
function ScenarioLayer({ units, pops, sizeOf, settled, living, usage, selected }) {
  const at = Object.fromEntries(units.map((u, i) => [u.id, { u, i }]).filter(([, v]) => v.u.lat != null));
  const ringed = Object.keys(at).filter((id) => settled[id] > 0 || living[id] > 0);
  const links = !selected ? [] : usage[selected.id]?.dongs ?? (isInflowKind(selected.typeId) ? [selected.dong] : []);
  return (
    <>
      {ringed.map((id) => (
        <CircleMarker
          key={`ring-${id}`} center={[at[id].u.lat, at[id].u.lng]} radius={sizeOf(pops[at[id].i]) + 5}
          pathOptions={{ color: INFLOW_COLOR, weight: 1.5, dashArray: "3 3", fill: false }} interactive={false}
        />
      ))}
      {selected && links.filter((id) => at[id]).map((id) => (
        <Polyline
          key={`link-${id}`} positions={[[selected.lat, selected.lng], [at[id].u.lat, at[id].u.lng]]}
          pathOptions={{ color: INFLOW_COLOR, weight: 1.5, dashArray: "4 3", opacity: 0.85 }} interactive={false}
        />
      ))}
    </>
  );
}

// 구간·판정 이름은 지금 조건 값(km·명)으로 — 조건 카드와 같은 표기. 사용률·유입은 한 줄씩.
// radii = 기준 거리 값들(시·군 시작값이 섞이면 [3, 5] → "3·5km 이내"). 제목 줄을 눌러(키보드 Enter·Space도) 접고 편다, 처음은 펼침.
function Legend({ radii, minPop, color, open, onToggle }) {
  return (
    <details className="legend" open={open} onToggle={(e) => onToggle(e.currentTarget.open)}>
      <summary>최근접 시설까지 거리</summary>
      <div className="legend-body">
      {bandLabels(radii).map((label, i) => (
        <span key={label}><i style={{ background: BAND_COLOR[i], opacity: 0.4 + GRID_OPACITY }} />{label}</span>
      ))}
      <strong>읍면동</strong>
      {Object.entries(dongStatusLabels(radii, minPop)).map(([k, label]) => (
        <span key={k}><i className="dot" style={{ background: STATUS_COLOR[k] }} />{label}</span>
      ))}
      <span>원 크기: 대상 인구</span>
      <strong>시설 사용률</strong>
      <span className="legend-row">
        {[["low", "70% 미만"], ["mid", "70~130%"], ["high", "130% 초과"]].map(([k, label]) => (
          <span key={k}><i className="dot" style={{ background: color, boxShadow: `0 0 0 ${USAGE_STROKE[k].weight}px ${USAGE_STROKE[k].color}` }} />{label}</span>
        ))}
      </span>
      <span><i className="ring" />유입 읍면동</span>
      </div>
    </details>
  );
}

/**
 * 지도 위 가운데 위쪽 "변화" 띠 한 줄(처음 상태와 다를 때만, change = null이면 없음): 분석 기준 평균 사용률(처음 대비 %p, 기준이 없으면 칸 없음),
 * 정주인구(증감분 + 정주 유입), 생활인구(0이면 칸 없음). 이름·값·단위만.
 */
function ChangeStrip({ change }) {
  if (!change) return null;
  const pp = change.rate != null && change.baseRate != null ? Math.round(change.rate * 100) - Math.round(change.baseRate * 100) : null;
  return (
    <div className="change-strip" role="status" aria-label="변화">
      <div>
        <strong>변화</strong>
        {change.rate != null && (
          <span>평균 사용률 <b>{Math.round(change.rate * 100)}%</b>{pp ? <em className={pp > 0 ? "up" : "down"}>{pp > 0 ? "+" : "−"}{Math.abs(pp)}%p</em> : null}</span>
        )}
        <span>정주인구 <b>{signedPeople(change.settled)}</b></span>
        {Math.round(change.living) !== 0 && <span>생활인구 <b>{signedPeople(change.living)}</b></span>}
      </div>
    </div>
  );
}

// 오른쪽 위 겹침 묶음(.map-controls 폭 340px + 오른쪽 여백 12px)이 가리는 폭. 지도가 좁으면 절반까지만 비킨다.
const OVERLAY_PX = 352;
const overlayWidth = (map) => Math.min(OVERLAY_PX, map.getSize().x / 2);

// 목록에서 고른 곳으로 옮길 때 시도할 가장 큰 확대 단계(여기부터 0.5씩 줄임)
const FOCUS_MAX_ZOOM = 12;

function MapSync({ bounds, focus, selPoint, circles, legendOpen }) {
  const map = useMap();
  // 지도 위에 겹친 요소의 자리(지도 컨테이너 px): 범례(접혀 있으면 제목 한 줄 높이), "변화" 띠(보일 때만)
  const boxOf = (selector) => {
    const r = map.getContainer().parentElement?.querySelector(selector)?.getBoundingClientRect();
    const origin = map.getContainer().getBoundingClientRect();
    return r ? { x: r.left - origin.left, y: r.top - origin.top, w: r.width, h: r.height } : null;
  };
  const legendBox = () => boxOf(".legend");
  const stripBox = () => boxOf(".change-strip");
  // 지도 모서리의 라이브러리 표기는 끄고, 배경지도 출처 표기(OpenStreetMap 이용 조건)만 남긴다.
  useEffect(() => { map.attributionControl?.setPrefix(false); }, [map]);
  // 처음 범위 맞춤은 오른쪽 위 묶음 폭과 왼쪽 아래 범례 높이만큼 비켜서 지역의 원·경계선이 가려지지 않게 한다.
  // 처음 범위 맞춤 여백: 오른쪽 위 묶음 폭, 왼쪽 아래 범례 높이(범례 위쪽 끝까지, 지도가 낮으면 지도 높이의 절반까지만),
  // 위쪽 "변화" 띠가 보이면 띠 아래 끝까지
  const fitPadding = () => {
    const legend = map.getContainer().parentElement?.querySelector(".legend");
    const strip = stripBox();
    const height = map.getSize().y;
    const legendBottom = legend ? Math.min(height - legend.offsetTop + 8, height / 2) : 24;
    const top = strip ? strip.y + strip.h + 8 : 24;
    return { paddingTopLeft: [24, top], paddingBottomRight: [24 + overlayWidth(map), Math.max(24, legendBottom)] };
  };
  const fitRegion = (animate) => bounds && map.flyToBounds([[bounds.south, bounds.west], [bounds.north, bounds.east]], { ...fitPadding(), animate, duration: 0.6 });
  // 처음 범위 맞춤은 지역의 원·경계선이 오른쪽 위 묶음과 범례에 가려지지 않게 한다.
  useEffect(() => {
    if (!bounds) return;
    map.fitBounds([[bounds.south, bounds.west], [bounds.north, bounds.east]], fitPadding());
  }, [bounds, map]);
  /*
   * 목록에서 고른 곳으로 옮긴다. 확대 단계는 12부터 0.5씩 줄여 가며, 고른 곳을 둘 자리 후보마다
   * 어떤 읍면동 원도 범례에 걸리지 않는 첫 조합을 쓴다(labels.js focusZoom). 그런 조합이 없으면 처음 범위로 돌아간다.
   */
  useEffect(() => {
    if (!focus || !bounds) return;
    const lg = legendBox();
    // 원이 걸리면 안 되는 자리: 범례와 "변화" 띠
    const blocked = [lg, stripBox()].filter(Boolean);
    const size = map.getSize();
    // 지역 전체가 보이는 단계보다 더 줄이지는 않는다(작은 지역은 그 단계 하나만)
    const minZoom = map.getBoundsZoom([[bounds.south, bounds.west], [bounds.north, bounds.east]]);
    const zooms = [];
    for (let z = Math.max(FOCUS_MAX_ZOOM, minZoom); z >= minZoom; z -= 0.5) zooms.push(z);
    // 고른 곳을 둘 자리: 겹침 묶음을 뺀 왼쪽 영역 가운데 → 범례 오른쪽 → 범례 위 → 범례 오른쪽 위
    const free = size.x - overlayWidth(map);
    const b = lg ?? { x: 0, y: size.y, w: 0, h: 0 };
    const right = (b.x + b.w + free) / 2;
    const spots = [[free / 2, size.y / 2], [right, size.y / 2], [free / 2, b.y / 2], [right, b.y / 2]];
    const hit = focusZoom(focus, circles, { zooms, spots, project: (p, z) => map.project([p.lat, p.lng], z), blocked });
    if (!hit) { fitRegion(true); return; }
    // 고른 곳이 화면 (x, y)에 오게 지도 가운데를 정한다
    const center = map.project([focus.lat, focus.lng], hit.zoom).add([size.x / 2 - hit.x, size.y / 2 - hit.y]);
    map.flyTo(map.unproject(center, hit.zoom), hit.zoom, { duration: 0.6 });
  }, [focus, map]); // 원·범위는 고른 순간 값으로 충분해 focus가 바뀔 때만 옮긴다
  // 범례를 펼쳤을 때(분석이 시작되어 범례가 새로 보일 때 포함) 읍면동 원이 그 밑에 들면 지금 높이로 범위를 다시 맞춘다
  useEffect(() => {
    const b = legendOpen && legendBox();
    if (!b) return;
    const hidden = circles.some((c) => {
      const p = map.latLngToContainerPoint([c.lat, c.lng]);
      return p.x + c.r > b.x && p.x - c.r < b.x + b.w && p.y + c.r > b.y && p.y - c.r < b.y + b.h;
    });
    if (hidden) fitRegion(true);
  }, [legendOpen, map]); // 펼친 순간의 원 자리로 충분
  // 지도에서 누른 곳이 겹침 묶음 밑에 들면 왼쪽 영역 가운데로 옮긴다(목록에서 고른 곳은 위에서 처리).
  useEffect(() => {
    if (!selPoint || (focus && focus.lat === selPoint.lat && focus.lng === selPoint.lng)) return;
    const freeWidth = map.getSize().x - overlayWidth(map);
    const p = map.latLngToContainerPoint([selPoint.lat, selPoint.lng]);
    if (p.x > freeWidth - 24) map.panBy([p.x - freeWidth / 2, 0]);
  }, [selPoint, focus, map]);
  return null;
}

export default function MapView(props) {
  const { bounds, boundaries = [], grid, units, facilities, visible, activeType, typeById, envOf, summary, env, selection } = props;
  const boundaryData = useMemo(
    () => ({ type: "FeatureCollection", features: boundaries.map((b) => ({ type: "Feature", properties: { code: b.code }, geometry: b.geometry })) }),
    [boundaries]
  );
  const regionCodes = [...new Set(boundaries.map((b) => b.code.slice(0, 5)))];
  const regionIndex = Object.fromEntries(regionCodes.map((c, i) => [c, i]));
  const regionCount = regionCodes.length;
  const regionKey = `${regionCodes.join(",")}-${boundaries.length}`;
  const [legendOpen, setLegendOpen] = useState(true);
  const { onSelectCell, onSelectDong, onSelectFacility, focus, adjusting, settled, living, usage } = props;
  // 기준 거리: 칸·시설은 그 시군구 값(시·군 시작값이 섞인 동안만 다름, spatial.js radiusIn)
  const rAt = (code) => radiusIn(env, code);
  // 원 크기·색 기준은 증감률 0일 때 최댓값이라, 증감률을 올리면 원이 커지고(기준의 2배까지) 내리면 작아진다.
  const baseMax = Math.max(props.baseMax, 1);
  const sizeOf = (pop) => 7 + 13 * Math.sqrt(Math.min(pop / baseMax, 2));
  const selectedFacility = selection?.kind === "facility" ? facilities.find((f) => f.id === selection.id) : null;
  // 연결선은 추가·전환으로 생긴 시설을 골랐을 때만
  const selectedAction = selectedFacility?.actionId ? selectedFacility : null;
  const selPoint = selectedFacility
    ?? (selection?.kind === "dong" ? units.find((u) => u.id === selection.id && u.lat != null) : null)
    ?? (selection?.kind === "cell" ? grid.find((c) => c.id === selection.id) : null);
  // 시설 점: 표시 체크가 켜진 유형 + 분석 기준의 공급 시설(체크와 관계없이 늘, scenario.js mapFacilities)
  const shown = mapFacilities(facilities, visible, activeType);
  const basisIds = new Set(basisTypeIds(activeType));
  // 분석 기준이 있을 때만 분석 그림(읍면동 원·격자 색·사용률 테두리·반경 원·유입 고리·숫자·범례). 없으면 경계와 시설 점만,
  // 격자는 투명하게 남겨 지점을 눌러 추가할 수 있게 한다.
  const analyzing = summary != null;

  return (
    <section className="map-wrap" aria-label="공간정보 지도">
      {/* 범위 맞춤이 정수 확대 단계에 걸려 지역이 작게 보이지 않도록 0.25 단계까지 허용(확대·축소 버튼은 1단계) */}
      <MapContainer center={COUNTRY_VIEW.center} zoom={COUNTRY_VIEW.zoom} zoomSnap={0.25} className="map" zoomControl>
        <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" errorTileUrl={BLANK_TILE} attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors' />
        {grid.map((c) => {
          const selected = selection?.kind === "cell" && selection.id === c.id;
          return (
            <Rectangle
              key={c.id}
              bounds={c.bounds}
              pathOptions={{ fillColor: BAND_COLOR[bandOf(c.dist, rAt(c.region))], fillOpacity: analyzing ? GRID_OPACITY : 0, color: selected ? "#111" : "#ffffff", weight: selected ? 3 : analyzing ? 0.3 : 0, opacity: selected ? 1 : 0.5 }}
              eventHandlers={{ click: () => onSelectCell(c.id) }}
            />
          );
        })}
        {boundaries.length > 0 && <GeoJSON key={regionKey} data={boundaryData} style={(f) => ({ ...BOUNDARY_STYLE, color: REGION_LINE[regionIndex[f.properties.code.slice(0, 5)] % REGION_LINE.length], opacity: regionCount > 1 ? 0.8 : 0.6 })} interactive={false} />}
        {analyzing && adjusting === "radius" && facilities.filter((f) => basisIds.has(f.typeId)).map((f) => (
          <Circle key={`r-${f.id}`} center={[f.lat, f.lng]} radius={rAt(regionOf(f.dong)) * 1000} pathOptions={radiusStyle(activeType.color, true)} interactive={false} />
        ))}
        {analyzing && selectedFacility && (
          <Circle
            center={[selectedFacility.lat, selectedFacility.lng]}
            radius={radiusIn(envOf(typeById(selectedFacility.typeId)), regionOf(selectedFacility.dong)) * 1000}
            pathOptions={radiusStyle(typeById(selectedFacility.typeId).color, false)}
            interactive={false}
          />
        )}
        {analyzing && units.map((u, i) => {
          if (u.lat == null) return null;
          const selected = selection?.kind === "dong" && selection.id === u.id;
          // 인구 조절 중에는 최소 대상 인구 미만인 원을 흐리게
          const faint = adjusting === "pop" && summary.pops[i] < env.minPop;
          return (
            <CircleMarker
              key={u.id}
              center={[u.lat, u.lng]}
              radius={sizeOf(summary.pops[i])}
              pathOptions={{
                fillColor: STATUS_COLOR[dongStatus(summary, i, env)],
                fillOpacity: faint ? 0.2 : 0.75,
                color: selected ? "#111" : "#ffffff",
                opacity: faint ? 0.4 : 1,
                weight: selected ? 3.5 : 1.5
              }}
              eventHandlers={{ click: () => onSelectDong(u.id) }}
            >
              <Tooltip direction="top" offset={[0, -8]}>{u.name}</Tooltip>
            </CircleMarker>
          );
        })}
        {shown.map((f) => {
          const t = typeById(f.typeId);
          const isActive = basisIds.has(f.typeId);
          const isSelected = selectedFacility?.id === f.id;
          // ◉ 유형 기존 시설은 테두리로 사용률 3단계, 추가·전환 시설은 흰 바탕 굵은 테두리
          const level = isActive && !f.added ? usageLevel(usage[f.id]?.rate) : null;
          const stroke = level ? USAGE_STROKE[level] : { color: t.color, weight: f.added ? 3 : 1.2 };
          return (
            <CircleMarker
              key={f.id}
              center={[f.lat, f.lng]}
              radius={f.added ? 7 : isActive ? 4.5 : 3}
              pathOptions={{
                color: isSelected ? "#111" : stroke.color,
                weight: isSelected ? 3 : stroke.weight,
                fillColor: f.added ? "#ffffff" : t.color,
                fillOpacity: isActive || f.added ? 0.95 : 0.5
              }}
              eventHandlers={{ click: () => onSelectFacility(f.id) }}
            >
              <Tooltip direction="top" offset={[0, -6]}>{f.name}</Tooltip>
            </CircleMarker>
          );
        })}
        {analyzing && (
          <ScenarioLayer
            units={units} pops={summary.pops} sizeOf={sizeOf} settled={settled} living={living} usage={usage} selected={selectedAction}
          />
        )}
        {analyzing && adjusting === "pop" && <PopLabels units={units} pops={summary.pops} sizeOf={sizeOf} />}
        <MapSync
          bounds={bounds} focus={focus} selPoint={selPoint}
          circles={analyzing ? units.flatMap((u, i) => (u.lat == null ? [] : [{ lat: u.lat, lng: u.lng, r: sizeOf(summary.pops[i]) }])) : []}
          legendOpen={analyzing && legendOpen}
        />
      </MapContainer>
      {/* 범례 항목은 모두 분석 그림의 뜻이라 기준이 없으면 범례 전체를 숨긴다(접기·펴기 상태는 남김) */}
      {analyzing && <Legend radii={radiusList(env)} minPop={env.minPop} color={activeType.color} open={legendOpen} onToggle={setLegendOpen} />}
      <ChangeStrip change={props.change} />
      {props.children}
    </section>
  );
}
