import { useEffect, useRef, useState } from "react";
import L from "leaflet";
import PendingData from "./PendingData.jsx";
import { buildRegionGrid, densityColor, DENSITY_LEGEND } from "../lib/geo.js";
import { prefersReducedMotion } from "../lib/motion.js";

// 정책 시뮬레이터 STAGE ③ 공간정보 탐색 지도 — 플로팅 패널(검색·레이어·테마·범례)·출처·시설물/격자 상세.
// 두 모드 모두 같은 지도·컨트롤을 두고, 예시 오버레이는 데모 표시에서만 레이어에 올린다.

// 지도 베이스 테마 — 전부 VWorld(국토교통부 공간정보)로 통일해 모든 테마가 한글 지명 표기.
// 키가 있으면 공식 WMTS API(+회색 추가), 없으면 공개 타일 서버(xdworld)를 사용(데모용, 운영 시 인증키 권장).
// 공개 서버 동작 확인: Base·white·midnight(라이브 타일 테스트). gray는 키 필요, Satellite는 라벨 없어 제외.
const VWORLD_KEY = import.meta.env.VITE_VWORLD_KEY;
const vworldUrl = (layer) =>
  VWORLD_KEY
    ? `https://api.vworld.kr/req/wmts/1.0.0/${VWORLD_KEY}/${layer}/{z}/{y}/{x}.png`
    : `https://xdworld.vworld.kr/2d/${layer}/service/{z}/{x}/{y}.png`;

const MAP_THEMES = [
  { id: "vw-midnight", label: "다크", url: vworldUrl("midnight"), maxZoom: 19 },
  { id: "vw-base", label: "일반", url: vworldUrl("Base"), maxZoom: 19 },
  // 회색(gray)은 공개 타일 서버 미제공 — 인증키(공식 API)일 때만 노출
  ...(VWORLD_KEY ? [{ id: "vw-gray", label: "회색", url: vworldUrl("gray"), maxZoom: 19 }] : [])
];
// 기본: 야간 — 다크 톤이라 대시보드와 정합, 지명이 한글로 표기됨.
const DEFAULT_THEME_ID = "vw-midnight";

// 소멸위험지수(낮을수록 위험) → 지도·목록·팝업에서 함께 쓰는 등급/색상.
export const RISK_GRADES = [
  {
    label: "고위험",
    rangeLabel: "0.15 미만",
    maxExclusive: 0.15,
    color: "var(--accent-red)",
    markerColor: "#ef4444",
    rgb: "239, 68, 68"
  },
  {
    label: "주의",
    rangeLabel: "0.15 이상 0.18 미만",
    maxExclusive: 0.18,
    color: "var(--accent-orange)",
    markerColor: "#f59e0b",
    rgb: "245, 158, 11"
  },
  {
    label: "관찰",
    rangeLabel: "0.18 이상",
    maxExclusive: Number.POSITIVE_INFINITY,
    color: "var(--accent-teal)",
    markerColor: "#10b981",
    rgb: "16, 185, 129"
  }
];

export function riskGrade(riskIndex) {
  return RISK_GRADES.find((grade) => riskIndex < grade.maxExclusive) ?? RISK_GRADES.at(-1);
}

/**
 * @param {{ appData: object, currentRegion: object, setCurrentRegion: Function, allowSeed: boolean,
 *   searchText: string, onSearch: Function, overlayNote: string|null, resizeKey: string }} props
 * overlayNote: 예시 오버레이가 없을 때 플로팅 패널 안내. resizeKey: 바뀌면 지도 크기를 다시 계산한다.
 */
export default function SimulatorMap({ appData, currentRegion, setCurrentRegion, allowSeed, searchText, onSearch, overlayNote, resizeKey }) {
  const [layerVis, setLayerVis] = useState({ markers: true, grid: true, facilities: true });
  // 지도 베이스 테마 (한글 지명을 제공하는 VWorld 타일)
  const [mapTheme, setMapTheme] = useState(DEFAULT_THEME_ID);

  const mapContainerRef = useRef(null);
  const mapRef = useRef(null);
  const tileLayerRef = useRef(null);
  const markerLayerRef = useRef(null);
  const gridLayerRef = useRef(null);
  const facilityLayerRef = useRef(null);

  // 지도 1회 초기화 — 줌 컨트롤은 우하단, 레이어 토글은 커스텀 플로팅 패널이 담당
  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return undefined;

    // 기본 attribution 컨트롤 비활성 — 출처는 좌하단 ⓘ 접이식 패널(.map-attribution)이 담당 (OSM 공간 제약 표기 방식)
    const map = L.map(mapContainerRef.current, { zoomControl: false, attributionControl: false }).setView(
      [35.1, 126.9],
      8
    );
    L.control.zoom({ position: "bottomright" }).addTo(map);

    // 베이스 타일은 mapTheme 전환 effect가 생성·교체한다 (여기선 미생성)

    const markerLayer = L.layerGroup();
    const gridLayer = L.layerGroup();
    const facilityLayer = L.layerGroup();

    markerLayer.addTo(map);
    gridLayer.addTo(map);
    facilityLayer.addTo(map);
    markerLayerRef.current = markerLayer;
    gridLayerRef.current = gridLayer;
    facilityLayerRef.current = facilityLayer;
    mapRef.current = map;

    setTimeout(() => map.invalidateSize(), 100);

    return () => {
      map.remove();
      mapRef.current = null;
      tileLayerRef.current = null;
      markerLayerRef.current = null;
      gridLayerRef.current = null;
      facilityLayerRef.current = null;
    };
    // 지도 인스턴스는 마운트 1회만 만든다 — 피처는 아래 적재 effect가 따로 채운다.
    // (지도를 다시 만들면 베이스 타일·레이어 토글 상태를 다시 맞춰야 해서 깨지기 쉽다.)
  }, []);

  // 오버레이 피처 적재 — 데모 OFF에서는 예시 마커·격자·시설물을 레이어에 **넣지 않는다**
  // (예전엔 CSS로 leaflet pane을 통째로 가렸다 — 그러면 왜 비었는지 읽히지 않고 팝업까지 죽는다).
  // 레이어 그룹과 지도 인스턴스는 그대로라 표시 레이어 토글·줌·테마는 계속 조작 가능하다.
  useEffect(() => {
    const markerLayer = markerLayerRef.current;
    const gridLayer = gridLayerRef.current;
    const facilityLayer = facilityLayerRef.current;
    if (!markerLayer || !gridLayer || !facilityLayer) return;

    markerLayer.clearLayers();
    gridLayer.clearLayers();
    facilityLayer.clearLayers();
    if (!allowSeed) return;

    // 시설물 공간정보 표시 — 정주여건·스마트팜 시설의 위치와 데이터를 표출 (레이어 적층)
    const FACILITY_COLORS = {
      스마트팜: "#10b981",
      임대주택: "#3b82f6",
      신재생: "#f59e0b",
      정원수: "#22d3ee"
    };
    (appData.facilities ?? []).forEach((f) => {
      const fColor = FACILITY_COLORS[f.type] ?? "#94a3b8";
      const fMarker = L.circleMarker([f.lat, f.lng], {
        radius: 6,
        fillColor: fColor,
        color: "#0b1220",
        weight: 1,
        opacity: 1,
        fillOpacity: 0.9
      });
      fMarker.bindPopup(`
        <div style="font-family: 'Inter', sans-serif;">
          <h4 style="margin-bottom:6px; color:var(--text-primary);"><span aria-hidden="true" style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${fColor};margin-right:6px;"></span>${f.name}</h4>
          <p style="font-size:12px; margin: 2px 0;">시설 유형: <strong>${f.type}</strong></p>
          <p style="font-size:12px; margin: 2px 0;">${f.metric}</p>
          <p style="font-size:12px; margin: 2px 0;">운영 예산: <strong>${f.operating_budget}억</strong></p>
        </div>
      `);
      fMarker.addTo(facilityLayer);
    });

    appData.regions.forEach((region) => {
      const grade = riskGrade(region.riskIndex);
      const color = grade.markerColor;
      const marker = L.circleMarker([region.lat, region.lng], {
        radius: 10,
        fillColor: color,
        color: "#ffffff",
        weight: 1,
        opacity: 1,
        fillOpacity: 0.8
      });
      marker.bindPopup(`
        <div style="font-family: 'Inter', sans-serif;">
          <h4 style="margin-bottom:6px; color:var(--text-primary);"><span aria-hidden="true" style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${color};margin-right:6px;"></span>${region.name}</h4>
          <p style="font-size:12px; margin: 2px 0;">인구수: <strong>${region.population.toLocaleString()}명</strong></p>
          <p style="font-size:12px; margin: 2px 0;">위험지수: <strong>${region.riskIndex}</strong> (${grade.label})</p>
          <p style="font-size:11px; margin-top:6px; color: var(--text-muted);">우측 패널에서 정책 변수를 설정하세요.</p>
        </div>
      `);
      marker.on("click", () => setCurrentRegion(region));
      marker.addTo(markerLayer);

      buildRegionGrid(region).forEach((cell) => {
        const rect = L.rectangle(cell.bounds, {
          color: "#0b1220",
          weight: 0.5,
          fillColor: densityColor(cell.density),
          fillOpacity: 0.55
        });
        rect.bindPopup(
          `<div style="font-family:'Inter',sans-serif;font-size:12px;">` +
            `격자 ${cell.gridId}<br/>㎢당 인구밀도: <strong>${cell.density}</strong></div>`
        );
        rect.addTo(gridLayer);
      });
    });

  }, [appData, setCurrentRegion, allowSeed]);

  // 베이스 테마 전환 — 기존 타일 제거 후 신규 타일 추가(오버레이는 별도 pane이라 위 유지)
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const def = MAP_THEMES.find((t) => t.id === mapTheme) ?? MAP_THEMES[0];
    if (tileLayerRef.current) map.removeLayer(tileLayerRef.current);
    // subdomains에 undefined를 넘기면 Leaflet 기본값('abc')을 덮어써 크래시 — 미정의 시 키 자체를 생략
    const tileOpts = { maxZoom: def.maxZoom };
    if (def.subdomains) tileOpts.subdomains = def.subdomains;
    const tile = L.tileLayer(def.url, tileOpts);
    tile.addTo(map);
    tile.bringToBack();
    tileLayerRef.current = tile;
  }, [mapTheme]);

  // 레이어 토글 반영
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const sync = (layer, visible) => {
      if (!layer) return;
      if (visible) map.addLayer(layer);
      else map.removeLayer(layer);
    };
    sync(markerLayerRef.current, layerVis.markers);
    sync(gridLayerRef.current, layerVis.grid);
    sync(facilityLayerRef.current, layerVis.facilities);
  }, [layerVis]);

  // 선택 지자체로 지도 이동
  useEffect(() => {
    if (mapRef.current) {
      mapRef.current.panTo([currentRegion.lat, currentRegion.lng], {
        animate: !prefersReducedMotion()
      });
    }
  }, [currentRegion]);

  // STAGE③ 재펼침/잠금해제 시 지도 크기 재계산 (display:none·blur 해제 후 타일 깨짐 방지)
  useEffect(() => {
    if (!mapRef.current) return undefined;
    const t = setTimeout(() => mapRef.current?.invalidateSize(), 200);
    return () => clearTimeout(t);
  }, [resizeKey]);

  return (
    <div className="sim-map-wrap" data-slot="sim.map">
      <div
        id="map"
        ref={mapContainerRef}
        role="region"
        aria-label="VWorld 지자체 소멸위험·인구밀도·시설물 지도"
        aria-describedby="map-accessible-summary"
      ></div>
      <p id="map-accessible-summary" className="sr-only mock-data-output">
        {allowSeed
          ? `현재 선택 지역은 ${currentRegion.name}, 위험지수 ${currentRegion.riskIndex}, ${riskGrade(currentRegion.riskIndex).label}입니다.`
          : `현재 선택 지역은 ${currentRegion.name}입니다. ${overlayNote ?? ""}`}
        위험등급 마커 {layerVis.markers ? "표시" : "숨김"}, 인구밀도 격자 {layerVis.grid ? "표시" : "숨김"},
        시설물 {layerVis.facilities ? "표시" : "숨김"} 상태입니다. 지역은 오른쪽 지자체 목록에서 키보드로 선택할 수 있습니다.
      </p>
      {/* 플로팅: 검색 + 레이어 + 범례 */}
      <div className="map-float map-float-tl">
        <div className="map-float-search">
          <i className="fa-solid fa-magnifying-glass" aria-hidden="true"></i>
          <input
            type="text"
            placeholder="지자체 검색"
            value={searchText}
            onChange={(e) => onSearch(e.target.value)}
            aria-label="지자체 검색"
          />
        </div>
        <div className="map-float-section-label">표시 레이어</div>
        {overlayNote && <p className="map-theme-note">{overlayNote}</p>}
        <label className="map-float-toggle">
          <input
            type="checkbox"
            checked={layerVis.markers}
            onChange={(e) => setLayerVis((v) => ({ ...v, markers: e.target.checked }))}
          />
          위험등급 마커
        </label>
        <label className="map-float-toggle">
          <input
            type="checkbox"
            checked={layerVis.grid}
            onChange={(e) => setLayerVis((v) => ({ ...v, grid: e.target.checked }))}
          />
          인구밀도 격자
        </label>
        <label className="map-float-toggle">
          <input
            type="checkbox"
            checked={layerVis.facilities}
            onChange={(e) => setLayerVis((v) => ({ ...v, facilities: e.target.checked }))}
          />
          시설물 (정주여건·스마트팜)
        </label>
        <div className="map-float-section-label">지도 테마</div>
        <div className="map-theme-switch">
          {MAP_THEMES.map((t) => (
            <button
              key={t.id}
              type="button"
              className={"map-theme-btn" + (mapTheme === t.id ? " active" : "")}
              aria-pressed={mapTheme === t.id}
              onClick={() => setMapTheme(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>
        {!VWORLD_KEY && (
          <p className="map-theme-note">
            모든 테마 한글 지명(VWorld 공개 타일) — 운영 배포 시 <code>VITE_VWORLD_KEY</code>(인증키)
            권장, 회색 테마 추가됨.
          </p>
        )}

        <div className="map-float-section-label">소멸위험지수</div>
        <div className="map-float-legend" aria-label="소멸위험지수 등급 범례">
          {RISK_GRADES.map((grade) => (
            <span key={grade.label} className="legend-item">
              <span className="legend-swatch" style={{ backgroundColor: grade.markerColor }} aria-hidden="true"></span>
              {grade.label} ({grade.rangeLabel})
            </span>
          ))}
        </div>

        <div className="map-float-section-label">인구밀도(㎢당)</div>
        <div className="map-float-legend" aria-label="인구밀도 범례">
          {DENSITY_LEGEND.map((d) => (
            <span key={d.label} className="legend-item">
              <span className="legend-swatch" style={{ backgroundColor: d.color }} aria-hidden="true"></span>
              {d.label}
            </span>
          ))}
        </div>
      </div>

      {/* 지도 출처 — 접이식 표기(ⓘ hover/포커스 시 펼침). 활성 타일 소스별 표기 의무 준수 */}
      <div className="map-attribution" tabIndex={0} aria-label="지도 출처 정보">
        <i className="fa-solid fa-circle-info" aria-hidden="true"></i>
        <span className="map-attribution-text">
          ©{" "}
          <a href="https://www.vworld.kr" target="_blank" rel="noreferrer">
            VWorld
          </a>{" "}
          · 국토교통부
        </span>
      </div>
    </div>
  );
}

/** 지도 시설물·선택 지역 격자 상세 — 데모 표시에서만 목록, 그 밖에는 같은 자리 안내. */
export function SimulatorMapDetails({ appData, currentRegion, allowSeed, note }) {
  return (
    <details className="map-data-details" data-slot="sim.map.details">
      <summary>지도 시설물·선택 지역 격자 정보</summary>
      <div className="map-data-grid">
        <section>
          <h4>시설물</h4>
          {allowSeed ? (
            <ul>
              {(appData.facilities ?? []).map((facility) => (
                <li key={`${facility.name}-${facility.lat}-${facility.lng}`}>
                  <strong>{facility.name}</strong> — {facility.type}, {facility.metric}, 운영 예산 {facility.operating_budget}억
                </li>
              ))}
            </ul>
          ) : (
            <PendingData text={note} />
          )}
        </section>
        <section>
          <h4>{currentRegion.name} 인구밀도 격자</h4>
          {allowSeed ? (
            <ul>
              {buildRegionGrid(currentRegion).map((cell) => (
                <li key={cell.gridId}>격자 {cell.gridId} — ㎢당 {cell.density}명</li>
              ))}
            </ul>
          ) : (
            <PendingData text={note} />
          )}
        </section>
      </div>
    </details>
  );
}
