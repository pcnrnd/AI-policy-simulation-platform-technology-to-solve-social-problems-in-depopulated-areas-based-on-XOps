import { useEffect, useId, useMemo, useRef, useState } from "react";

// 한 번에 고를 수 있는 시군구 수 — 화면이 느려지지 않게. 1347×697에서 ◉ 바꾸기가 1곳 0.1초·3곳 0.25초·5곳 0.6초·8곳 0.7초,
// 지역 더하기가 5곳까지 0.75초 안·8곳 0.8초라 ◉ 0.6초 안인 5곳으로 둔다(API 상한은 8). 다 고르면 나머지 칸은 더 고를 수 없다.
export const MAX_REGIONS = 5;
// 검색 결과로 보이는 최대 줄 수
const MAX_ROWS = 60;

/**
 * 왼쪽 맨 위 요약 상자의 지역 선택: 고른 시군구 이름(셋 넘으면 "외 n"), 펼치면 고른 지역 칩(빼기), 시도 고르기, 시군구 검색, 시군구 체크 목록.
 * 고르거나 빼는 즉시 onChange(코드 목록). 마지막 한 곳은 뺄 수 없다.
 */
export default function RegionPicker({ list, selected, onChange }) {
  const [sido, setSido] = useState("");
  const [query, setQuery] = useState("");
  const box = useRef(null);
  const ids = useId();
  const byCode = useMemo(() => Object.fromEntries(list.map((r) => [r.code, r])), [list]);
  const sidos = useMemo(() => [...new Set(list.map((r) => r.sido))], [list]);
  const chosen = new Set(selected);
  const q = query.trim().replace(/\s/g, "");
  const rows = list
    .filter((r) => (!sido || r.sido === sido) && (!q || `${r.sido}${r.name}`.replace(/\s/g, "").includes(q)))
    .slice(0, MAX_ROWS);
  const names = selected.map((c) => byCode[c]?.name ?? c);
  const label = names.length > 3 ? `${names.slice(0, 2).join(" · ")} 외 ${names.length - 2}` : names.join(" · ");
  const full = selected.length >= MAX_REGIONS;
  const toggle = (code, on) => onChange(on ? [...selected, code] : selected.filter((c) => c !== code));

  // 바깥을 누르거나 Esc면 닫는다
  useEffect(() => {
    const close = (e) => { if (box.current?.open && (e.key === "Escape" || (e.type === "pointerdown" && !box.current.contains(e.target)))) box.current.open = false; };
    document.addEventListener("pointerdown", close);
    document.addEventListener("keydown", close);
    return () => { document.removeEventListener("pointerdown", close); document.removeEventListener("keydown", close); };
  }, []);

  if (!list.length) return null;
  return (
    <details className="region-picker" ref={box}>
      <summary aria-label={`지역 ${names.join(", ")}`}>{label}</summary>
      <div className="region-panel">
        <div className="region-chips">
          {selected.map((c) => (
            <span key={c} className="chip">
              {byCode[c]?.sido} {byCode[c]?.name}
              {selected.length > 1 && <button type="button" aria-label={`${byCode[c]?.name} 빼기`} onClick={() => toggle(c, false)}>×</button>}
            </span>
          ))}
        </div>
        <div className="region-filter">
          <label htmlFor={`${ids}-sido`} className="sr-only">시도</label>
          <select id={`${ids}-sido`} value={sido} onChange={(e) => setSido(e.target.value)}>
            <option value="">전체 시도</option>
            {sidos.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
          <label htmlFor={`${ids}-q`} className="sr-only">시군구 검색</label>
          <input id={`${ids}-q`} type="search" placeholder="시군구 검색" value={query} onChange={(e) => setQuery(e.target.value)} />
        </div>
        <ul className="region-list">
          {rows.map((r) => {
            const on = chosen.has(r.code);
            return (
              <li key={r.code}>
                <label>
                  <input type="checkbox" checked={on} disabled={(on && selected.length === 1) || (!on && full)} onChange={(e) => toggle(r.code, e.target.checked)} />
                  <span>{r.name}</span><small>{r.sido}</small>
                </label>
              </li>
            );
          })}
        </ul>
      </div>
    </details>
  );
}
