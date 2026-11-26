import { useEffect, useId, useRef, useState } from "react";
import { ALL_BANDS } from "../lib/ingu.js";
import { popText } from "../lib/spatial.js";

// 터치로 조절한 뒤 손을 떼도 잠깐 그림을 남겨 둔다(ms)
const TOUCH_HOLD_MS = 800;

/**
 * 항목 이름 옆 회색 "?" 설명(WCAG 1.4.13). 포인터를 올리거나 키보드로 고르면 열리고, 설명 위로 포인터를 옮겨도 남으며,
 * Esc·포커스 이탈·밖 누르기로 닫힌다. 터치는 누를 때마다 열고 닫는다. 카드가 화면 오른쪽 끝이라 "?" 아래 왼쪽으로 연다.
 */
function Help({ id, label, text }) {
  const [open, setOpen] = useState(false);
  const wrap = useRef(null);
  const pointer = useRef("mouse");
  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => { if (e.key === "Escape") setOpen(false); };
    const onDown = (e) => { if (!wrap.current?.contains(e.target)) setOpen(false); };
    document.addEventListener("keydown", onKey);
    document.addEventListener("pointerdown", onDown);
    return () => { document.removeEventListener("keydown", onKey); document.removeEventListener("pointerdown", onDown); };
  }, [open]);
  const hover = (on) => (e) => { if (e.pointerType !== "touch") setOpen(on); };
  return (
    <span className="help-wrap" ref={wrap} onPointerEnter={hover(true)} onPointerLeave={hover(false)}>
      <button
        type="button" className="help" aria-label={`${label} 설명`} aria-expanded={open}
        onPointerDown={(e) => { pointer.current = e.pointerType; }}
        onClick={() => setOpen((o) => (pointer.current === "touch" ? !o : true))}
        onFocus={(e) => { if (e.target.matches(":focus-visible")) setOpen(true); }}
        onBlur={() => setOpen(false)}
      >?</button>
      <span role="tooltip" id={id} className="help-tip" hidden={!open}>{text}</span>
    </span>
  );
}

// 이름·슬라이더·값 한 줄(증감 줄과 같은 모양)
function Slider({ label, describedBy, text, value, min, max, step, onChange, adjust }) {
  const id = useId();
  return (
    <div className="band-row slider-inline">
      <label htmlFor={id}>{label}</label>
      <input id={id} type="range" min={min} max={max} step={step} value={value} aria-describedby={describedBy} onChange={(e) => onChange(Number(e.target.value))} {...adjust} />
      <output htmlFor={id}>{text}</output>
    </div>
  );
}

const signedPct = (v) => `${v > 0 ? "+" : ""}${v}%`;

// 연령 구간 증감 한 줄: 이름·슬라이더·값(−50~+50%, 5 간격). ◉ 시설 대상 구간이 아니면 흐리게(조절은 됨)
function BandRow({ band, value, onChange, adjust, describedBy, faint }) {
  const id = useId();
  return (
    <div className={`band-row${faint ? " faint" : ""}`}>
      <label htmlFor={id}>{band.label}</label>
      <input id={id} type="range" min={-50} max={50} step={5} value={value} aria-describedby={describedBy} onChange={(e) => onChange(Number(e.target.value))} {...adjust} />
      <output htmlFor={id}>{signedPct(value)}</output>
    </div>
  );
}

/**
 * 슬라이더를 조작하는 동안(포인터로 잡고 있을 때, 키보드 포커스가 있을 때)만 onAdjust(kind), 끝나면 onAdjust(null).
 * 지도는 이 값으로 거리 → 시설 반경 원, 인구 → 읍면동 숫자·미만 흐리게를 그린다.
 */
function useAdjust(onAdjust) {
  const timer = useRef(null);
  const holding = useRef(false);
  const show = (kind) => { clearTimeout(timer.current); onAdjust(kind); };
  const hide = (ms = 0) => { clearTimeout(timer.current); timer.current = setTimeout(() => onAdjust(null), ms); };
  return (kind) => ({
    onPointerDown: (e) => {
      holding.current = true;
      show(kind);
      const ac = new AbortController();
      const end = () => { ac.abort(); holding.current = false; hide(e.pointerType === "touch" ? TOUCH_HOLD_MS : 0); };
      window.addEventListener("pointerup", end, { signal: ac.signal });
      window.addEventListener("pointercancel", end, { signal: ac.signal });
    },
    onFocus: (e) => { if (e.target.matches(":focus-visible")) show(kind); },
    onKeyDown: () => show(kind),
    // 다른 슬라이더를 잡아 포커스가 옮겨 갈 때는 새 슬라이더 쪽 그림을 지우지 않는다
    onBlur: () => { if (!holding.current) hide(); }
  });
}

// 지도 오른쪽 위에 겹쳐 놓는 시뮬레이션 환경변수 카드·선택 상세(children). 지도 컨테이너 밖 형제 요소라 누르기·끌기·휠이 지도로 가지 않는다.
// "부족 판정" 묶음(거리·대상 인구 기준 두 줄)과 "인구 증감" 묶음(9구간, ◉ 시설과 관계없이 늘 9줄, 대상 구간은 진하게)을 두고,
// 값은 범례·요약과 같은 표기(km·명·%)로 보인다. 이 카드는 줄지 않고(? 설명이 카드 밖으로 열림), 아래 선택 상세가 남은 높이에서 스크롤한다.
// radiusLabel = 거리 값 표기(시·군 시작값이 섞이고 아직 움직이지 않았으면 "3·5km").
export default function MapControls(props) {
  // radiusKey = 거리 값을 둘 자리(분석 기준 유형 id, 기준이 없으면 아직 바꾸지 않은 유형 공통 자리)
  const { radiusKey, bands, radiusKm, radiusLabel, setRadius, growth, setGrowth, minPop, setMinPop, popScale, onAdjust } = props;
  const adjust = useAdjust(onAdjust);
  const ids = useId();
  const target = new Set(bands.map((b) => b.id)); // ◉ 시설 대상 구간(진하게)
  return (
    <div className="map-controls">
      <details className="env-card" open>
        <summary>시뮬레이션 환경변수</summary>
        <fieldset className="env-group">
          <legend>
            부족 판정
            <Help id={`${ids}-short`} label="부족 판정" text="가장 가까운 시설이 이 거리보다 멀고, 대상 인구가 이 수 이상인 읍면동을 부족 지역으로 봅니다." />
          </legend>
          <Slider
            label="거리" text={`${radiusLabel} 초과`} describedBy={`${ids}-short`} inline
            value={radiusKm} min={0.3} max={5} step={0.1} adjust={adjust("radius")}
            onChange={(v) => setRadius((prev) => ({ ...prev, [radiusKey]: v }))}
          />
          <Slider
            label="대상 인구" text={`${popText(minPop)} 이상`} describedBy={`${ids}-short`} inline
            value={minPop} min={0} max={popScale.max} step={popScale.step} adjust={adjust("pop")} onChange={setMinPop}
          />
        </fieldset>
        <fieldset className="env-group">
          <legend>
            인구 증감
            <Help id={`${ids}-grow`} label="인구 증감" text="연령 구간마다 인구를 늘리거나 줄여 보는 가정입니다. 진한 줄이 선택한 시설의 대상 연령입니다. 거리는 바뀌지 않습니다." />
          </legend>
          <div className="bands">
            {ALL_BANDS.map((b) => (
              <BandRow
                key={b.id} band={b} value={growth[b.id] ?? 0} describedBy={`${ids}-grow`} adjust={adjust("pop")} faint={!target.has(b.id)}
                onChange={(v) => setGrowth((prev) => ({ ...prev, [b.id]: v }))}
              />
            ))}
          </div>
        </fieldset>
      </details>
      {props.children}
    </div>
  );
}
