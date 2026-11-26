import { useEffect, useRef, useState } from "react";
import { TARGET_AGE_LABEL } from "../lib/ingu.js";
import { groupBasis } from "../lib/scenario.js";
import { radiusText } from "../lib/spatial.js";
import { fmt, km } from "./format.js";
import RegionPicker from "./RegionPicker.jsx";
import { ScenarioTab } from "./ScenarioTabs.jsx";

const signed = (d, text) => `${d > 0 ? "+" : "−"}${text(Math.abs(d))}`;

/** 숫자 하나: 지금 값, 처음(동작·증감 0, 지금 거리·기준)과 화면 값이 다르면 아래에 "처음 {값} ±차이"(늘면 up·줄면 down 색). */
function Kpi({ label, now, base, text, scale = 1 }) {
  // 판정·차이 모두 화면에 보이는 자리수(scale: 명·곳 1, km 소수 둘째 자리 100)로 — 보이는 두 값의 차와 같게
  const d = (Math.round(now * scale) - Math.round(base * scale)) / scale;
  return (
    <div>
      <span>{label}</span>
      <strong>{text(now)}</strong>
      {d !== 0 && <small><span className="was">처음 {text(base)}</span><b className={d > 0 ? "up" : "down"}>{signed(d, text)}</b></small>}
    </div>
  );
}

/**
 * 왼쪽 맨 위 요약: 지역 선택(고른 시군구) → 분석 기준(그룹·유형) 이름·대상 연령(표시만, 고르기는 "분석" 탭) →
 * 숫자 3개(처음과 다르면 처음·차이). 정주·생활인구 변화는 지도 위 "변화" 띠에. 분석 기준이 없으면 지역과 안내 한 줄만.
 */
function SummaryCard({ type, summary, base, radiusLabel, hasPop, picker }) {
  if (!type) {
    return (
      <section className="summary">
        {picker}
        <p className="eyebrow">분석 탭에서 기준을 고르세요</p>
      </section>
    );
  }
  return (
    <section className="summary">
      {picker}
      <p className="eyebrow">{type.label} · {TARGET_AGE_LABEL[type.target]}</p>
      {hasPop ? (
        <>
          <div className="kpis">
            <Kpi label={`${radiusLabel} 밖 대상 인구`} now={summary.total - summary.covered} base={base.summary.total - base.summary.covered} text={(v) => `${fmt(v)}명`} />
            <Kpi label="부족 읍면동" now={summary.shortage.length} base={base.summary.shortage.length} text={(v) => `${v}`} />
            <Kpi label="평균 거리" now={summary.meanDistKm ?? 0} base={base.summary.meanDistKm ?? 0} text={km} scale={100} />
          </div>
        </>
      ) : <p>자료를 불러오지 못했습니다. 새로 고침해 주세요.</p>}
    </section>
  );
}

/**
 * 전체 지도 표시 체크: 모두 켜지면 체크, 모두 꺼지면 빈칸, 일부면 중간 상태. 누르면 모두 켜거나(체크가 아닐 때) 모두 끈다
 * (지도 표시만 바뀌고 분석 기준은 그대로, 분석 기준의 공급 시설 점은 늘 보인다).
 */
function AllToggle({ ids, visible, toggle }) {
  const ref = useRef(null);
  const on = ids.filter((id) => visible.has(id)).length;
  const all = on === ids.length;
  const some = on > 0 && !all;
  useEffect(() => { if (ref.current) ref.current.indeterminate = some; }, [some]);
  return (
    <label className="all-toggle">
      <input ref={ref} type="checkbox" checked={all} onChange={() => toggle(ids, !all)} />
      전체
    </label>
  );
}

// 분석 탭 한 줄: 누르면 그 기준으로 분석, 고른 줄(on)을 다시 누르면 분석 끔
function BasisButton({ id, active, onChoose, head, children }) {
  return (
    <button type="button" className={`basis-row${head ? " basis-head" : ""}${active ? " on" : ""}`} aria-pressed={active} onClick={() => onChoose(id)}>
      {children}
    </button>
  );
}

/**
 * "분석" 탭: 그룹 줄(그룹 전체로 분석, scenario.js groupBasis)과 그 아래 유형 줄(유형 하나로 분석). 동그라미·체크박스 없이 버튼이라
 * 키보드(Tab·Enter·Space)로도 고른다. 시설 수 = 그 기준이 공급으로 쓰는 시설 수(의료·보건 그룹은 산부인과 뺌).
 */
function AnalysisTab({ groups, dbStatus, facilities, activeTypeId, onChooseBasis }) {
  const count = (ids) => facilities.filter((f) => ids.includes(f.typeId)).length;
  return (
    <section>
      {dbStatus === "error" && <p className="hint">자료를 불러오지 못했습니다. 새로 고침해 주세요.</p>}
      {groups.map((g) => {
        const gb = groupBasis(g);
        return (
          <div className="group" key={g.id} style={{ "--group": g.color }}>
            <BasisButton id={gb.id} active={activeTypeId === gb.id} onChoose={onChooseBasis} head>
              <span className="swatch" aria-hidden="true" />
              <span>{g.label}</span>
              <span className="count">{count(gb.members)}</span>
            </BasisButton>
            {g.types.map((t) => (
              <BasisButton key={t.id} id={t.id} active={activeTypeId === t.id} onChoose={onChooseBasis}>
                <span>{t.label}</span>
                <span className="count">{count([t.id])}</span>
              </BasisButton>
            ))}
          </div>
        );
      })}
    </section>
  );
}

// "시설 분류" 탭: 지도 표시만(체크·그룹 체크·전체). 유형 이름을 누르면 기존 시설 목록만 펼친다. 분석 기준은 바꾸지 않는다.
function FacilityTab({ groups, dbStatus, facilities, visible, toggle, selection, onSelectFacility }) {
  const allIds = groups.flatMap((g) => g.types.map((t) => t.id));
  return (
    <section>
      <h2 className="facility-head">사회문제 요인별 시설{allIds.length > 0 && <AllToggle ids={allIds} visible={visible} toggle={toggle} />}</h2>
      {dbStatus === "error" && <p className="hint">자료를 불러오지 못했습니다. 새로 고침해 주세요.</p>}
      {groups.map((g) => {
        const ids = g.types.map((t) => t.id);
        const allOn = ids.length > 0 && ids.every((id) => visible.has(id));
        return (
          <div className="group" key={g.id} style={{ "--group": g.color }}>
            <label className="group-head">
              <input type="checkbox" checked={allOn} onChange={(e) => toggle(ids, e.target.checked)} />
              <span className="swatch" aria-hidden="true" />
              {g.label}
            </label>
            {g.types.map((t) => {
              const list = facilities.filter((f) => f.typeId === t.id);
              return (
                <details className="type-row" key={t.id}>
                  <summary>
                    <input type="checkbox" aria-label={`${t.label} 지도 표시`} checked={visible.has(t.id)} onChange={(e) => toggle([t.id], e.target.checked)} onClick={(e) => e.stopPropagation()} />
                    <span className="type-name">{t.label}</span>
                    <span className="count">{list.length}</span>
                  </summary>
                  <ul className="facility-list">
                    {list.map((f) => (
                      <li key={f.id}>
                        <button type="button" className={selection?.id === f.id ? "on" : ""} onClick={() => onSelectFacility(f.id)}>
                          {f.name}
                        </button>
                      </li>
                    ))}
                  </ul>
                </details>
              );
            })}
          </div>
        );
      })}
    </section>
  );
}

export default function ControlPanel(props) {
  const { activeType, summary, env, hasPop, scenario, base, regionList, regions, onChangeRegions } = props;
  // 처음 열리는 탭은 "분석"(분석 켜고 끄기는 이 탭에서만)
  const [tab, setTab] = useState("analysis");
  const tabs = [
    ["analysis", "분석"],
    ["facility", "시설 분류"],
    ["added", `추가 시설 ${scenario.actions.length}`]
  ];

  return (
    <aside className="panel" aria-label="시설 분류와 시뮬레이션 결과">
      <SummaryCard
        type={activeType} summary={summary} base={base} radiusLabel={radiusText(env)} hasPop={hasPop}
        picker={<RegionPicker list={regionList} selected={regions} onChange={onChangeRegions} />}
      />
      <div className="tabs" role="tablist" aria-label="왼쪽 패널 보기">
        {tabs.map(([id, label]) => (
          <button key={id} type="button" role="tab" id={`tab-${id}`} aria-selected={tab === id} aria-controls={`pane-${id}`} className={tab === id ? "on" : ""} onClick={() => setTab(id)}>
            {label}
          </button>
        ))}
      </div>
      {/* 탭을 바꿔도 펼친 유형 목록이 접히지 않도록 탭 내용을 모두 두고 숨기기만 한다 */}
      <div className="tab-body" id="pane-analysis" role="tabpanel" aria-labelledby="tab-analysis" hidden={tab !== "analysis"}>
        <AnalysisTab {...props} />
      </div>
      <div className="tab-body" id="pane-facility" role="tabpanel" aria-labelledby="tab-facility" hidden={tab !== "facility"}>
        <FacilityTab {...props} toggle={props.onToggleTypes} />
      </div>
      <div className="tab-body" id="pane-added" role="tabpanel" aria-labelledby="tab-added" hidden={tab !== "added"}>
        <ScenarioTab {...props} type={activeType} />
      </div>
    </aside>
  );
}
