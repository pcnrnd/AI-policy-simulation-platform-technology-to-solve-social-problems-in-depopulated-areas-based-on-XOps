import { useState } from "react";
import { Line } from "react-chartjs-2";
import Card from "./Card.jsx";
import CollapsibleStage from "./CollapsibleStage.jsx";
import PendingData from "./PendingData.jsx";
import { displayId } from "../lib/modelops.js";
import { useChartTheme } from "../hooks/useChartTheme.js";
import { formatUnitValue, signedAmount, signedShort, scopeFactors, periodView, FACTOR_EXPLAIN_TEXT, FACTOR_IDLE_TEXT, FACTOR_SCOPE_EMPTY_TEXT } from "../lib/realSimulation.js";

// STAGE ② 요인분석 결과 = 모든 결과(결정 26). 두 모드 같은 구조, 값만 다르다.
// 첫 줄: 요인 기여도(+/− 구역에 요인 이름·기여도 값) + 문제 유형 진단. 둘째 줄: 모델 정보 + 검증. 아래 계산 근거.
// UI(카드 구조)는 항상 노출하고, 데이터는 STAGE ① 요인분석 완료(locked=false) 후 공개한다. 줄마다 두 카드는 같은 높이(pl-flow-grid stretch).
// view: factorAnalysisView 결과. scope: 요인 범위("all" | "policy") — 기여도 카드의 표시 범위만 바꾼다(모델·예측값·기여도 합 검증은 전체 기준).
// diagnosis: 두 모드 같은 diagnosisView 결과. pendingText: 잠금 중 대신 보일 문구. diagnosisSource: 진단 값의 출처("mock" | "api").
// 구역마다 기여도 큰 순 ZONE_LIMIT개를 먼저 보이고, 더 있으면 [모두 보기]로 펼친다.
// period: ① 분석 기간 결과(없으면 예측월 한 달 기여도). label: 요인 이름. 기간 결과가 있으면 기여도 카드 머리에 요인별|월별, 월평균|합계 전환.
// 월별은 요인별 본문을 같은 자리에 숨겨 두고 그 위에 선 그래프를 겹쳐 카드 높이를 바꾸지 않는다.
const ZONE_LIMIT = 5;
const TREND_COLORS = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6"];
const SHAPES = [{ id: "factor", label: "요인별" }, { id: "month", label: "월별" }];
const AGGS = [{ id: "mean", label: "월평균" }, { id: "total", label: "합계" }];
export default function FactorResultStage({ open, onToggle, locked = false, running = false, view = null, scope = "all", diagnosis = null,
  diagnosisSource = "api", pendingText = null, evidence = null, period = null, label = (feature) => feature }) {
  const [expanded, setExpanded] = useState(false);
  const [shape, setShape] = useState("factor");
  const [agg, setAgg] = useState("mean");
  const ct = useChartTheme();
  const ready = !locked && view;
  const pending = () => <PendingData running={running} text={pendingText ?? FACTOR_IDLE_TEXT} />;
  const periodShown = ready && period ? periodView(period, { label, agg, scope }) : null;
  const rows = periodShown ? periodShown.factors : ready ? scopeFactors([...view.factors, ...view.rest], scope) : [];
  const monthly = periodShown && shape === "month";
  const zones = [
    { key: "pos", title: "높인 요인", icon: "fa-arrow-trend-up", rows: rows.filter((r) => r.value > 0) },
    { key: "neg", title: "낮춘 요인", icon: "fa-arrow-trend-down", rows: rows.filter((r) => r.value < 0) }
  ];
  const more = zones.some((z) => z.rows.length > ZONE_LIMIT);
  const value = (number) => formatUnitValue(number, view?.unit);
  const versionTitle = view?.model.versionId; // 화면은 짧은 ID, 전체 ID는 title
  return (
    <CollapsibleStage
      id="stage-result"
      no="STAGE ②"
      title="요인분석 결과"
      sub="데이터 기반 지자체 사회문제 진단"
      open={open}
      onToggle={onToggle}
    >
      <div className="pl-flow-grid pl-flow-2">
        {/* 요인 기여도 — 기존 ①-1 목록과 ② 상관관계 카드를 합쳤다(+/− 구역에 이름과 기여도 값). */}
        <Card title="예측을 높이고 낮춘 요인" icon="fa-list-check" dataSource="api" data-slot="sim.result.card-1"
          headerRight={periodShown ? (
            <>
              <Segmented label="기여도 보기" options={SHAPES} value={shape} onChange={setShape} />
              <Segmented label="기여도 집계" options={AGGS} value={agg} onChange={setAgg} />
            </>
          ) : null}>
          {!ready ? pending() : rows.length === 0 ? <PendingData text={FACTOR_SCOPE_EMPTY_TEXT} /> : (
            <div className={"pl-contrib-body" + (monthly ? " is-monthly" : "")}>
              <div className="pl-corr-grid" aria-hidden={monthly || undefined}>
                {zones.map((z) => (
                  <div key={z.key} className={`pl-corr pl-corr-${z.key}`}>
                    <div className="pl-corr-head">
                      <i className={`fa-solid ${z.icon}`} aria-hidden="true"></i> {z.title}
                    </div>
                    <ul>
                      {z.rows.length ? (expanded ? z.rows : z.rows.slice(0, ZONE_LIMIT)).map((r) => (
                        <li key={r.feature}><span title={r.label}>{r.label}</span><b title={signedAmount(r.value, view.unit)}>{signedShort(r.value, view.unit)}</b></li>
                      )) : <li>해당 요인 없음</li>}
                    </ul>
                  </div>
                ))}
              </div>
              {more && (
                <div className="pl-contrib-foot" aria-hidden={monthly || undefined}>
                  <button type="button" className="pl-contrib-more" onClick={() => setExpanded((v) => !v)} aria-expanded={expanded} tabIndex={monthly ? -1 : undefined}>
                    {expanded ? "접기" : `모두 보기 (${rows.length})`}
                  </button>
                </div>
              )}
              {monthly && (
                <div className="pl-contrib-trend">
                  <Line data={{ labels: periodShown.trend.labels, datasets: periodShown.trend.lines.map((line, i) => ({
                    label: line.label, data: line.data, borderColor: TREND_COLORS[i], backgroundColor: TREND_COLORS[i],
                    pointRadius: 2, borderWidth: 2, tension: 0.2, spanGaps: false })) }}
                  options={{ responsive: true, maintainAspectRatio: false, animation: false,
                    scales: { y: { grid: { color: ct.grid }, ticks: { color: ct.tick } }, x: { grid: { display: false }, ticks: { color: ct.tick, maxRotation: 0, autoSkip: true } } },
                    plugins: { legend: { position: "bottom", labels: { color: ct.legend, boxWidth: 10, font: { size: 11 } } },
                      tooltip: { callbacks: { label: (cx) => `${cx.dataset.label}: ${signedAmount(cx.raw, view.unit)}` } } } }} />
                </div>
              )}
              <p className="pl-stat-baseline sr-only">{FACTOR_EXPLAIN_TEXT}</p>
            </div>
          )}
        </Card>

        <Card title="문제 유형 진단 결과" icon="fa-stethoscope" dataSource={diagnosisSource} data-slot="sim.result.card-2"
          headerRight={!locked && diagnosis ? (
            <span className="pl-diag-link"><i className="fa-solid fa-arrow-down-long" aria-hidden="true"></i> ③ 시뮬레이션으로 연결</span>
          ) : null}>
          {!locked && diagnosis ? (
            <DiagnosisBody diagnosis={diagnosis} />
          ) : (
            <PendingData running={running} text={pendingText ?? "[요인분석 실행] 후 문제 유형 진단 결과가 표시됩니다."} />
          )}
        </Card>
      </div>

      <div className="pl-flow-grid pl-flow-2 pl-result-row">
        {/* 모델 정보 — 기여도 분해식과 활성 버전. 화면에는 특정 알고리즘 이름을 쓰지 않는다(사용자 지시 11-17). */}
        <Card title="요인분석 모델" icon="fa-microchip" dataSource="api" data-slot="sim.result.card-3">
          {!ready ? pending() : (
            <div className="pl-model-pair">
              <div className="pl-model-box pl-model-xai">
                <span className="pl-model-tag">기여도 분해</span>
                <strong>요인별 기여도 합 + 기준값 {value(view.model.intercept)} {view.unit}</strong>
                <p>= 모델 추정값 {value(view.model.prediction)} {view.unit}</p>
              </div>
              <div className="pl-model-box pl-model-dl">
                <span className="pl-model-tag">학습 모델</span>
                <strong>요인분석 · 활성 버전 <span title={versionTitle}>{displayId(versionTitle)}</span></strong>
                <div className="pl-model-chips">
                  {view.model.features.map((m) => <span className="pl-chip" key={m}>{m}</span>)}
                </div>
              </div>
            </div>
          )}
        </Card>

        {/* 검증 결과(결과 융합) */}
        <Card title="통계적 모델 분석 결과" icon="fa-chart-column" dataSource="api" data-slot="sim.result.card-4">
          {!ready ? pending() : (
            <>
              <div className="pl-fusion-mark">
                <i className="fa-solid fa-plus" aria-hidden="true"></i> 결과 융합
              </div>
              <p className="pl-stat-baseline">
                검증 평균 절대 오차 {value(view.validation.mae)} {view.unit} · 기준선 {value(view.validation.baselineMae)} {view.unit}
              </p>
              <div className="pl-stage-label">분석 대상</div>
              <ul className="pl-mini-list">
                {/* 분석 기간은 마지막 줄(기준월) 뒤에 붙인다 — 줄을 늘리지 않아 ② 둘째 줄 높이가 그대로다. */}
                {view.validation.targets.map((t, i, all) => (
                  <li key={t}>{periodShown && i === all.length - 1 ? `${t} · ${periodShown.info}` : t}</li>
                ))}
              </ul>
            </>
          )}
        </Card>
      </div>
      {evidence}
    </CollapsibleStage>
  );
}

/** 두세 값 중 하나를 고르는 버튼 묶음(카드 머리 오른쪽). */
function Segmented({ label, options, value, onChange }) {
  return (
    <span className="pl-seg" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o.id} type="button" className={"pl-seg-btn" + (o.id === value ? " is-on" : "")} aria-pressed={o.id === value} onClick={() => onChange(o.id)}>
          {o.label}
        </button>
      ))}
    </span>
  );
}

const KIND_TONE = { 감소형: "down", 증가형: "up", 정체형: "flat" };

/** 문제 유형 진단 본문 — 두 모드 같은 구조: 머리(유형·변화율, 지표별 상태) → 요인 행(이름·기여도) → 점검 항목. ③ 연결 표지는 카드 제목 줄 오른쪽. */
function DiagnosisBody({ diagnosis: d }) {
  return (
    <div className="pl-diag">
      <div className={`pl-diag-head is-${KIND_TONE[d.kind] ?? "none"}`}>
        <div className="pl-diag-head-main">
          {d.kind && <span className="pl-diag-kind">{d.kind}</span>}
          <strong className="pl-diag-change">{d.change}</strong>
          <span className="pl-diag-basis">{d.basis}</span>
        </div>
        {d.indicators?.length > 0 && (
          <div className="pl-diag-indicators">
            <span className="pl-diag-tag">지표별</span>
            {d.indicators.map((e) => (
              <span key={e.target} className={`pl-diag-chip is-${KIND_TONE[e.kind] ?? "none"}`}>
                {e.name}{e.kind && <b>{e.kind}</b>}<span>{e.change}</span>
              </span>
            ))}
          </div>
        )}
      </div>
      {d.rows?.length > 0 && (
        <dl className="pl-diag-rows">
          {d.rows.map((r) => (
            <div key={r.name} className={`pl-diag-row is-${r.dir}`}>
              <dt>{r.name}</dt>
              <dd className="pl-diag-label">{r.label}</dd>
              <dd className="pl-diag-value">{r.value}</dd>
            </div>
          ))}
        </dl>
      )}
      <div className="pl-diag-checks">
        <span className="pl-diag-tag">점검 항목</span>
        {d.items?.length ? d.items.map((text) => <span key={text} className="pl-diag-check">{text}</span>)
          : <span className="pl-diag-none">해당 항목 없음</span>}
      </div>
    </div>
  );
}
