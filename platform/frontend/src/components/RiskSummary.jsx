import { RISK_GRADES, riskGrade } from "../lib/riskGrade.js";

/**
 * ③ 분포 카드 머리 — 선택 지자체의 지방소멸위험지수 값·등급 막대·인구·고령인구 비율·합계출산율.
 * 값은 지역 자료의 출처 값(basis source)만 쓰고, 없는 값은 칸을 비운다. 두 모드 같은 자리.
 * @param {{ region: object }} props
 */
export default function RiskSummary({ region }) {
  const isSource = (field) => region.basis?.[field] === "source" && Number.isFinite(region[field]);
  const grade = isSource("riskIndex") ? riskGrade(region.riskIndex) : null;
  const items = [
    ["인구", isSource("population") ? `${region.population.toLocaleString("ko-KR")}명` : null],
    ["고령인구 비율", isSource("agingIndex") ? `${region.agingIndex}%` : null],
    ["합계출산율", isSource("birthRate") ? `${region.birthRate}명` : null]
  ].filter(([, v]) => v);
  return (
    <div className="risk-summary">
      <div className="risk-summary-head">
        <span className="risk-summary-name">{region.name}</span>
        {grade && <strong className="risk-summary-value">지방소멸위험지수 {region.riskIndex} · {grade.label}</strong>}
      </div>
      {grade && (
        <ol className="risk-scale" aria-label={`지방소멸위험지수 등급 ${grade.label}`}>
          {RISK_GRADES.map((g) => (
            <li key={g.label} className={g === grade ? "on" : undefined} style={{ "--grade": g.markerColor }}>
              <span className="risk-scale-bar" aria-hidden="true"></span>
              <span className="risk-scale-label">{g.label}</span>
            </li>
          ))}
        </ol>
      )}
      <dl className="risk-summary-stats">
        {items.map(([k, v]) => [<dt key={`${k}-t`}>{k}</dt>, <dd key={`${k}-d`}>{v}</dd>])}
      </dl>
    </div>
  );
}
