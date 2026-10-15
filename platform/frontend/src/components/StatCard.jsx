import SourceChip from "./SourceChip.jsx";

// dataSource: 표시값의 출처("api" | "mock"). data-values-source 속성과 라벨 옆 출처 칩으로 나가며,
// 지정하지 않으면 둘 다 붙지 않는다.
export default function StatCard({ label, icon, value, unit, footer, valueStyle, dataSource }) {
  return (
    <div className="card stat-card" data-values-source={dataSource}>
      <div className="stat-header">
        <span className="stat-label">
          {label} <SourceChip dataSource={dataSource} />
        </span>
        {icon && (
          <span className="stat-icon">
            <i className={"fa-solid " + icon}></i>
          </span>
        )}
      </div>
      <div className="stat-value" style={valueStyle}>
        {value}
        {unit && <span className="stat-unit">{unit}</span>}
      </div>
      {footer && <div className="stat-footer">{footer}</div>}
    </div>
  );
}
