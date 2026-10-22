// dataSource: 표시값의 출처("api" | "mock"). data-values-source 속성으로만 나가며,
// 화면에는 출처 태그를 그리지 않는다. 지정하지 않으면 속성도 붙지 않는다.
export default function StatCard({ label, icon, value, unit, footer, valueStyle, dataSource }) {
  return (
    <div className="card stat-card" data-values-source={dataSource}>
      <div className="stat-header">
        <span className="stat-label">
          {label}
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
