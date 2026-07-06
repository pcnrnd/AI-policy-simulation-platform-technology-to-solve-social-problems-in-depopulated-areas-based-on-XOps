export default function StatCard({ label, icon, value, footer, valueStyle }) {
  return (
    <div className="card stat-card">
      <div className="stat-header">
        <span className="stat-label">{label}</span>
        {icon && (
          <span className="stat-icon">
            <i className={"fa-solid " + icon}></i>
          </span>
        )}
      </div>
      <div className="stat-value" style={valueStyle}>
        {value}
      </div>
      {footer && <div className="stat-footer">{footer}</div>}
    </div>
  );
}
