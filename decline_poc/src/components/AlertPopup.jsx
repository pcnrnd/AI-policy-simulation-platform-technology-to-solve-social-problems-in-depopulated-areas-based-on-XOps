import { useAppState } from "../context/AppStateContext.jsx";

export default function AlertPopupContainer() {
  const { alerts, dismissAlert } = useAppState();

  return (
    <div id="popup-container">
      {alerts.map((alert) => (
        <div className="alert-popup" key={alert.id}>
          <i
            className="fa-solid fa-triangle-exclamation"
            style={{ fontSize: 24, color: "var(--accent-red)" }}
          ></i>
          <div>
            <div style={{ fontWeight: 700, color: "#ffffff", fontSize: 14 }}>{alert.title}</div>
            <div
              style={{
                fontSize: 11,
                color: "var(--text-secondary)",
                marginTop: 2
              }}
            >
              {alert.message}
            </div>
          </div>
          <button
            className="alert-icon-btn btn-danger"
            style={{ width: 24, height: 24, fontSize: 10, marginLeft: 12 }}
            onClick={() => dismissAlert(alert.id)}
            aria-label="알림 닫기"
          >
            X
          </button>
        </div>
      ))}
    </div>
  );
}
