import { useAppState } from "../context/AppStateContext.jsx";

// 다음 단계 CTA 배너 — 이벤트가 발생한 탭에서만 렌더하고 이동 버튼 1개만 둔다.
// 이동은 navigateToTab으로 처리해 탭 전환 후 초점이 해당 탭 패널로 옮겨진다(기존 크로스탭 규약).
const TONE_ICON = {
  info: "fa-circle-info",
  warn: "fa-triangle-exclamation",
  success: "fa-circle-check"
};

export default function NextStepBanner({ message, actionLabel, targetTab, tone = "info" }) {
  const { navigateToTab } = useAppState();

  return (
    <div className={`xops-next-step is-${tone}`} role="status" aria-live="polite">
      <span className="xops-next-step-msg">
        <i className={`fa-solid ${TONE_ICON[tone] ?? TONE_ICON.info}`} aria-hidden="true"></i>{" "}
        {message}
      </span>
      <button
        type="button"
        className="btn btn-secondary xops-next-step-btn"
        onClick={() => navigateToTab(targetTab)}
      >
        {actionLabel} <i className="fa-solid fa-arrow-right" aria-hidden="true"></i>
      </button>
    </div>
  );
}
