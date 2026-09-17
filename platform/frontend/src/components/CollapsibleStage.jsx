import InfoTip from "./InfoTip.jsx";

// 접고/펼칠 수 있는 단계 섹션 래퍼.
// 주의: body는 항상 DOM에 두고 CSS로만 숨긴다(언마운트 금지) — STAGE③ Leaflet
// 지도 인스턴스가 재생성되며 깨지는 것을 방지하기 위함.
// subTip: sub와 같은 보조 설명을 본문 대신 제목 옆 툴팁으로 보여 준다(둘 중 하나만 쓴다).
export default function CollapsibleStage({ id, no, title, sub, subTip, open = true, onToggle, children }) {
  const handleKey = (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onToggle?.();
    }
  };

  return (
    <section id={id} className={"pl-stage" + (open ? "" : " collapsed")}>
      <div
        className="pl-stage-head"
        role="button"
        tabIndex={0}
        aria-expanded={open}
        aria-controls={`${id}-body`}
        onClick={onToggle}
        onKeyDown={handleKey}
      >
        <span className="pl-stage-badge">{no}</span>
        <h2 className="pl-stage-title">{title}</h2>
        {/* 머리글 전체가 role=button이므로 툴팁 버튼의 클릭·키 입력이 접기 토글로 새지 않게 막는다. */}
        {subTip && (
          <span
            className="pl-stage-tip"
            onClick={(e) => e.stopPropagation()}
            onKeyDown={(e) => e.stopPropagation()}
          >
            <InfoTip text={subTip} label={`${title} 설명 보기`} />
          </span>
        )}
        {sub && <p className="pl-stage-sub">{sub}</p>}
        <i className="fa-solid fa-chevron-down pl-stage-chevron" aria-hidden="true"></i>
      </div>
      <div className="pl-stage-body" id={`${id}-body`} role="region" aria-label={title}>
        {children}
      </div>
    </section>
  );
}
