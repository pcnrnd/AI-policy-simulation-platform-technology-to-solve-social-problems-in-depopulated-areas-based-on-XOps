// 접고/펼칠 수 있는 단계 섹션 래퍼.
// 주의: body는 항상 DOM에 두고 CSS로만 숨긴다(언마운트 금지) — STAGE③ Leaflet
// 지도 인스턴스가 재생성되며 깨지는 것을 방지하기 위함.
export default function CollapsibleStage({ id, no, title, sub, open = true, onToggle, children }) {
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
        {sub && <p className="pl-stage-sub">{sub}</p>}
        <i className="fa-solid fa-chevron-down pl-stage-chevron" aria-hidden="true"></i>
      </div>
      <div className="pl-stage-body" id={`${id}-body`} role="region" aria-label={title}>
        {children}
      </div>
    </section>
  );
}
