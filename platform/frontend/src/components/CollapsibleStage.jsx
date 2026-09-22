import InfoTip from "./InfoTip.jsx";

// 접고/펼칠 수 있는 단계 섹션 래퍼.
// 주의: body는 항상 DOM에 두고 CSS로만 숨긴다(언마운트 금지) — STAGE③ Leaflet
// 지도 인스턴스가 재생성되며 깨지는 것을 방지하기 위함.
// subTip: sub와 같은 보조 설명을 본문 대신 제목 옆 툴팁으로 보여 준다(둘 중 하나만 쓴다).
export default function CollapsibleStage({ id, no, title, sub, subTip, open = true, onToggle, children }) {
  return (
    <section id={id} className={"pl-stage" + (open ? "" : " collapsed")}>
      {/* 머리글은 flex 컨테이너일 뿐이고 접기 조작은 끝의 토글 버튼이 맡는다. 머리글 자체를
          role=button 으로 두면 안에 놓인 InfoTip 버튼이 버튼 안의 버튼이 되어 초점·역할이 깨진다.
          ponytail: 아무 데나 눌러도 접히던 기존 조작감은 컨테이너 onClick으로 남긴다(마우스 전용
          보조 수단이고, 키보드 경로는 아래 토글 버튼이 네이티브로 처리한다). */}
      <div className="pl-stage-head" onClick={onToggle}>
        <span className="pl-stage-badge">{no}</span>
        <h2 className="pl-stage-title">{title}</h2>
        {/* 툴팁 버튼의 클릭이 머리글 컨테이너로 올라가 접기 토글로 새지 않게 막는다. */}
        {subTip && (
          <span className="pl-stage-tip" onClick={(e) => e.stopPropagation()}>
            <InfoTip text={subTip} label={`${title} 설명 보기`} />
          </span>
        )}
        {sub && <p className="pl-stage-sub">{sub}</p>}
        {/* onClick을 달지 않는다 — 버튼 클릭(마우스·Enter·Space 모두)이 머리글로 버블링되어
            컨테이너 onClick 한 번만 실행된다. 양쪽에 달면 한 번 눌러 두 번 토글된다. */}
        <button
          type="button"
          className="pl-stage-toggle"
          aria-expanded={open}
          aria-controls={`${id}-body`}
          aria-label={`${title} 접기/펼치기`}
        >
          <i className="fa-solid fa-chevron-down pl-stage-chevron" aria-hidden="true"></i>
        </button>
      </div>
      <div className="pl-stage-body" id={`${id}-body`} role="region" aria-label={title}>
        {children}
      </div>
    </section>
  );
}
