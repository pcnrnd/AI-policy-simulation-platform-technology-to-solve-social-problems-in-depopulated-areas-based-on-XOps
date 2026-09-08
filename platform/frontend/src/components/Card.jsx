// rest: 호출부가 카드 껍데기에 직접 붙이는 DOM 속성(예: data-values-source).
export function Card({ title, titleId, titleTabIndex, icon, headerRight, className = "", children, style, ...rest }) {
  return (
    <div className={"card " + className} style={style} {...rest}>
      {(title || headerRight) && (
        <div className="card-title-area">
          {title && (
            <h3 id={titleId} tabIndex={titleTabIndex} className="card-title">
              {icon && <i className={"fa-solid " + icon} aria-hidden="true"></i>}
              {title}
            </h3>
          )}
          {headerRight}
        </div>
      )}
      {children}
    </div>
  );
}

export default Card;
