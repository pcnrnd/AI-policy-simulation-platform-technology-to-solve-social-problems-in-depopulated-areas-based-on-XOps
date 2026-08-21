export function Card({ title, titleId, titleTabIndex, icon, headerRight, className = "", children, style }) {
  return (
    <div className={"card " + className} style={style}>
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
