// dataSource: 표시값의 출처("api" | "mock"). 지정하지 않으면 속성 자체가 붙지 않는다.
export function Card({
  title,
  titleId,
  titleTabIndex,
  icon,
  headerRight,
  className = "",
  children,
  style,
  dataSource
}) {
  return (
    <div className={"card " + className} style={style} data-source={dataSource}>
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
