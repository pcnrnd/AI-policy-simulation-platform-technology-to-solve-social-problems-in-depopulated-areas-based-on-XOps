// dataSource: 표시값의 출처("api" | "mock" | "realdata"). data-values-source 속성으로만 나가고 화면에는 출처 태그를 그리지 않는다.
// rest: 호출부가 카드 껍데기에 직접 붙이는 DOM 속성(data-values-source를 그대로 넘기는 호출부 포함).
export function Card({
  title,
  titleId,
  titleTabIndex,
  icon,
  headerRight,
  className = "",
  children,
  style,
  dataSource,
  ...rest
}) {
  // 같은 속성이 두 경로로 오므로 명시 prop dataSource를 우선하고, 없으면 rest로 온 값을 쓴다.
  const source = dataSource ?? rest["data-values-source"];
  return (
    <div
      className={"card " + className}
      style={style}
      {...rest}
      data-values-source={source}
    >
      {(title || headerRight) && (
        <div className="card-title-area">
          {title && (
            <h3 id={titleId} tabIndex={titleTabIndex} className="card-title">
              {icon && <i className={"fa-solid " + icon} aria-hidden="true"></i>}
              {title}
            </h3>
          )}
          {headerRight && <span className="card-title-side">{headerRight}</span>}
        </div>
      )}
      {children}
    </div>
  );
}

export default Card;
