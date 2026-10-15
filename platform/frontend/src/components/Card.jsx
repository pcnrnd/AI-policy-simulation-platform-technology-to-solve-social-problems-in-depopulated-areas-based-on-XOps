import SourceChip from "./SourceChip.jsx";

// dataSource: 표시값의 출처("api" | "mock" | "realdata"). data-values-source 속성으로 나가고,
// 제목 오른쪽에 모드별 출처 칩(SourceChip)을 그린다. sourceLabel 이 있으면 칩 문구로 우선한다.
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
  sourceLabel,
  ...rest
}) {
  // 같은 속성이 두 경로로 오므로 명시 prop dataSource를 우선하고, 없으면 rest로 온 값을 쓴다.
  const source = dataSource ?? rest["data-values-source"];
  const hasChip = Boolean(source || sourceLabel);
  return (
    <div
      className={"card " + className}
      style={style}
      {...rest}
      data-values-source={source}
    >
      {(title || headerRight || hasChip) && (
        <div className="card-title-area">
          {title && (
            <h3 id={titleId} tabIndex={titleTabIndex} className="card-title">
              {icon && <i className={"fa-solid " + icon} aria-hidden="true"></i>}
              {title}
            </h3>
          )}
          {(hasChip || headerRight) && (
            <span className="card-title-side">
              {hasChip && <SourceChip dataSource={source} sourceLabel={sourceLabel} />}
              {headerRight}
            </span>
          )}
        </div>
      )}
      {children}
    </div>
  );
}

export default Card;
