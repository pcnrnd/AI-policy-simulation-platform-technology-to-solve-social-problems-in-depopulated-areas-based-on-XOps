export function Card({ title, icon, headerRight, className = "", children, style }) {
  return (
    <div className={"card " + className} style={style}>
      {(title || headerRight) && (
        <div className="card-title-area">
          {title && (
            <h3 className="card-title">
              {icon && <i className={"fa-solid " + icon}></i>}
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
