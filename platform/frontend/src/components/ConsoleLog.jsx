import { useEffect, useRef } from "react";

// 로그 레벨은 색상 + 문자열·아이콘으로 이중 부호화 (DESIGN.md §5.7)
const LOG_LEVELS = {
  INFO: { className: "log-info", icon: "fa-circle-info" },
  WARN: { className: "log-warn", icon: "fa-triangle-exclamation" },
  ERROR: { className: "log-err", icon: "fa-circle-xmark" },
  SUCCESS: { className: "log-success", icon: "fa-circle-check" }
};

const LEGACY_TYPE_LEVEL = {
  "log-info": "INFO",
  "log-warn": "WARN",
  "log-err": "ERROR",
  "log-success": "SUCCESS"
};

function presentLog(log) {
  let message = String(log.message ?? "").trim();
  const prefixMatch = message.match(/^(INFO|WARN(?:ING)?|ERROR|SUCCESS|ALERT)\s*:\s*/i);
  const explicitLevel = prefixMatch?.[1]?.toUpperCase();
  const candidateLevel =
    explicitLevel === "WARNING" || explicitLevel === "ALERT"
      ? "WARN"
      : explicitLevel || String(log.level ?? "").toUpperCase() || LEGACY_TYPE_LEVEL[log.type] || "INFO";
  const level = LOG_LEVELS[candidateLevel] ? candidateLevel : "INFO";

  // 정규화 전 형식의 로그를 받아도 접두사를 중복 렌더링하지 않는다.
  if (prefixMatch) message = message.slice(prefixMatch[0].length).trimStart();

  return { message, level, ...LOG_LEVELS[level] };
}

export default function ConsoleLog({ logs, height = 180 }) {
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div
      className="console-log"
      style={{ height }}
      ref={scrollRef}
      role="log"
      aria-live="polite"
      aria-relevant="additions"
    >
      {logs.map((log, idx) => {
        const presentation = presentLog(log);
        return (
          <div key={idx} className="log-entry">
            <span className="log-time">[{log.time}]</span>
            <span className={presentation.className}>
              <i className={`fa-solid ${presentation.icon} log-level-icon`} aria-hidden="true"></i>
              [{presentation.level}] {presentation.message}
            </span>
          </div>
        );
      })}
    </div>
  );
}
