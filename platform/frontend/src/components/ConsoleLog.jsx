import { useEffect, useRef } from "react";

export default function ConsoleLog({ logs, height = 180 }) {
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div className="console-log" style={{ height }} ref={scrollRef}>
      {logs.map((log, idx) => (
        <div key={idx} className="log-entry">
          <span className="log-time">[{log.time}]</span>
          <span className={log.type}>{log.message}</span>
        </div>
      ))}
    </div>
  );
}
