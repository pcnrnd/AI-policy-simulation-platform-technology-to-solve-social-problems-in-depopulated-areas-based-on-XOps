import { useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

// 마우스·키보드·터치에서 같은 방식으로 열리고 화면 가장자리에서 잘리지 않는 도움말.
export default function InfoTip({ text, label = "설명 보기" }) {
  const [open, setOpen] = useState(false);
  const [position, setPosition] = useState({ left: 0, top: 0, placement: "top" });
  const triggerRef = useRef(null);
  const bubbleRef = useRef(null);
  const tooltipId = useId();

  useLayoutEffect(() => {
    if (!open) return undefined;
    const updatePosition = () => {
      const rect = triggerRef.current?.getBoundingClientRect();
      if (!rect) return;
      const bubbleHeight = bubbleRef.current?.getBoundingClientRect().height ?? 0;
      const halfWidth = Math.min(130, Math.max(80, (window.innerWidth - 24) / 2));
      const left = Math.min(window.innerWidth - halfWidth - 12, Math.max(halfWidth + 12, rect.left + rect.width / 2));
      const spaceAbove = rect.top - 12;
      const spaceBelow = window.innerHeight - rect.bottom - 12;
      const placement = spaceAbove >= bubbleHeight || spaceAbove >= spaceBelow ? "top" : "bottom";
      const unclampedTop = placement === "top" ? rect.top - 8 : rect.bottom + 8;
      const top = placement === "top"
        ? Math.max(bubbleHeight + 12, unclampedTop)
        : Math.min(window.innerHeight - bubbleHeight - 12, unclampedTop);
      setPosition({
        left,
        top: Math.max(12, top),
        placement
      });
    };
    updatePosition();
    window.addEventListener("resize", updatePosition);
    window.addEventListener("scroll", updatePosition, true);
    return () => {
      window.removeEventListener("resize", updatePosition);
      window.removeEventListener("scroll", updatePosition, true);
    };
  }, [open]);

  useEffect(() => {
    if (!open) return undefined;
    const closeFromOutside = (event) => {
      if (!triggerRef.current?.contains(event.target)) setOpen(false);
    };
    const closeFromEscape = (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        setOpen(false);
        triggerRef.current?.focus();
      }
    };
    document.addEventListener("pointerdown", closeFromOutside);
    document.addEventListener("keydown", closeFromEscape);
    return () => {
      document.removeEventListener("pointerdown", closeFromOutside);
      document.removeEventListener("keydown", closeFromEscape);
    };
  }, [open]);

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        className="infotip"
        aria-label={label}
        aria-expanded={open}
        aria-describedby={open ? tooltipId : undefined}
        onClick={() => setOpen(true)}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => {
          if (document.activeElement !== triggerRef.current) setOpen(false);
        }}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
      >
        <i className="fa-solid fa-circle-question" aria-hidden="true"></i>
      </button>
      {open && createPortal(
        <span
          ref={bubbleRef}
          id={tooltipId}
          className={`infotip-bubble is-open is-${position.placement}`}
          role="tooltip"
          style={{ left: position.left, top: position.top }}
        >
          {text}
        </span>,
        document.body
      )}
    </>
  );
}
