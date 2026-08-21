import { useEffect, useId, useRef } from "react";
import { createPortal } from "react-dom";

const FOCUSABLE =
  'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

/**
 * 되돌리기 어려운 동작을 확인하는 공통 대화상자.
 * 취소 시 실행 버튼으로 초점을 복귀하고, 실행 후 대상이 사라진 경우에는 호출자가 nextFocusSelector를 지정한다.
 */
export default function ConfirmDialog({
  open,
  title,
  description,
  confirmLabel,
  onConfirm,
  onCancel,
  busy = false,
  nextFocusSelector,
  fallbackFocusSelector
}) {
  const titleId = useId();
  const descriptionId = useId();
  const dialogRef = useRef(null);
  const cancelRef = useRef(null);
  const previousFocusRef = useRef(null);
  const onCancelRef = useRef(onCancel);
  const busyRef = useRef(busy);

  useEffect(() => {
    onCancelRef.current = onCancel;
    busyRef.current = busy;
  }, [busy, onCancel]);

  useEffect(() => {
    if (!open) return undefined;

    previousFocusRef.current = document.activeElement;
    const appRoot = document.getElementById("root")?.firstElementChild;
    const previousAriaHidden = appRoot?.getAttribute("aria-hidden");
    appRoot?.setAttribute("inert", "");
    appRoot?.setAttribute("aria-hidden", "true");
    const frame = requestAnimationFrame(() => cancelRef.current?.focus());

    const onKeyDown = (event) => {
      if (event.key === "Escape" && !busyRef.current) {
        event.preventDefault();
        onCancelRef.current();
        return;
      }
      if (event.key !== "Tab") return;

      const focusable = [...(dialogRef.current?.querySelectorAll(FOCUSABLE) ?? [])];
      if (focusable.length === 0) {
        event.preventDefault();
        dialogRef.current?.focus();
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (!dialogRef.current?.contains(document.activeElement)) {
        event.preventDefault();
        first.focus();
      } else if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };

    document.addEventListener("keydown", onKeyDown);
    return () => {
      cancelAnimationFrame(frame);
      document.removeEventListener("keydown", onKeyDown);
      appRoot?.removeAttribute("inert");
      if (previousAriaHidden === null) appRoot?.removeAttribute("aria-hidden");
      else appRoot?.setAttribute("aria-hidden", previousAriaHidden);
      const previous = previousFocusRef.current;
      requestAnimationFrame(() => {
        if (previous && document.contains(previous)) previous.focus();
        else {
          const nextTarget = nextFocusSelector ? document.querySelector(nextFocusSelector) : null;
          const fallbackTarget = fallbackFocusSelector ? document.querySelector(fallbackFocusSelector) : null;
          (nextTarget ?? fallbackTarget)?.focus();
        }
      });
    };
  }, [fallbackFocusSelector, nextFocusSelector, open]);

  useEffect(() => {
    if (open && busy) dialogRef.current?.focus();
  }, [busy, open]);

  if (!open) return null;

  return createPortal(
    <div
      className="confirm-dialog-backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && !busy) onCancel();
      }}
    >
      <div
        ref={dialogRef}
        className="confirm-dialog"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={descriptionId}
        aria-busy={busy}
        tabIndex={-1}
      >
        <div className="confirm-dialog-icon" aria-hidden="true">
          <i className="fa-solid fa-triangle-exclamation"></i>
        </div>
        <div className="confirm-dialog-body">
          <h2 id={titleId}>{title}</h2>
          <p id={descriptionId}>{description}</p>
        </div>
        <div className="confirm-dialog-actions">
          <button ref={cancelRef} type="button" className="btn btn-secondary" onClick={onCancel} disabled={busy}>
            취소
          </button>
          <button type="button" className="btn btn-danger" onClick={onConfirm} disabled={busy}>
            {busy && <i className="fa-solid fa-spinner fa-spin" aria-hidden="true"></i>}
            {busy ? "처리 중" : confirmLabel}
          </button>
        </div>
      </div>
    </div>,
    document.body
  );
}
