import { useEffect, useRef } from "react";
import { getLabelColor } from "../types";

interface Props {
  x: number;
  y: number;
  labels: string[];
  currentLabel?: string;
  onSelect: (label: string) => void;
  onDelete?: () => void;
  onClose: () => void;
}

export default function AnnotationPopup({
  x,
  y,
  labels,
  currentLabel,
  onSelect,
  onDelete,
  onClose,
}: Props) {
  const popupRef = useRef<HTMLDivElement>(null);

  // 4.4: Clamp popup position to viewport after mount
  useEffect(() => {
    const el = popupRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    let adjustedX = x;
    let adjustedY = y;

    if (rect.right > vw) {
      adjustedX = vw - rect.width - 8;
    }
    if (rect.bottom > vh) {
      adjustedY = y - rect.height - 8;
    }
    if (adjustedX < 0) adjustedX = 8;
    if (adjustedY < 0) adjustedY = 8;

    if (adjustedX !== x || adjustedY !== y) {
      el.style.left = `${adjustedX}px`;
      el.style.top = `${adjustedY}px`;
    }
  }, [x, y]);

  // Close on Escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  return (
    <>
      <div
        style={{
          position: "fixed",
          inset: 0,
          zIndex: 999,
        }}
        onClick={onClose}
      />
      <div
        ref={popupRef}
        className="annotation-popup"
        style={{ left: x, top: y }}
      >
        {labels.map((label) => (
          <button
            key={label}
            className={label === currentLabel ? "current-label" : ""}
            style={{ background: getLabelColor(label) }}
            onClick={() => onSelect(label)}
          >
            {label}
          </button>
        ))}
        {onDelete && (
          <button className="delete-btn" onClick={onDelete}>
            DELETE
          </button>
        )}
      </div>
    </>
  );
}
