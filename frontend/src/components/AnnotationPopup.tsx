import { useEffect, useEffectEvent, useRef } from "react";
import { getLabelColor } from "../hosted/types";

interface Props {
  x: number;
  y: number;
  labels: string[];
  onSelect: (label: string) => void;
  onDelete?: () => void;
  onClose: () => void;
}

function useNativePopover(
  popupRef: React.RefObject<HTMLDivElement | null>,
  onClose: () => void,
) {
  const handleClose = useEffectEvent(onClose);

  useEffect(() => {
    const element = popupRef.current;
    if (!element) return;

    const handleToggle = (event: ToggleEvent) => {
      if (event.newState === "closed") handleClose();
    };
    element.addEventListener("toggle", handleToggle);
    element.showPopover();

    return () => {
      element.removeEventListener("toggle", handleToggle);
      element.hidePopover();
    };
  }, [popupRef]);
}

function usePopoverPosition(
  popupRef: React.RefObject<HTMLDivElement | null>,
  x: number,
  y: number,
) {
  useEffect(() => {
    const element = popupRef.current;
    if (!element) return;
    const rect = element.getBoundingClientRect();
    let adjustedX = x;
    let adjustedY = y;

    if (rect.right > window.innerWidth) adjustedX = window.innerWidth - rect.width - 8;
    if (rect.bottom > window.innerHeight) adjustedY = y - rect.height - 8;
    if (adjustedX < 0) adjustedX = 8;
    if (adjustedY < 0) adjustedY = 8;

    element.style.left = `${adjustedX}px`;
    element.style.top = `${adjustedY}px`;
  }, [popupRef, x, y]);
}

export default function AnnotationPopup({
  x,
  y,
  labels,
  onSelect,
  onDelete,
  onClose,
}: Props) {
  const popupRef = useRef<HTMLDivElement>(null);
  useNativePopover(popupRef, onClose);
  usePopoverPosition(popupRef, x, y);

  return (
    <div
      ref={popupRef}
      className="annotation-popup"
      popover="auto"
      style={{ left: x, top: y }}
    >
      {labels.map((label) => (
        <button
          key={label}
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
  );
}
