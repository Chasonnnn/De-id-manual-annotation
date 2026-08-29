import { cleanup, render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import AnnotationPopup from "./AnnotationPopup";

const showPopover = vi.fn();
const hidePopover = vi.fn();

beforeEach(() => {
  Object.defineProperties(HTMLElement.prototype, {
    showPopover: { configurable: true, value: showPopover },
    hidePopover: { configurable: true, value: hidePopover },
  });
});

afterEach(() => {
  cleanup();
  delete (HTMLElement.prototype as { showPopover?: () => void }).showPopover;
  delete (HTMLElement.prototype as { hidePopover?: () => void }).hidePopover;
  vi.clearAllMocks();
});

function toggleEvent(newState: "open" | "closed"): Event {
  const event = new Event("toggle");
  Object.defineProperty(event, "newState", { value: newState });
  return event;
}

describe("AnnotationPopup", () => {
  it("opens as a native popover and reports native light dismissal", () => {
    const firstClose = vi.fn();
    const latestClose = vi.fn();
    const view = render(
      <AnnotationPopup
        x={0}
        y={0}
        labels={["NAME"]}
        onSelect={vi.fn()}
        onClose={firstClose}
      />,
    );

    view.rerender(
      <AnnotationPopup
        x={0}
        y={0}
        labels={["NAME"]}
        onSelect={vi.fn()}
        onClose={latestClose}
      />,
    );
    const popup = view.container.querySelector(".annotation-popup") as HTMLDivElement;

    expect(popup.getAttribute("popover")).toBe("auto");
    expect(showPopover).toHaveBeenCalledOnce();
    expect(showPopover.mock.instances[0]).toBe(popup);
    expect(view.queryByRole("button", { name: "Close annotation menu" })).toBeNull();

    popup.dispatchEvent(toggleEvent("open"));
    expect(latestClose).not.toHaveBeenCalled();

    popup.dispatchEvent(toggleEvent("closed"));

    expect(latestClose).toHaveBeenCalledOnce();
    expect(firstClose).not.toHaveBeenCalled();
  });
});
