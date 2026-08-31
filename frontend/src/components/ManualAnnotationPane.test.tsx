import { cleanup, fireEvent, render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { buildNewSpansFromSelection } from "../annotationSelection";
import ManualAnnotationPane from "./ManualAnnotationPane";

beforeEach(() => {
  Object.defineProperties(HTMLElement.prototype, {
    showPopover: { configurable: true, value: vi.fn() },
    hidePopover: { configurable: true, value: vi.fn() },
  });
});

afterEach(() => {
  cleanup();
  window.getSelection()?.removeAllRanges();
  delete (HTMLElement.prototype as { showPopover?: () => void }).showPopover;
  delete (HTMLElement.prototype as { hidePopover?: () => void }).hidePopover;
});

describe("buildNewSpansFromSelection", () => {
  it("splits a URL around transcript speaker prefixes", () => {
    const text = "More at www.\nTutor: alimmenta.\nTutor: com today";
    const start = text.indexOf("www.");
    const end = text.indexOf(" today");

    expect(buildNewSpansFromSelection(text, start, end, "URL")).toEqual([
      {
        start,
        end: start + 4,
        label: "URL",
        text: "www.",
      },
      {
        start: text.indexOf("alimmenta."),
        end: text.indexOf("alimmenta.") + "alimmenta.".length,
        label: "URL",
        text: "alimmenta.",
      },
      {
        start: text.indexOf("com today"),
        end: text.indexOf("com today") + 3,
        label: "URL",
        text: "com",
      },
    ]);
  });

  it("skips unrelated turns inside a split Amara.org URL", () => {
    const text = "By Amara.\nTutor: Here we are.\nTutor: org today";
    const start = text.indexOf("Amara.");
    const end = text.indexOf(" today");

    expect(buildNewSpansFromSelection(text, start, end, "URL")).toEqual([
      {
        start,
        end: start + "Amara.".length,
        label: "URL",
        text: "Amara.",
      },
      {
        start: text.indexOf("org today"),
        end: text.indexOf("org today") + "org".length,
        label: "URL",
        text: "org",
      },
    ]);
  });

  it("keeps non-URL selections contiguous", () => {
    const text = "Anna\nTutor: Maria";

    expect(buildNewSpansFromSelection(text, 0, text.length, "NAME")).toEqual([
      {
        start: 0,
        end: text.length,
        label: "NAME",
        text,
      },
    ]);
  });
});

describe("ManualAnnotationPane", () => {
  it("keeps selected text highlighted while the annotation type menu is open", () => {
    render(
      <ManualAnnotationPane
        text="Hello Adoni."
        spans={[]}
        labels={["NAME", "ADDRESS"]}
        onSpansChange={vi.fn()}
      />,
    );
    const segment = document.querySelector('[data-offset="0"]');
    const textNode = segment?.firstChild;
    expect(textNode).toBeInstanceOf(Text);

    const range = document.createRange();
    range.setStart(textNode!, 6);
    range.setEnd(textNode!, 11);
    Object.defineProperty(range, "getBoundingClientRect", {
      value: () => ({
        bottom: 40,
        height: 20,
        left: 20,
        right: 70,
        top: 20,
        width: 50,
        x: 20,
        y: 20,
        toJSON: () => ({}),
      }),
    });
    const selection = window.getSelection();
    expect(selection).not.toBeNull();
    selection!.removeAllRanges();
    selection!.addRange(range);

    fireEvent.mouseUp(segment!);

    expect(document.querySelector(".annotation-popup")?.textContent).toContain("NAME");
    expect(selection!.rangeCount).toBe(1);
    expect(selection!.toString()).toBe("Adoni");
  });
});
