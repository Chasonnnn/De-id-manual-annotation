import { describe, expect, it, vi } from "vitest";
import { fireEvent, render } from "@testing-library/react";

import AnnotatedText from "./AnnotatedText";

function getRenderedRawText(container: HTMLElement): string {
  return Array.from(container.querySelectorAll("[data-offset]"))
    .map((node) =>
      Array.from(node.childNodes)
        .filter(
          (child) =>
            !(
              child instanceof HTMLElement &&
              child.dataset.annotationLabel === "true"
            ),
        )
        .map((child) => child.textContent ?? "")
        .join(""),
    )
    .join("");
}

describe("AnnotatedText", () => {
  it("renders the correct pre-annotation text when backend offsets follow Python indexing", () => {
    const text = "prefix 😂\nstudent: Greenville\nvolunteer: hi";
    const start = Array.from("prefix 😂\nstudent: ").length;
    const end = start + Array.from("Greenville").length;
    const { container } = render(
      <AnnotatedText
        text={text}
        spans={[{ start, end, label: "LOCATION", text: "Greenville" }]}
      />,
    );

    const span = container.querySelector(".ann-span");
    expect(span).toBeTruthy();
    expect(span?.childNodes[0]?.textContent).toBe("Greenville");
    expect(span?.getAttribute("data-offset")).toBe(String(start));
    expect(span?.getAttribute("data-offset-end")).toBe(String(end));
  });

  it("does not duplicate raw text when spans overlap", () => {
    const text = "abcdef";
    const { container } = render(
      <AnnotatedText
        text={text}
        spans={[
          { start: 1, end: 4, label: "NAME", text: "bcd" },
          { start: 3, end: 6, label: "LOCATION", text: "def" },
        ]}
      />,
    );

    expect(getRenderedRawText(container)).toBe(text);
  });

  it("uses a native button for clickable spans", () => {
    const onSpanClick = vi.fn();
    const { container } = render(
      <AnnotatedText
        text="hello anna"
        spans={[{ start: 6, end: 10, label: "NAME", text: "anna" }]}
        clickable
        onSpanClick={onSpanClick}
      />,
    );

    const span = container.querySelector(".ann-span");
    expect(span?.tagName).toBe("BUTTON");

    fireEvent.click(span as HTMLElement);

    expect(onSpanClick).toHaveBeenCalledTimes(1);
    expect(onSpanClick).toHaveBeenCalledWith(0, expect.any(Object));
  });

  it("uses entity colors normally and binary colors in comparison mode", () => {
    const manual = [
      { start: 0, end: 5, label: "NAME", text: "Alice" },
      { start: 10, end: 13, label: "NAME", text: "Bob" },
    ];
    const reference = [
      { start: 0, end: 5, label: "NAME", text: "Alice" },
      { start: 10, end: 13, label: "SCHOOL", text: "Bob" },
    ];
    const { container, rerender } = render(
      <AnnotatedText text="Alice met Bob" spans={manual} comparisonSpans={reference} />,
    );

    expect(container.querySelectorAll(".comparison-match")).toHaveLength(0);
    expect(container.querySelectorAll(".comparison-difference")).toHaveLength(0);

    rerender(
      <AnnotatedText
        text="Alice met Bob"
        spans={manual}
        comparisonSpans={reference}
        comparisonMode
      />,
    );

    expect(container.querySelectorAll(".comparison-match")).toHaveLength(1);
    expect(container.querySelectorAll(".comparison-difference")).toHaveLength(1);
  });

  it("shows a missing annotation as a difference on the unannotated side", () => {
    const { container } = render(
      <AnnotatedText
        text="Alice met Bob"
        spans={[]}
        comparisonSpans={[{ start: 10, end: 13, label: "NAME", text: "Bob" }]}
        comparisonMode
      />,
    );

    expect(container.querySelector(".comparison-difference")?.textContent).toBe("Bob");
  });
});
