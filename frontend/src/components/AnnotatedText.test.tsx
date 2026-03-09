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

  it("adds button semantics for clickable spans and supports keyboard activation", () => {
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
    expect(span?.getAttribute("role")).toBe("button");
    expect(span?.getAttribute("tabindex")).toBe("0");

    fireEvent.keyDown(span as HTMLElement, { key: "Enter" });

    expect(onSpanClick).toHaveBeenCalledTimes(1);
    expect(onSpanClick).toHaveBeenCalledWith(0, expect.any(Object));
  });
});
