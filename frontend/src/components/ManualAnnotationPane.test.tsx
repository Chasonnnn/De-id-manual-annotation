import { describe, expect, it } from "vitest";
import { buildNewSpansFromSelection } from "../annotationSelection";

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
