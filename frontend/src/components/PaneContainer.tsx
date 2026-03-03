import {
  type ReactNode,
  useCallback,
  useRef,
} from "react";

interface Props {
  children: ReactNode;
}

export function useSyncScroll() {
  const panesRef = useRef<(HTMLDivElement | null)[]>([]);
  const scrollingRef = useRef(false);

  const registerPane = useCallback(
    (index: number) => (el: HTMLDivElement | null) => {
      panesRef.current[index] = el;
    },
    [],
  );

  const handleScroll = useCallback(
    (_index: number) => (scrollTop: number) => {
      if (scrollingRef.current) return;
      scrollingRef.current = true;
      for (const pane of panesRef.current) {
        if (pane) {
          pane.scrollTop = scrollTop;
        }
      }
      requestAnimationFrame(() => {
        scrollingRef.current = false;
      });
    },
    [],
  );

  return { registerPane, handleScroll };
}

export default function PaneContainer({ children }: Props) {
  return <div className="pane-container">{children}</div>;
}
