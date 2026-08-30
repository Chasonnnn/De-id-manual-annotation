import type { ComponentProps } from "react";

function Card({ className, ...props }: ComponentProps<"div">) {
  return (
    <div
      data-slot="card"
      className={["flex flex-col gap-6 rounded-xl border border-slate-200 bg-white text-slate-950 shadow-sm", className].filter(Boolean).join(" ")}
      {...props}
    />
  );
}

export { Card };
