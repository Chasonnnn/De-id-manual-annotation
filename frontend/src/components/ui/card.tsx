import type { ComponentProps } from "react";
import { cn } from "@/lib/utils";

function Card({ className, ...props }: ComponentProps<"div">) {
  return (
    <div
      data-slot="card"
      className={cn("flex flex-col gap-6 rounded-xl border border-slate-200 bg-white text-slate-950 shadow-sm", className)}
      {...props}
    />
  );
}

export { Card };
