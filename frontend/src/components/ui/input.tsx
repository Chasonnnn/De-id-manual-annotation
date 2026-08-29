import type { ComponentProps } from "react";
import { cn } from "@/lib/utils";

function Input({ className, type, ...props }: ComponentProps<"input">) {
  return (
    <input
      data-slot="input"
      type={type}
      className={cn(
        "h-10 w-full min-w-0 rounded-md border border-slate-300 bg-transparent px-3 py-1 text-base outline-none transition-colors placeholder:text-slate-500 disabled:pointer-events-none disabled:opacity-50 focus-visible:border-blue-600 focus-visible:ring-3 focus-visible:ring-blue-500/20 md:text-sm",
        className,
      )}
      {...props}
    />
  );
}

export { Input };
