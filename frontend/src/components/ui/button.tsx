import type { ComponentProps } from "react";

const BASE = "inline-flex shrink-0 items-center justify-center gap-2 rounded-md text-sm font-medium whitespace-nowrap transition-colors outline-none disabled:pointer-events-none disabled:opacity-50 focus-visible:ring-3 focus-visible:ring-blue-500/25";
const VARIANTS = {
  default: "bg-[#2356d8] text-white hover:bg-[#173fa8]",
  outline: "border border-slate-300 bg-white text-slate-900 hover:bg-slate-50",
  ghost: "bg-transparent text-inherit hover:bg-slate-100/10",
} as const;
const SIZES = {
  default: "h-10 px-4 py-2",
  sm: "h-9 rounded-md px-3",
  compact: "h-9 rounded-md px-3 text-xs",
} as const;

type ButtonProps = ComponentProps<"button"> & {
  variant?: keyof typeof VARIANTS;
  size?: keyof typeof SIZES;
};

function Button({ className, variant = "default", size = "default", type = "button", ...props }: ButtonProps) {
  return (
    <button
      data-slot="button"
      type={type}
      className={[BASE, VARIANTS[variant], SIZES[size], className].filter(Boolean).join(" ")}
      {...props}
    />
  );
}

export { Button };
