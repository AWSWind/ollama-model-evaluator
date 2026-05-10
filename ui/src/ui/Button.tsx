import { forwardRef, type ButtonHTMLAttributes } from "react";

import { cn } from "./cn";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "sm" | "md";

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
}

const variantClasses: Record<Variant, string> = {
  primary:
    "bg-accent text-white border-accent hover:brightness-110 " +
    "shadow-sm shadow-accent/30",
  secondary:
    "bg-bg text-fg border-border hover:border-border-strong hover:bg-bg-alt",
  ghost:
    "bg-transparent text-fg-muted border-transparent hover:bg-bg-alt hover:text-fg",
  danger:
    "bg-fail text-white border-fail hover:brightness-110",
};

const sizeClasses: Record<Size, string> = {
  sm: "text-xs px-2.5 py-1",
  md: "text-sm px-3 py-1.5",
};

/**
 * Base button primitive. Renders a real ``<button>`` so keyboard and
 * screen-reader semantics stay correct; colours/padding come from the
 * design tokens. Wrap any ``<a>``-styled action in
 * :component:`LinkButton` (below) when you want navigation instead.
 */
export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { className, variant = "secondary", size = "md", type = "button", ...rest },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-md",
        "border font-medium transition-colors",
        "disabled:opacity-60 disabled:cursor-not-allowed",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/50",
        variantClasses[variant],
        sizeClasses[size],
        className,
      )}
      {...rest}
    />
  );
});
