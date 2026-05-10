import { type ReactNode } from "react";
import { X } from "lucide-react";

import { cn } from "./cn";

export interface ChipProps {
  children: ReactNode;
  onRemove?: () => void;
  variant?: "accent" | "ghost" | "neutral";
  className?: string;
  title?: string;
}

const variantClasses: Record<NonNullable<ChipProps["variant"]>, string> = {
  accent:
    "bg-accent-soft text-accent border-transparent",
  neutral:
    "bg-bg-alt text-fg-muted border-border",
  ghost:
    "bg-transparent text-fg-subtle border-dashed border-border-strong",
};

/**
 * Pill-shaped tag or selection indicator.
 *
 * Pass ``onRemove`` to render a trailing ``×`` affordance (common for
 * selected-item chips in a multi-select). Pass ``variant="ghost"`` for
 * the "+ add" placeholder style used when inviting an action.
 */
export function Chip({
  children,
  onRemove,
  variant = "accent",
  className,
  title,
}: ChipProps): JSX.Element {
  return (
    <span
      title={title}
      className={cn(
        "inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full",
        "text-xs font-medium border",
        variantClasses[variant],
        className,
      )}
    >
      {children}
      {onRemove ? (
        <button
          type="button"
          aria-label="Remove"
          onClick={onRemove}
          className="ml-0.5 -mr-0.5 p-0.5 rounded hover:bg-black/10 dark:hover:bg-white/10 opacity-70 hover:opacity-100 transition"
        >
          <X className="h-3 w-3" aria-hidden="true" />
        </button>
      ) : null}
    </span>
  );
}
