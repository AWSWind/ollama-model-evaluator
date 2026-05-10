import { cn } from "./cn";

export interface ProgressProps {
  /** Percentage complete, 0..100. Values outside the range are clamped. */
  value: number;
  className?: string;
  "aria-label"?: string;
}

/**
 * Horizontal progress bar. Purely visual — the caller supplies the
 * ``aria-label`` and any surrounding counters. The fill animates via
 * a CSS transition so rapid event-stream updates don't flicker.
 */
export function Progress({ value, className, "aria-label": ariaLabel }: ProgressProps): JSX.Element {
  const safe = Math.max(0, Math.min(100, value));
  return (
    <div
      className={cn(
        "h-1.5 rounded-full bg-border overflow-hidden",
        className,
      )}
      role="progressbar"
      aria-valuenow={Math.round(safe)}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label={ariaLabel}
    >
      <div
        className="h-full bg-accent transition-[width] duration-300 ease-out"
        style={{ width: `${safe}%` }}
      />
    </div>
  );
}
