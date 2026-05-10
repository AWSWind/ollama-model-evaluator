import { type ReactNode } from "react";

import { cn } from "./cn";

type Tone = "pass" | "fail" | "running" | "warn" | "neutral";

export interface PillProps {
  tone?: Tone;
  className?: string;
  children: ReactNode;
  dot?: boolean;
}

const toneClasses: Record<Tone, string> = {
  pass: "bg-pass-soft text-pass",
  fail: "bg-fail-soft text-fail",
  running: "bg-info-soft text-info",
  warn: "bg-warn/15 text-warn",
  neutral: "bg-bg-alt text-fg-muted border border-border",
};

/**
 * Small status badge; use for Run/TestCase status, pass/fail verdicts,
 * and live-indicator labels. ``dot`` adds a leading coloured dot that
 * (when ``tone="running"``) pulses subtly via a CSS animation, matching
 * the event-stream "connected" indicator aesthetic.
 */
export function Pill({ tone = "neutral", className, children, dot = false }: PillProps): JSX.Element {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded px-2 py-0.5 text-xxs font-semibold",
        toneClasses[tone],
        className,
      )}
    >
      {dot ? (
        <span
          aria-hidden="true"
          className={cn(
            "h-1.5 w-1.5 rounded-full bg-current",
            tone === "running" && "animate-pulse",
          )}
        />
      ) : null}
      {children}
    </span>
  );
}
