import * as TooltipPrimitive from "@radix-ui/react-tooltip";
import { type ReactNode } from "react";

import { cn } from "./cn";

/**
 * Thin wrapper around ``@radix-ui/react-tooltip``.
 *
 * Wraps the provider at render time so callers don't need to thread
 * one through the tree; the provider is a no-op when no tooltips are
 * mounted. Styling matches the design tokens (bg, border, fg, tiny
 * shadow) and the content stays readable in both light and dark.
 */
export interface TooltipProps {
  /** Text (or any node) shown on hover / focus. */
  content: ReactNode;
  /** The element the tooltip decorates. */
  children: ReactNode;
  /** Delay before showing, in ms. Defaults to 200ms for near-instant feel. */
  delayMs?: number;
  /** Placement relative to the trigger. */
  side?: "top" | "right" | "bottom" | "left";
  /** Optional class on the content panel. */
  className?: string;
}

export function Tooltip({
  content,
  children,
  delayMs = 200,
  side = "top",
  className,
}: TooltipProps): JSX.Element {
  return (
    <TooltipPrimitive.Provider delayDuration={delayMs} skipDelayDuration={100}>
      <TooltipPrimitive.Root>
        <TooltipPrimitive.Trigger asChild>{children}</TooltipPrimitive.Trigger>
        <TooltipPrimitive.Portal>
          <TooltipPrimitive.Content
            side={side}
            sideOffset={6}
            className={cn(
              "z-50 max-w-xs rounded-md border border-border bg-bg px-3 py-2",
              "text-xs leading-relaxed text-fg shadow-card dark:shadow-card-dark",
              "animate-in fade-in-0 zoom-in-95 duration-150",
              "data-[side=top]:slide-in-from-bottom-1",
              "data-[side=bottom]:slide-in-from-top-1",
              "data-[side=left]:slide-in-from-right-1",
              "data-[side=right]:slide-in-from-left-1",
              className,
            )}
          >
            {content}
            <TooltipPrimitive.Arrow className="fill-border" />
          </TooltipPrimitive.Content>
        </TooltipPrimitive.Portal>
      </TooltipPrimitive.Root>
    </TooltipPrimitive.Provider>
  );
}
