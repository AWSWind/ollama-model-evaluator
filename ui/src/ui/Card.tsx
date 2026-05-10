import { forwardRef, type HTMLAttributes } from "react";

import { cn } from "./cn";

/**
 * Framed container primitive used across the app.
 *
 * Intentionally minimal — just the card chrome (border + rounded +
 * padding + background + shadow). Content is whatever the caller puts
 * inside. Exposes :component:`CardHeader`, :component:`CardTitle`,
 * :component:`CardHint` sub-primitives for the common header pattern
 * so tables/forms/charts all start with the same visual rhythm.
 */
export const Card = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(
  function Card({ className, ...rest }, ref) {
    return (
      <div
        ref={ref}
        className={cn(
          "theme-surface rounded-card border border-border bg-bg",
          "p-5 md:p-6 mb-4 shadow-card dark:shadow-card-dark",
          className,
        )}
        {...rest}
      />
    );
  },
);

export function CardHeader({ className, ...rest }: HTMLAttributes<HTMLDivElement>): JSX.Element {
  return <div className={cn("mb-3", className)} {...rest} />;
}

export function CardTitle({ className, ...rest }: HTMLAttributes<HTMLHeadingElement>): JSX.Element {
  return (
    <h3
      className={cn(
        "text-[0.95rem] font-semibold tracking-tight text-fg m-0",
        className,
      )}
      {...rest}
    />
  );
}

export function CardHint({ className, ...rest }: HTMLAttributes<HTMLParagraphElement>): JSX.Element {
  return (
    <p
      className={cn("text-xs text-fg-muted mt-1 mb-0", className)}
      {...rest}
    />
  );
}
