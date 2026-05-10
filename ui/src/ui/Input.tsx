import { forwardRef, type InputHTMLAttributes, type LabelHTMLAttributes, type ReactNode } from "react";

import { cn } from "./cn";

/** Tailwind classes shared by text and number inputs. */
export const inputBaseClasses = cn(
  "block w-full rounded-md border border-border bg-bg px-3 py-1.5",
  "text-sm text-fg placeholder:text-fg-subtle",
  "focus-visible:outline-none focus-visible:border-accent focus-visible:ring-2 focus-visible:ring-accent/25",
  "disabled:opacity-60 disabled:cursor-not-allowed",
);

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(
  function Input({ className, ...rest }, ref) {
    return <input ref={ref} className={cn(inputBaseClasses, className)} {...rest} />;
  },
);

export function Label({
  className,
  children,
  ...rest
}: LabelHTMLAttributes<HTMLLabelElement> & { children: ReactNode }): JSX.Element {
  return (
    <label
      className={cn(
        "block text-[0.72rem] font-semibold uppercase tracking-wider",
        "text-fg-muted mb-1",
        className,
      )}
      {...rest}
    >
      {children}
    </label>
  );
}
