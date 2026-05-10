import { forwardRef, type HTMLAttributes, type TableHTMLAttributes } from "react";

import { cn } from "./cn";

export function Table({
  className,
  ...rest
}: TableHTMLAttributes<HTMLTableElement>): JSX.Element {
  return (
    <div className="overflow-x-auto">
      <table
        className={cn("w-full border-collapse text-sm", className)}
        {...rest}
      />
    </div>
  );
}

export const Thead = forwardRef<HTMLTableSectionElement, HTMLAttributes<HTMLTableSectionElement>>(
  function Thead({ className, ...rest }, ref) {
    return (
      <thead
        ref={ref}
        className={cn(
          "bg-bg-alt text-fg-muted",
          "[&_th]:text-[0.72rem] [&_th]:uppercase [&_th]:tracking-wider [&_th]:font-semibold",
          "[&_th]:px-3 [&_th]:py-2 [&_th]:text-left [&_th]:border-b [&_th]:border-border",
          className,
        )}
        {...rest}
      />
    );
  },
);

export const Tbody = forwardRef<HTMLTableSectionElement, HTMLAttributes<HTMLTableSectionElement>>(
  function Tbody({ className, ...rest }, ref) {
    return (
      <tbody
        ref={ref}
        className={cn(
          "[&_tr:hover]:bg-bg-alt",
          "[&_td]:px-3 [&_td]:py-2.5 [&_td]:border-b [&_td]:border-border",
          className,
        )}
        {...rest}
      />
    );
  },
);
