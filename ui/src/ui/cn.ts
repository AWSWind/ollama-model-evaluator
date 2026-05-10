import clsx, { type ClassValue } from "clsx";

/**
 * Thin alias over :func:`clsx` so every UI primitive imports the same
 * class-merging helper. Prefer ``cn`` over inline template literals
 * whenever you need conditional classes — the imports stay consistent
 * and future dedupe work (e.g. if we ever add ``tailwind-merge``) is a
 * one-file change.
 */
export function cn(...values: ClassValue[]): string {
  return clsx(...values);
}
