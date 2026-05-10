import { createContext, useCallback, useContext, useEffect, useState } from "react";

/**
 * Theme mode.
 *
 * * ``"light"`` — explicit light
 * * ``"dark"`` — explicit dark
 * * ``"auto"`` — follow ``prefers-color-scheme``
 *
 * ``auto`` is the default on first visit so a user landing from a dark
 * OS immediately sees dark chrome. The user's subsequent toggle choices
 * are persisted to localStorage.
 */
export type ThemeMode = "light" | "dark" | "auto";

const STORAGE_KEY = "ollama-eval-theme";

interface ThemeContextValue {
  mode: ThemeMode;
  /** Resolved theme after accounting for ``auto``. Always ``light`` or ``dark``. */
  resolved: "light" | "dark";
  /** Set mode explicitly (persisted). */
  setMode: (mode: ThemeMode) => void;
  /** Toggle between light and dark, skipping auto. Persists the choice. */
  toggle: () => void;
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

function readStoredMode(): ThemeMode {
  if (typeof window === "undefined") return "auto";
  // Allow ?theme=dark|light|auto in the URL to override (and persist) the
  // stored choice. Handy for sharing dark-mode links and for headless
  // screenshotting tools.
  try {
    const qs = new URLSearchParams(window.location.search);
    const override = qs.get("theme");
    if (override === "dark" || override === "light" || override === "auto") {
      window.localStorage.setItem(STORAGE_KEY, override);
      return override;
    }
  } catch {
    // ignore storage or URL parsing failures
  }
  const raw = window.localStorage.getItem(STORAGE_KEY);
  if (raw === "light" || raw === "dark" || raw === "auto") return raw;
  return "auto";
}

function prefersDark(): boolean {
  if (typeof window === "undefined" || !window.matchMedia) return false;
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}

function applyToDocument(resolved: "light" | "dark"): void {
  const root = document.documentElement;
  if (resolved === "dark") {
    root.setAttribute("data-theme", "dark");
    root.style.colorScheme = "dark";
  } else {
    root.removeAttribute("data-theme");
    root.style.colorScheme = "light";
  }
}

/**
 * Mount once at the app root.
 *
 * Reads stored mode (falling back to ``auto``), applies the matching
 * ``data-theme`` attribute, and listens for OS preference changes while
 * ``mode === "auto"``. Child components can then call
 * :func:`useTheme` for the current resolved theme and a toggle.
 */
export function ThemeProvider({ children }: { children: React.ReactNode }): JSX.Element {
  const [mode, setModeState] = useState<ThemeMode>(() => readStoredMode());
  const [resolved, setResolved] = useState<"light" | "dark">(() => {
    const initial = readStoredMode();
    if (initial === "dark") return "dark";
    if (initial === "light") return "light";
    return prefersDark() ? "dark" : "light";
  });

  // Propagate the resolved theme to the document so CSS variables and
  // Tailwind ``dark:`` variants agree.
  useEffect(() => {
    applyToDocument(resolved);
  }, [resolved]);

  // Re-resolve whenever the user flips mode, or when the OS preference
  // changes while we are on ``auto``. The media-query listener tears
  // itself down when the effect re-runs so there is never more than
  // one subscription.
  useEffect(() => {
    if (mode === "light" || mode === "dark") {
      setResolved(mode);
      return;
    }
    const mql = window.matchMedia("(prefers-color-scheme: dark)");
    setResolved(mql.matches ? "dark" : "light");
    const handler = (e: MediaQueryListEvent): void => {
      setResolved(e.matches ? "dark" : "light");
    };
    mql.addEventListener("change", handler);
    return () => mql.removeEventListener("change", handler);
  }, [mode]);

  const setMode = useCallback((next: ThemeMode): void => {
    setModeState(next);
    try {
      window.localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // ignore storage failures (private-mode etc.)
    }
  }, []);

  const toggle = useCallback((): void => {
    setMode(resolved === "dark" ? "light" : "dark");
  }, [resolved, setMode]);

  const value: ThemeContextValue = { mode, resolved, setMode, toggle };
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext);
  if (!ctx) {
    throw new Error("useTheme() must be used within <ThemeProvider>");
  }
  return ctx;
}
