import { Moon, Sun } from "lucide-react";

import { useTheme } from "../theme";
import { Button } from "./Button";

/**
 * Top-bar affordance for flipping between light and dark. Reads the
 * resolved theme from :func:`useTheme` so the label always reflects
 * the state a user would see *after* clicking — "switch to light" when
 * currently dark, and vice versa — matching conventional usage.
 */
export function ThemeToggle(): JSX.Element {
  const { resolved, toggle } = useTheme();
  const isDark = resolved === "dark";
  return (
    <Button
      variant="ghost"
      size="sm"
      onClick={toggle}
      aria-label={isDark ? "Switch to light theme" : "Switch to dark theme"}
      title={isDark ? "Switch to light theme" : "Switch to dark theme"}
      data-testid="theme-toggle"
    >
      {isDark ? (
        <>
          <Sun className="h-3.5 w-3.5" aria-hidden="true" />
          <span>Light</span>
        </>
      ) : (
        <>
          <Moon className="h-3.5 w-3.5" aria-hidden="true" />
          <span>Dark</span>
        </>
      )}
    </Button>
  );
}
