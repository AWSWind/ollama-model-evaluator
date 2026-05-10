/** @type {import('tailwindcss').Config} */
export default {
  // ``data-theme="dark"`` on <html> toggles the dark variants. Matches
  // the ThemeProvider bootstrap in src/theme.tsx so both CSS variables
  // and Tailwind utility classes switch together.
  darkMode: ["selector", '[data-theme="dark"]'],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      // All colours route through CSS variables declared in styles.css
      // so changing the palette is a one-file edit and the variables
      // are also accessible from plain CSS and inline styles.
      colors: {
        bg: "rgb(var(--bg) / <alpha-value>)",
        "bg-alt": "rgb(var(--bg-alt) / <alpha-value>)",
        fg: "rgb(var(--fg) / <alpha-value>)",
        "fg-muted": "rgb(var(--fg-muted) / <alpha-value>)",
        "fg-subtle": "rgb(var(--fg-subtle) / <alpha-value>)",
        border: "rgb(var(--border) / <alpha-value>)",
        "border-strong": "rgb(var(--border-strong) / <alpha-value>)",
        accent: "rgb(var(--accent) / <alpha-value>)",
        "accent-soft": "rgb(var(--accent-soft) / <alpha-value>)",
        pass: "rgb(var(--pass) / <alpha-value>)",
        "pass-soft": "rgb(var(--pass-soft) / <alpha-value>)",
        fail: "rgb(var(--fail) / <alpha-value>)",
        "fail-soft": "rgb(var(--fail-soft) / <alpha-value>)",
        warn: "rgb(var(--warn) / <alpha-value>)",
        info: "rgb(var(--info) / <alpha-value>)",
        "info-soft": "rgb(var(--info-soft) / <alpha-value>)",
      },
      fontFamily: {
        sans: [
          "Inter",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
        mono: [
          "JetBrains Mono",
          "SF Mono",
          "Consolas",
          "monospace",
        ],
      },
      fontSize: {
        xxs: ["0.72rem", { lineHeight: "1rem" }],
      },
      borderRadius: {
        card: "0.5rem",
      },
      boxShadow: {
        card: "0 1px 2px rgb(0 0 0 / 0.06)",
        "card-dark": "0 1px 2px rgb(0 0 0 / 0.5)",
      },
    },
  },
  plugins: [],
};
