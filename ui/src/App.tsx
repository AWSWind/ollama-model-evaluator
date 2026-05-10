import { Link, NavLink, Navigate, Route, Routes } from "react-router-dom";
import { GitCompare, History as HistoryIcon, PlayCircle } from "lucide-react";

import { NewRun } from "./routes/NewRun";
import { RunDetail } from "./routes/RunDetail";
import { History } from "./routes/History";
import { Compare } from "./routes/Compare";
import { ThemeToggle, cn } from "./ui";

/**
 * Root application shell.
 *
 * Layout: a fixed-width sidebar on the left (logo + primary nav) and
 * a scrolling main column on the right that hosts the route content.
 * Theme is driven by ``<ThemeProvider>`` in :mod:`./main.tsx`.
 */
export function App(): JSX.Element {
  return (
    <div className="min-h-screen grid grid-cols-1 md:grid-cols-[240px_1fr]">
      <aside
        aria-label="Primary"
        data-testid="app-nav"
        className="theme-surface border-b md:border-b-0 md:border-r border-border bg-bg-alt px-4 py-5"
      >
        <Link to="/" className="flex items-center gap-2.5 mb-8 px-2">
          <span
            aria-hidden="true"
            className="h-4 w-4 rounded bg-gradient-to-br from-accent to-info"
          />
          <span className="font-bold text-[0.95rem] tracking-tight text-fg">
            Evaluator
          </span>
        </Link>

        <nav className="flex flex-col gap-0.5">
          <SidebarLink to="/runs/new" icon={<PlayCircle className="h-4 w-4" />}>
            New Run
          </SidebarLink>
          <SidebarLink to="/history" icon={<HistoryIcon className="h-4 w-4" />}>
            History
          </SidebarLink>
          <SidebarLink to="/compare" icon={<GitCompare className="h-4 w-4" />}>
            Compare
          </SidebarLink>
        </nav>
      </aside>

      <main className="min-w-0">
        <header className="flex items-center justify-end gap-2 px-6 md:px-8 py-3 border-b border-border bg-bg">
          <ThemeToggle />
        </header>
        <div className="px-6 md:px-8 py-6 md:py-8 max-w-[1200px]">
          <Routes>
            <Route path="/" element={<Navigate to="/runs/new" replace />} />
            <Route path="/runs/new" element={<NewRun />} />
            <Route path="/runs/:runId" element={<RunDetail />} />
            <Route path="/history" element={<History />} />
            <Route path="/compare" element={<Compare />} />
            <Route
              path="*"
              element={
                <p className="text-fg-muted">
                  Not found. Try{" "}
                  <Link to="/runs/new" className="text-accent hover:underline">
                    New Run
                  </Link>
                  .
                </p>
              }
            />
          </Routes>
        </div>
      </main>
    </div>
  );
}

interface SidebarLinkProps {
  to: string;
  icon: React.ReactNode;
  children: React.ReactNode;
}

function SidebarLink({ to, icon, children }: SidebarLinkProps): JSX.Element {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        cn(
          "flex items-center gap-2.5 px-2.5 py-1.5 rounded-md text-sm",
          "transition-colors",
          isActive
            ? "bg-accent-soft text-accent font-semibold"
            : "text-fg-muted hover:bg-bg hover:text-fg",
        )
      }
    >
      <span aria-hidden="true">{icon}</span>
      {children}
    </NavLink>
  );
}
