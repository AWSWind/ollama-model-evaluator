import { Link, Navigate, Route, Routes } from "react-router-dom";

import { NewRun } from "./routes/NewRun";
import { RunDetail } from "./routes/RunDetail";
import { History } from "./routes/History";
import { Compare } from "./routes/Compare";

/**
 * Root application shell.
 *
 * Provides top-level navigation to each UI route:
 *
 * - ``/runs/new`` — submit a new Run (Requirement 15.3).
 * - ``/history`` — browse historical Run_Reports (Requirement 16.1).
 * - ``/runs/:id`` — live Run-detail view (Requirement 15.5–15.10).
 * - ``/compare`` — side-by-side diff of two Runs (Requirement 16.4).
 */
export function App(): JSX.Element {
  return (
    <div>
      <nav aria-label="Primary" data-testid="app-nav">
        <ul style={{ display: "flex", gap: "1rem", listStyle: "none", padding: "0.5rem" }}>
          <li>
            <Link to="/runs/new">New Run</Link>
          </li>
          <li>
            <Link to="/history">History</Link>
          </li>
        </ul>
      </nav>
      <main>
        <Routes>
          <Route path="/" element={<Navigate to="/runs/new" replace />} />
          <Route path="/runs/new" element={<NewRun />} />
          <Route path="/runs/:runId" element={<RunDetail />} />
          <Route path="/history" element={<History />} />
          <Route path="/compare" element={<Compare />} />
          <Route
            path="*"
            element={<p>Not found. Try <Link to="/runs/new">New Run</Link>.</p>}
          />
        </Routes>
      </main>
    </div>
  );
}
