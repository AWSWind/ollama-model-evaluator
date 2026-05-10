import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import { App } from "./App";

/**
 * Entry point for the Ollama Model Evaluator UI.
 *
 * Wraps the {@link App} component tree in a {@link QueryClientProvider}
 * (TanStack Query powers REST caching per design §UI state) and a
 * {@link BrowserRouter} so the route components can use
 * `react-router-dom` hooks such as `useNavigate`, `useParams`, and
 * `useSearchParams`.
 */

// Single QueryClient for the whole app. We keep caches hot for the
// session; aggressive refetch-on-focus is not desirable because runs
// emit events via WebSocket and would otherwise refetch redundantly.
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

const container = document.getElementById("root");
if (!container) {
  throw new Error("Root element #root not found in index.html");
}

createRoot(container).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
);
