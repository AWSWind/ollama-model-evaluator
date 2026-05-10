import { describe, expect, test, vi, beforeEach, afterEach } from "vitest";
import fc from "fast-check";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { render, screen, fireEvent, waitFor, cleanup } from "@testing-library/react";

import { NewRun } from "./NewRun";
import { ApiError } from "../api/apiClient";
import * as apiClient from "../api/apiClient";

/**
 * Property 36 — UI 400-error field mapping (Task 22.3, Requirement 15.4).
 *
 * Given an envelope ``{error_code: "validation_failed", field,
 * message}`` returned by the Backend, the rendered NewRun view renders
 * ``message`` as a sibling of the input whose ``data-field`` equals
 * ``field``.
 */
describe("NewRun 400 envelope field mapping", () => {
  beforeEach(() => {
    // The models/suites fetches happen on mount; stub them so the form
    // renders deterministically regardless of the property iteration.
    vi.spyOn(apiClient, "listModels").mockResolvedValue([
      { name: "llama3:8b", digest: null, parameter_size: null },
    ]);
    vi.spyOn(apiClient, "listSuites").mockResolvedValue(["reasoning-basics"]);
  });

  afterEach(() => {
    vi.restoreAllMocks();
    cleanup();
  });

  /** Validates: Requirements 15.4. */
  test("renders message next to the data-field input named by field", async () => {
    const fieldArb = fc.constantFrom(
      "models",
      "suites",
      "repetitions",
      "concurrency",
      "tag_filter",
    );
    const messageArb = fc.string({ minLength: 1, maxLength: 40 }).map((s) =>
      // Keep ASCII letters/digits so the innerText assertion is stable.
      s.replace(/[^A-Za-z0-9 ]/g, "").trim() || "invalid",
    );

    await fc.assert(
      fc.asyncProperty(fieldArb, messageArb, async (field, message) => {
        // Stub submitRun to reject with a controlled ApiError on every
        // iteration of the property so the UI surfaces the envelope.
        const submitSpy = vi
          .spyOn(apiClient, "submitRun")
          .mockRejectedValue(
            new ApiError(400, message, {
              error_code: "validation_failed",
              field,
              message,
            }),
          );

        const queryClient = new QueryClient({
          defaultOptions: { queries: { retry: false } },
        });
        const { unmount } = render(
          <QueryClientProvider client={queryClient}>
            <MemoryRouter>
              <NewRun />
            </MemoryRouter>
          </QueryClientProvider>,
        );

        // Wait for queries to resolve so the submit button is enabled.
        await waitFor(() => {
          expect(screen.getByText("Submit Run")).toBeInTheDocument();
        });

        fireEvent.submit(screen.getByTestId("new-run-form"));

        await waitFor(() => {
          const errorEl = document.querySelector(
            `[data-field-error="${field}"]`,
          );
          expect(errorEl).not.toBeNull();
          expect(errorEl?.textContent).toBe(message);
        });

        // Sibling-of-input assertion: the error element must live in
        // the same FieldBlock wrapper as the input with matching
        // ``data-field``.
        const input = document.querySelector(`[data-field="${field}"]`);
        expect(input).not.toBeNull();
        const block = input?.closest(`[data-field-block="${field}"]`);
        expect(block).not.toBeNull();
        expect(
          block!.querySelector(`[data-field-error="${field}"]`),
        ).not.toBeNull();

        submitSpy.mockRestore();
        unmount();
      }),
      { numRuns: 20 },
    );
  });
});
