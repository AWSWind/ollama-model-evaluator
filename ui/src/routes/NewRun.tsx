import { useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";

import {
  ApiError,
  listModels,
  listSuiteSummaries,
  submitRun,
  type ModelInfo,
  type RunConfig,
  type SuiteSummary,
} from "../api/apiClient";
import { errorMessages } from "../api/errorMessages";

/**
 * Input shape collected from the NewRun form.
 *
 * Kept separate from {@link RunConfig} so the property test for
 * "form → body" (Property 35) can drive a form-state generator without
 * regenerating the full API body.
 */
export interface NewRunFormSelections {
  models: string[];
  suites: string[];
  repetitions: number;
  concurrency: number;
  /** Comma-separated tags string as typed by the user. */
  tagFilter: string;
}

/**
 * Pure helper exported for Property 35 and exercised by the form
 * submit handler. Builds the exact :class:`RunConfig` body the UI
 * POSTs to ``/api/runs`` from a {@link NewRunFormSelections}.
 *
 * The body is a plain object with the same fields as RunConfig and no
 * additional fields. ``tag_filter`` is the parsed version of the
 * comma-separated string; blank tokens are dropped.
 */
export function buildRunConfigBody(
  selections: NewRunFormSelections,
): RunConfig {
  const tags = selections.tagFilter
    .split(",")
    .map((t) => t.trim())
    .filter((t) => t.length > 0);
  return {
    models: [...selections.models],
    suites: [...selections.suites],
    repetitions: selections.repetitions,
    concurrency: selections.concurrency,
    tag_filter: tags,
  };
}

function toSelectedValues(event: React.ChangeEvent<HTMLSelectElement>): string[] {
  return Array.from(event.target.selectedOptions, (o) => o.value);
}

/**
 * Empirical seconds-per-case upper-bound observed on the `.224` host
 * (``qwen3.6:27b`` at ~11 tok/s). Used as the default coefficient in
 * {@link estimateRunSeconds}. This is a rough rule of thumb, not a
 * measurement — real-world cases vary widely by prompt length and
 * response length. The UI's "est. time" line is intentionally labelled
 * "rough" so users know not to take it as a guarantee.
 */
const DEFAULT_SECONDS_PER_CASE = 15;

/**
 * Compute an estimated wall-clock duration (in seconds) for a run.
 *
 * Formula: ``ceil(|M| * sum(cases) * R / C) * seconds_per_case``.
 *
 * * ``M`` — number of models selected
 * * ``sum(cases)`` — total test cases across every selected suite
 * * ``R`` — repetitions
 * * ``C`` — concurrency (clamped to ``max(1, C)``)
 *
 * Exported for the UI property tests so the estimate is deterministic
 * and easy to assert against.
 */
export function estimateRunSeconds(
  modelCount: number,
  totalCases: number,
  repetitions: number,
  concurrency: number,
  secondsPerCase: number = DEFAULT_SECONDS_PER_CASE,
): number {
  if (
    modelCount <= 0 ||
    totalCases <= 0 ||
    repetitions <= 0 ||
    secondsPerCase <= 0
  ) {
    return 0;
  }
  const safeConcurrency = Math.max(1, concurrency);
  const serialCalls = modelCount * totalCases * repetitions;
  const effectiveCalls = Math.ceil(serialCalls / safeConcurrency);
  return effectiveCalls * secondsPerCase;
}

/**
 * Format a duration in seconds as a short human-readable string.
 *
 * * ``<60s`` → ``"45s"``
 * * ``<3600s`` → ``"12m 30s"`` (whole seconds dropped if zero)
 * * else → ``"1h 24m"`` (seconds dropped above the hour)
 */
export function formatDuration(totalSeconds: number): string {
  if (!Number.isFinite(totalSeconds) || totalSeconds <= 0) {
    return "0s";
  }
  const rounded = Math.round(totalSeconds);
  if (rounded < 60) {
    return `${rounded}s`;
  }
  if (rounded < 3600) {
    const m = Math.floor(rounded / 60);
    const s = rounded % 60;
    return s === 0 ? `${m}m` : `${m}m ${s}s`;
  }
  const h = Math.floor(rounded / 3600);
  const m = Math.round((rounded % 3600) / 60);
  return m === 0 ? `${h}h` : `${h}h ${m}m`;
}

/**
 * Per-suite metadata displayed alongside each option.
 *
 * ``caseCount`` is ``null`` while the summaries fetch is in flight or
 * if it failed — we still render the suite name so the user can select
 * it, but the count + estimate line stays as "…" until the fetch
 * resolves.
 */
interface SuiteMeta {
  name: string;
  caseCount: number | null;
  description: string | null;
}

/**
 * NewRun route (Requirement 15.3).
 *
 * Fetches models and suites, lets the user assemble a
 * :class:`RunConfig`, POSTs it, and navigates to ``/runs/:id`` on
 * success. Validation failures surface next to the offending input
 * using ``data-field`` on the inputs so the 400-error property test
 * (Property 36) can locate the rendered message.
 *
 * Additionally enriches the Suites multi-select with per-suite case
 * counts and a live "total est. time" readout so users know how big
 * the run will be before they submit. Suite metadata comes from the
 * bulk ``GET /api/suites/summaries`` endpoint so a cold page load
 * issues only two small requests (``models`` + ``suite summaries``)
 * instead of ``N+1`` where ``N`` is the number of suites.
 */
export function NewRun(): JSX.Element {
  const navigate = useNavigate();

  const modelsQuery = useQuery({
    queryKey: ["models"],
    queryFn: listModels,
  });

  // Single bulk fetch that returns name + test_case_count + description
  // for every suite. Roughly a 100× payload reduction compared with
  // fetching the full EvaluationSuite for each name individually.
  const suitesSummaryQuery = useQuery({
    queryKey: ["suite-summaries"],
    queryFn: listSuiteSummaries,
    staleTime: 60 * 1000,
  });

  const suiteSummaries: SuiteSummary[] = useMemo(
    () => suitesSummaryQuery.data ?? [],
    [suitesSummaryQuery.data],
  );
  const suiteMetaByName: Record<string, SuiteMeta> = useMemo(() => {
    const map: Record<string, SuiteMeta> = {};
    for (const s of suiteSummaries) {
      map[s.name] = {
        name: s.name,
        caseCount: s.test_case_count,
        description: s.description ?? null,
      };
    }
    return map;
  }, [suiteSummaries]);
  const suiteOptions: string[] = useMemo(
    () => suiteSummaries.map((s) => s.name),
    [suiteSummaries],
  );

  const [selections, setSelections] = useState<NewRunFormSelections>({
    models: [],
    suites: [],
    repetitions: 1,
    concurrency: 1,
    tagFilter: "",
  });

  const [fieldError, setFieldError] = useState<{
    field: string | null;
    message: string;
  } | null>(null);

  const submitMutation = useMutation({
    mutationFn: async (config: RunConfig) => submitRun(config),
    onSuccess: (response) => {
      navigate(`/runs/${response.run_id}`);
    },
    onError: (error) => {
      if (error instanceof ApiError && error.envelope) {
        const env = error.envelope;
        // Prefer the Backend's specific message; fall back to the
        // canned copy in errorMessages if the Backend omitted one.
        const message = env.message || errorMessages[env.error_code];
        setFieldError({ field: env.field ?? null, message });
      } else {
        setFieldError({
          field: null,
          message: (error as Error).message || "Submission failed",
        });
      }
    },
  });

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>): void => {
    event.preventDefault();
    setFieldError(null);
    const body = buildRunConfigBody(selections);
    submitMutation.mutate(body);
  };

  const modelOptions: ModelInfo[] = useMemo(
    () => modelsQuery.data ?? [],
    [modelsQuery.data],
  );

  // ---- Live totals & estimates --------------------------------------
  // Only count a suite's cases once its details have arrived; if the
  // detail fetch failed or is still in flight, treat the count as 0 so
  // the estimate stays conservative rather than suggesting an
  // arbitrarily-low time.
  const selectedCasesTotal = selections.suites.reduce((sum, name) => {
    const meta = suiteMetaByName[name];
    return sum + (meta?.caseCount ?? 0);
  }, 0);
  const estimatedSeconds = estimateRunSeconds(
    Math.max(1, selections.models.length),
    selectedCasesTotal,
    selections.repetitions,
    selections.concurrency,
  );
  const anySuiteStillLoading = selections.suites.some(
    (name) => suiteMetaByName[name]?.caseCount == null,
  );

  return (
    <section aria-labelledby="new-run-heading">
      <h2 id="new-run-heading">New Run</h2>
      {modelsQuery.isLoading ? <p>Loading models…</p> : null}
      {suitesSummaryQuery.isLoading ? <p>Loading suites…</p> : null}
      <form onSubmit={handleSubmit} data-testid="new-run-form">
        <FieldBlock label="Models" htmlFor="models" field="models" error={fieldError}>
          <select
            id="models"
            multiple
            data-field="models"
            value={selections.models}
            onChange={(e) =>
              setSelections((prev) => ({ ...prev, models: toSelectedValues(e) }))
            }
          >
            {modelOptions.map((m) => (
              <option key={m.name} value={m.name}>
                {m.name}
                {m.parameter_size ? `  (${m.parameter_size})` : ""}
              </option>
            ))}
          </select>
        </FieldBlock>

        <FieldBlock label="Suites" htmlFor="suites" field="suites" error={fieldError}>
          <select
            id="suites"
            multiple
            data-field="suites"
            size={Math.min(12, Math.max(4, suiteOptions.length))}
            value={selections.suites}
            onChange={(e) =>
              setSelections((prev) => ({ ...prev, suites: toSelectedValues(e) }))
            }
          >
            {suiteOptions.map((name) => {
              const meta = suiteMetaByName[name];
              const count = meta?.caseCount;
              const perSuiteEstimate =
                count !== null && count !== undefined
                  ? estimateRunSeconds(
                      Math.max(1, selections.models.length),
                      count,
                      selections.repetitions,
                      selections.concurrency,
                    )
                  : null;
              const label =
                count === null || count === undefined
                  ? `${name}  — loading…`
                  : `${name}  — ${count} case${count === 1 ? "" : "s"} · ~${formatDuration(perSuiteEstimate ?? 0)}`;
              return (
                <option
                  key={name}
                  value={name}
                  title={meta?.description ?? ""}
                >
                  {label}
                </option>
              );
            })}
          </select>
          <p
            data-testid="suite-totals"
            style={{
              margin: "0.25rem 0 0 0",
              fontSize: "0.9em",
              color: "#555",
            }}
          >
            {selections.suites.length === 0 ? (
              <>Select one or more suites to see the total.</>
            ) : (
              <>
                Total: <strong>{selectedCasesTotal}</strong> case
                {selectedCasesTotal === 1 ? "" : "s"} across{" "}
                <strong>{selections.suites.length}</strong> suite
                {selections.suites.length === 1 ? "" : "s"} · rough est.{" "}
                <strong>~{formatDuration(estimatedSeconds)}</strong>
                {selections.models.length > 1 ? (
                  <> for {selections.models.length} models</>
                ) : null}
                {selections.repetitions > 1 ? (
                  <> × {selections.repetitions} rep
                    {selections.repetitions === 1 ? "" : "s"}</>
                ) : null}
                {anySuiteStillLoading ? " (refreshing…)" : ""}
              </>
            )}
          </p>
          {selections.suites.length > 0 ? (
            <ul
              data-testid="suite-descriptions"
              style={{
                margin: "0.5rem 0 0 0",
                paddingLeft: "1.25rem",
                fontSize: "0.85em",
                color: "#666",
              }}
            >
              {selections.suites.map((name) => {
                const desc = suiteMetaByName[name]?.description;
                if (!desc) return null;
                return (
                  <li key={name}>
                    <strong>{name}</strong> — {desc}
                  </li>
                );
              })}
            </ul>
          ) : null}
        </FieldBlock>

        <FieldBlock
          label="Repetitions"
          htmlFor="repetitions"
          field="repetitions"
          error={fieldError}
        >
          <input
            id="repetitions"
            type="number"
            min={1}
            data-field="repetitions"
            value={selections.repetitions}
            onChange={(e) =>
              setSelections((prev) => ({
                ...prev,
                repetitions: Number(e.target.value),
              }))
            }
          />
        </FieldBlock>

        <FieldBlock
          label="Concurrency"
          htmlFor="concurrency"
          field="concurrency"
          error={fieldError}
        >
          <input
            id="concurrency"
            type="number"
            min={1}
            data-field="concurrency"
            value={selections.concurrency}
            onChange={(e) =>
              setSelections((prev) => ({
                ...prev,
                concurrency: Number(e.target.value),
              }))
            }
          />
        </FieldBlock>

        <FieldBlock
          label="Tag filter (comma-separated)"
          htmlFor="tag_filter"
          field="tag_filter"
          error={fieldError}
        >
          <input
            id="tag_filter"
            type="text"
            data-field="tag_filter"
            value={selections.tagFilter}
            onChange={(e) =>
              setSelections((prev) => ({ ...prev, tagFilter: e.target.value }))
            }
          />
        </FieldBlock>

        <button type="submit" disabled={submitMutation.isPending}>
          {submitMutation.isPending ? "Submitting…" : "Submit Run"}
        </button>

        {fieldError && fieldError.field === null ? (
          <p role="alert" data-testid="form-error">
            {fieldError.message}
          </p>
        ) : null}
      </form>
    </section>
  );
}

interface FieldBlockProps {
  label: string;
  htmlFor: string;
  field: string;
  error: { field: string | null; message: string } | null;
  children: React.ReactNode;
}

/**
 * Wrap each field so the validation-error message can be rendered as a
 * sibling of the input. Property 36 asserts that the message appears
 * next to the input with matching ``data-field`` — this block is the
 * shape the property test walks.
 */
function FieldBlock({
  label,
  htmlFor,
  field,
  error,
  children,
}: FieldBlockProps): JSX.Element {
  const showError = error !== null && error.field === field;
  return (
    <div data-field-block={field}>
      <label htmlFor={htmlFor}>{label}</label>
      {children}
      {showError ? (
        <span data-field-error={field} role="alert">
          {error.message}
        </span>
      ) : null}
    </div>
  );
}
