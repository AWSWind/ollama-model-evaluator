import { useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { ArrowRight, Cpu, Layers, Plus, Sliders } from "lucide-react";

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
import {
  Button,
  Card,
  CardHeader,
  CardHint,
  CardTitle,
  Chip,
  Input,
  Label,
  Tooltip,
  cn,
} from "../ui";
import {
  CATEGORIES,
  categoryForSuite,
  summaryForSuite,
} from "./suiteCategories";

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

/**
 * Empirical seconds-per-case upper-bound observed on the benchmark
 * host (``qwen3.6:27b`` at ~11 tok/s). Used as the default coefficient
 * in {@link estimateRunSeconds}. This is a rough rule of thumb, not a
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

/** Format a duration in seconds as ``"45s"`` / ``"12m 30s"`` / ``"1h 24m"``. */
export function formatDuration(totalSeconds: number): string {
  if (!Number.isFinite(totalSeconds) || totalSeconds <= 0) return "0s";
  const rounded = Math.round(totalSeconds);
  if (rounded < 60) return `${rounded}s`;
  if (rounded < 3600) {
    const m = Math.floor(rounded / 60);
    const s = rounded % 60;
    return s === 0 ? `${m}m` : `${m}m ${s}s`;
  }
  const h = Math.floor(rounded / 3600);
  const m = Math.round((rounded % 3600) / 60);
  return m === 0 ? `${h}h` : `${h}h ${m}m`;
}

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
 * Suite metadata comes from the bulk ``GET /api/suites/summaries``
 * endpoint so a cold page load issues only two small requests rather
 * than N+1.
 */
export function NewRun(): JSX.Element {
  const navigate = useNavigate();

  const modelsQuery = useQuery({
    queryKey: ["models"],
    queryFn: listModels,
  });
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
  const suiteNames: string[] = useMemo(
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

  const toggleModel = (name: string): void =>
    setSelections((prev) => ({
      ...prev,
      models: prev.models.includes(name)
        ? prev.models.filter((m) => m !== name)
        : [...prev.models, name],
    }));

  const toggleSuite = (name: string): void =>
    setSelections((prev) => ({
      ...prev,
      suites: prev.suites.includes(name)
        ? prev.suites.filter((s) => s !== name)
        : [...prev.suites, name],
    }));

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

  const selectedModelCount = selections.models.length;
  const unselectedModels = modelOptions.filter(
    (m) => !selections.models.includes(m.name),
  );
  const unselectedSuites = suiteNames.filter(
    (s) => !selections.suites.includes(s),
  );

  return (
    <section aria-labelledby="new-run-heading">
      <h1
        id="new-run-heading"
        className="text-[1.4rem] font-semibold tracking-tight mb-1"
      >
        New Run
      </h1>
      <p className="text-sm text-fg-muted mb-6">
        Pick models and suites, kick off a benchmark. Estimated time updates
        live.
      </p>

      {modelsQuery.isLoading ? (
        <p className="text-fg-muted text-sm">Loading models…</p>
      ) : null}
      {suitesSummaryQuery.isLoading ? (
        <p className="text-fg-muted text-sm">Loading suites…</p>
      ) : null}

      <form onSubmit={handleSubmit} data-testid="new-run-form">
        {/* ---------- Models ---------- */}
        <Card data-field-block="models">
          <CardHeader>
            <div className="flex items-center gap-2">
              <Cpu className="h-4 w-4 text-accent" aria-hidden="true" />
              <CardTitle>Models</CardTitle>
            </div>
            <CardHint>
              Live from Ollama · {modelOptions.length}{" "}
              {modelOptions.length === 1 ? "available" : "available"}
            </CardHint>
          </CardHeader>

          {/* Hidden native select drives Property 35/36 (data-field) */}
          <select
            id="models"
            multiple
            className="sr-only"
            aria-hidden="true"
            tabIndex={-1}
            data-field="models"
            value={selections.models}
            onChange={() => {
              // Controlled by the chips; no-op here keeps React happy.
            }}
          >
            {modelOptions.map((m) => (
              <option key={m.name} value={m.name}>
                {m.name}
              </option>
            ))}
          </select>

          <div className="flex flex-wrap gap-2">
            {selections.models.length === 0 ? (
              <span className="text-xs text-fg-subtle italic">
                No models selected yet.
              </span>
            ) : (
              selections.models.map((name) => {
                const meta = modelOptions.find((m) => m.name === name);
                const chip = (
                  <Chip
                    variant="accent"
                    onRemove={() => toggleModel(name)}
                  >
                    {name}
                    {meta?.parameter_size ? (
                      <span className="opacity-70 ml-1">
                        · {meta.parameter_size}
                      </span>
                    ) : null}
                  </Chip>
                );
                const tooltipContent = meta ? (
                  <div className="space-y-1">
                    <p className="font-semibold">{meta.name}</p>
                    {meta.parameter_size ? (
                      <p>Parameters: {meta.parameter_size}</p>
                    ) : null}
                    {renderExtraModelInfo(meta)}
                  </div>
                ) : null;
                if (!tooltipContent) {
                  return <span key={name}>{chip}</span>;
                }
                return (
                  <Tooltip key={name} content={tooltipContent}>
                    <span>{chip}</span>
                  </Tooltip>
                );
              })
            )}
          </div>

          {unselectedModels.length > 0 ? (
            <div className="mt-4 border-t border-border pt-3">
              <p className="text-xxs uppercase tracking-wider text-fg-muted font-semibold mb-2">
                Available
              </p>
              <div className="flex flex-wrap gap-1.5">
                {unselectedModels.map((m) => {
                  const tooltipContent = (
                    <div className="space-y-1">
                      <p className="font-semibold">{m.name}</p>
                      {m.parameter_size ? (
                        <p>Parameters: {m.parameter_size}</p>
                      ) : null}
                      {renderExtraModelInfo(m)}
                    </div>
                  );
                  const btn = (
                    <button
                      type="button"
                      onClick={() => toggleModel(m.name)}
                      className={cn(
                        "inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full",
                        "text-xs border border-dashed border-border-strong text-fg-muted",
                        "hover:border-accent hover:text-accent hover:bg-accent-soft transition-colors",
                      )}
                    >
                      <Plus className="h-3 w-3" aria-hidden="true" />
                      {m.name}
                      {m.parameter_size ? (
                        <span className="opacity-70">
                          · {m.parameter_size}
                        </span>
                      ) : null}
                    </button>
                  );
                  return (
                    <Tooltip key={m.name} content={tooltipContent}>
                      <span>{btn}</span>
                    </Tooltip>
                  );
                })}
              </div>
            </div>
          ) : null}

          {fieldError && fieldError.field === "models" ? (
            <p
              role="alert"
              data-field-error="models"
              className="mt-3 text-xs text-fail"
            >
              {fieldError.message}
            </p>
          ) : null}
        </Card>

        {/* ---------- Suites ---------- */}
        <Card data-field-block="suites">
          <CardHeader>
            <div className="flex items-center gap-2">
              <Layers className="h-4 w-4 text-accent" aria-hidden="true" />
              <CardTitle>Suites</CardTitle>
            </div>
            <CardHint>
              {suiteNames.length} suites · {totalCasesAcrossAllSuites(suiteMetaByName)} cases total
            </CardHint>
          </CardHeader>

          <select
            id="suites"
            multiple
            className="sr-only"
            aria-hidden="true"
            tabIndex={-1}
            data-field="suites"
            value={selections.suites}
            onChange={() => {
              // Controlled by the buttons below.
            }}
          >
            {suiteNames.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>

          {/* Selected suites — chip row at the top. Removed with × */}
          {selections.suites.length > 0 ? (
            <div className="mb-4">
              <p className="text-xxs uppercase tracking-wider text-fg-muted font-semibold mb-2">
                Selected
              </p>
              <div className="flex flex-wrap gap-2">
                {selections.suites.map((name) => {
                  const meta = suiteMetaByName[name];
                  const count = meta?.caseCount;
                  const perSuiteEstimate =
                    count !== null && count !== undefined
                      ? estimateRunSeconds(
                          Math.max(1, selectedModelCount),
                          count,
                          selections.repetitions,
                          selections.concurrency,
                        )
                      : null;
                  const label =
                    count === null || count === undefined
                      ? `${name} · loading…`
                      : `${name} · ${count} cases · ~${formatDuration(perSuiteEstimate ?? 0)}`;
                  const chip = (
                    <Chip
                      variant="accent"
                      onRemove={() => toggleSuite(name)}
                    >
                      {label}
                    </Chip>
                  );
                  if (!meta?.description) {
                    return <span key={name}>{chip}</span>;
                  }
                  return (
                    <Tooltip key={name} content={meta.description}>
                      <span>{chip}</span>
                    </Tooltip>
                  );
                })}
              </div>
            </div>
          ) : null}

          {/* Unselected suites — grouped by category with inline descriptions */}
          {unselectedSuites.length > 0 ? (
            <div className="space-y-4">
              {CATEGORIES.map((cat) => {
                const suitesInCat = unselectedSuites
                  .filter((name) => categoryForSuite(name) === cat)
                  .sort();
                if (suitesInCat.length === 0) return null;
                return (
                  <div key={cat}>
                    <div className="flex items-baseline justify-between mb-1.5 border-b border-border pb-1">
                      <p className="text-xs uppercase tracking-wider text-fg-muted font-semibold">
                        {cat}
                      </p>
                      <p className="text-xxs text-fg-subtle">
                        {suitesInCat.length} suite
                        {suitesInCat.length === 1 ? "" : "s"}
                      </p>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-1.5">
                      {suitesInCat.map((name) => {
                        const meta = suiteMetaByName[name];
                        return (
                          <SuiteOptionButton
                            key={name}
                            name={name}
                            count={meta?.caseCount ?? null}
                            description={meta?.description ?? null}
                            modelCount={Math.max(1, selectedModelCount)}
                            repetitions={selections.repetitions}
                            concurrency={selections.concurrency}
                            onClick={() => toggleSuite(name)}
                          />
                        );
                      })}
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="text-xs text-fg-subtle italic">
              All available suites are selected.
            </p>
          )}

          {fieldError && fieldError.field === "suites" ? (
            <p
              role="alert"
              data-field-error="suites"
              className="mt-3 text-xs text-fail"
            >
              {fieldError.message}
            </p>
          ) : null}
        </Card>

        {/* ---------- Run parameters ---------- */}
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Sliders className="h-4 w-4 text-accent" aria-hidden="true" />
              <CardTitle>Run parameters</CardTitle>
            </div>
          </CardHeader>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
            <div data-field-block="repetitions">
              <Label htmlFor="repetitions">Repetitions</Label>
              <Input
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
              {fieldError && fieldError.field === "repetitions" ? (
                <p
                  role="alert"
                  data-field-error="repetitions"
                  className="mt-1 text-xs text-fail"
                >
                  {fieldError.message}
                </p>
              ) : null}
            </div>
            <div data-field-block="concurrency">
              <Label htmlFor="concurrency">Concurrency</Label>
              <Input
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
              {fieldError && fieldError.field === "concurrency" ? (
                <p
                  role="alert"
                  data-field-error="concurrency"
                  className="mt-1 text-xs text-fail"
                >
                  {fieldError.message}
                </p>
              ) : null}
            </div>
            <div data-field-block="tag_filter">
              <Label htmlFor="tag_filter">Tag filter</Label>
              <Input
                id="tag_filter"
                type="text"
                placeholder="science, math"
                data-field="tag_filter"
                value={selections.tagFilter}
                onChange={(e) =>
                  setSelections((prev) => ({
                    ...prev,
                    tagFilter: e.target.value,
                  }))
                }
              />
              {fieldError && fieldError.field === "tag_filter" ? (
                <p
                  role="alert"
                  data-field-error="tag_filter"
                  className="mt-1 text-xs text-fail"
                >
                  {fieldError.message}
                </p>
              ) : null}
            </div>
          </div>
        </Card>

        {/* ---------- Submit + totals ---------- */}
        <Card>
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
            <p
              data-testid="suite-totals"
              className="text-sm text-fg-muted"
            >
              {selections.suites.length === 0 ? (
                <>Select one or more suites to see the total.</>
              ) : (
                <>
                  Total:{" "}
                  <strong className="text-fg font-semibold num">
                    {selectedCasesTotal}
                  </strong>{" "}
                  case{selectedCasesTotal === 1 ? "" : "s"} across{" "}
                  <strong className="text-fg font-semibold num">
                    {selections.suites.length}
                  </strong>{" "}
                  suite{selections.suites.length === 1 ? "" : "s"} · rough est.{" "}
                  <strong className="text-fg font-semibold num">
                    ~{formatDuration(estimatedSeconds)}
                  </strong>
                  {selectedModelCount > 1 ? (
                    <> for {selectedModelCount} models</>
                  ) : null}
                  {selections.repetitions > 1 ? (
                    <>
                      {" "}
                      × {selections.repetitions} rep
                      {selections.repetitions === 1 ? "" : "s"}
                    </>
                  ) : null}
                  {anySuiteStillLoading ? " (refreshing…)" : ""}
                </>
              )}
            </p>
            <Button
              type="submit"
              variant="primary"
              disabled={submitMutation.isPending}
            >
              {submitMutation.isPending ? "Submitting…" : "Submit Run"}
              <ArrowRight className="h-4 w-4" aria-hidden="true" />
            </Button>
          </div>

          {fieldError && fieldError.field === null ? (
            <p
              role="alert"
              data-testid="form-error"
              className="mt-3 text-xs text-fail"
            >
              {fieldError.message}
            </p>
          ) : null}
        </Card>
      </form>
    </section>
  );
}

function totalCasesAcrossAllSuites(
  meta: Record<string, SuiteMeta>,
): number {
  return Object.values(meta).reduce(
    (sum, m) => sum + (m.caseCount ?? 0),
    0,
  );
}

/** Compact human-readable byte count (SI, base 1024). */
function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  return `${value.toFixed(value < 10 ? 1 : 0)} ${units[i]}`;
}

/**
 * Render the optional extended-metadata lines (quantisation level and
 * size on disk) for a model.
 *
 * The typed :class:`ModelInfo` from OpenAPI only declares ``name``,
 * ``digest``, and ``parameter_size`` because those are the fields the
 * Run_Report persists. The live ``GET /api/models`` endpoint returns
 * the richer shape straight from ``/api/tags`` (Ollama's inventory),
 * which includes ``size`` bytes and ``quantization_level``. We probe
 * for them at render time via :class:`Record<string, unknown>` casts
 * so the component tolerates both shapes without widening the shared
 * TypeScript types.
 */
function renderExtraModelInfo(model: ModelInfo): JSX.Element | null {
  const record = model as unknown as Record<string, unknown>;
  const quant = typeof record.quantization_level === "string"
    ? record.quantization_level
    : null;
  const size = typeof record.size === "number" ? record.size : null;
  if (!quant && !size) return null;
  return (
    <>
      {quant ? <p>Quantisation: {quant}</p> : null}
      {size ? <p>Size on disk: {formatBytes(size)}</p> : null}
    </>
  );
}

/**
 * Card-like button representing an unselected suite in the picker.
 *
 * Shows a compact summary by default; hovering reveals the full
 * backend description in a Radix tooltip. This keeps the picker
 * scannable while still making the methodology caveats visible when
 * users want them.
 *
 * * Primary line: suite name · case count · rough ETA
 * * Secondary line (inline): short summary from ``summaryForSuite``
 * * Tooltip: the verbose backend description (unchanged)
 */
interface SuiteOptionButtonProps {
  name: string;
  count: number | null;
  description: string | null;
  modelCount: number;
  repetitions: number;
  concurrency: number;
  onClick: () => void;
}

function SuiteOptionButton({
  name,
  count,
  description,
  modelCount,
  repetitions,
  concurrency,
  onClick,
}: SuiteOptionButtonProps): JSX.Element {
  const estimate =
    count != null
      ? estimateRunSeconds(modelCount, count, repetitions, concurrency)
      : null;
  const summary = summaryForSuite(name, description);

  const body = (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "group text-left w-full rounded-md border border-dashed border-border-strong",
        "bg-bg-alt/50 px-3 py-2 transition-colors",
        "hover:border-accent hover:bg-accent-soft/60 hover:border-solid",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/50",
      )}
    >
      <div className="flex items-center gap-1.5">
        <Plus
          className="h-3 w-3 text-fg-subtle group-hover:text-accent"
          aria-hidden="true"
        />
        <span className="text-sm font-medium text-fg">{name}</span>
        {count != null ? (
          <span className="text-xs text-fg-muted ml-auto num">
            {count} · ~{formatDuration(estimate ?? 0)}
          </span>
        ) : (
          <span className="text-xs text-fg-subtle ml-auto">loading…</span>
        )}
      </div>
      <p className="text-xs text-fg-muted mt-1 truncate leading-snug">
        {summary}
      </p>
    </button>
  );

  if (!description || description === summary) {
    return body;
  }
  return (
    <Tooltip content={description} side="top">
      {body}
    </Tooltip>
  );
}
