/**
 * Error-related types for the Backend HTTP API.
 *
 * The Backend's :class:`ErrorEnvelope` and :class:`ErrorCode` are not
 * included in the generated OpenAPI operations (they are used only on
 * non-2xx responses and FastAPI does not put them in the schema by
 * default). We mirror the Pydantic enum in
 * ``backend/src/ollama_evaluator/api/errors.py`` here as a TypeScript
 * string union so callers get compile-time checks when branching on
 * ``envelope.error_code``.
 */

export type ErrorCode =
  | "ollama_unreachable"
  | "model_not_found"
  | "suite_not_found"
  | "run_not_found"
  | "validation_failed"
  | "suite_invalid"
  | "no_common_dimensions"
  | "dataset_fetch_failed"
  | "field_map_invalid"
  | "run_timeout"
  | "run_error"
  | "metric_error";

export interface ErrorEnvelope {
  error_code: ErrorCode;
  message: string;
  field?: string | null;
}
