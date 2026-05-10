/**
 * User-facing error messages keyed by the Backend's ``ErrorCode``
 * enum (see ``backend/src/ollama_evaluator/api/errors.py``).
 *
 * The Backend's ``ErrorEnvelope.message`` is already human-readable;
 * these strings are the UI's *fallback* when we need a generic
 * message (e.g. the envelope is missing or the UI wants to prefix a
 * consistent sentence regardless of the Backend's wording).
 *
 * Requirement 15.4 asks the UI to display the Backend-returned message
 * for validation failures. Consumers should prefer
 * ``envelope.message`` and only fall back to this record when the
 * envelope did not carry a message.
 */
import type { ErrorCode } from "./errorTypes";

export const errorMessages: Record<ErrorCode, string> = {
  ollama_unreachable:
    "Could not reach the Ollama server. Check that it is running and that the configured base URL is correct.",
  model_not_found:
    "The requested Ollama model is not available on the server.",
  suite_not_found:
    "The requested Evaluation Suite could not be found.",
  run_not_found:
    "No Run with the requested id was found.",
  validation_failed:
    "The submitted configuration failed validation.",
  suite_invalid:
    "The Evaluation Suite file is invalid.",
  no_common_dimensions:
    "The two selected Runs share no common (model, suite) dimensions and cannot be compared.",
  dataset_fetch_failed:
    "Fetching the dataset from the HuggingFace Hub failed.",
  field_map_invalid:
    "The HuggingFace field map references a row field that is missing or has the wrong type.",
  run_timeout:
    "A Test Case execution exceeded the configured Ollama timeout.",
  run_error:
    "The Run could not complete because of an internal error.",
  metric_error:
    "A metric failed to score this Test Case.",
};
