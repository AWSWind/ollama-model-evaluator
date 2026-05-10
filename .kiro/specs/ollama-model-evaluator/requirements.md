# Requirements Document

## Introduction

The Ollama Model Evaluator is a system that evaluates large language models (LLMs) running locally on an Ollama server. It consists of two components delivered in order: a backend that orchestrates evaluations, scores responses, and persists results; and a web UI that lets users select models and datasets, monitor running benchmarks in real time, and browse current and historical results. The backend executes configurable evaluation suites against one or more Ollama models, measures both performance characteristics (latency, throughput, token counts) and response quality (correctness against expected outputs, scoring via configurable metrics), and produces structured reports. The system is designed to be run locally by developers who want to pick the best local model for a given task, track regressions as models are updated, and reproduce benchmark results.

## Glossary

- **Ollama_Server**: A locally running Ollama daemon exposing an HTTP API (default `http://localhost:11434`) that hosts one or more local LLMs.
- **Model**: A named LLM available on the Ollama_Server, identified by its Ollama model tag (for example `llama3:8b`, `mistral:7b-instruct`).
- **Evaluator**: The overall system defined by this specification, comprising the Backend and the UI.
- **Backend**: The server-side component that orchestrates Runs, calls the Ollama_Server, computes Metrics, persists results, and exposes the Backend_API.
- **Backend_API**: The HTTP API exposed by the Backend for use by the UI and by scripts.
- **UI**: The web-based user interface that consumes the Backend_API to let users configure Runs, monitor progress in real time, and browse results.
- **Dataset**: A named collection of Test_Cases used as input to a Run; in this specification an Evaluation_Suite is the concrete form of a Dataset.
- **Evaluation_Suite**: A named, versioned collection of Test_Cases grouped by a common purpose (for example `reasoning-basics`, `code-generation`).
- **Test_Case**: A single evaluation unit consisting of a prompt, optional system prompt, optional expected output, optional reference data, and metadata such as tags and scoring configuration.
- **Metric**: A named function that scores a Model response for a Test_Case and returns a numeric value in a defined range (for example exact-match, regex-match, JSON-schema-valid, BLEU, embedding-cosine-similarity, LLM-as-judge).
- **Run**: A single execution of the Backend against a defined set of Models and Evaluation_Suites, producing one Run_Report.
- **Run_Report**: A structured artifact produced by a Run that contains per-Test_Case results, per-Metric scores, aggregate statistics, and performance measurements.
- **Run_Event**: A discrete state-change message emitted by the Backend during a Run, such as `test-case-started`, `test-case-completed`, `run-progress`, or `run-completed`.
- **History_Store**: A persistent local data store maintained by the Backend that retains Run_Reports, Run_Events, and Evaluation_Suite metadata across restarts.
- **Performance_Metrics**: Quantitative measurements of a Model's runtime behavior, including time-to-first-token, total response time, tokens per second, and total tokens generated.
- **Judge_Model**: An LLM used as a Metric to score another Model's output (LLM-as-judge pattern).
- **Config_File**: A user-supplied file (YAML or JSON) that declares which Models, Evaluation_Suites, Metrics, and runtime options a Run should use.

## Requirements

### Requirement 1: Connect to Ollama Server

**User Story:** As a developer, I want the Backend to connect to a local Ollama_Server, so that I can evaluate models hosted on it.

#### Acceptance Criteria

1. THE Backend SHALL accept an Ollama_Server base URL via configuration, with a default of `http://localhost:11434`.
2. WHEN the Backend starts a Run, THE Backend SHALL verify connectivity to the Ollama_Server by calling its health or version endpoint.
3. IF the Ollama_Server is unreachable at Run start, THEN THE Backend SHALL abort the Run and record an error state on the Run with a message identifying the unreachable URL.
4. THE Backend SHALL support configuring a request timeout per Ollama_Server call, with a default of 120 seconds.
5. IF an individual Ollama_Server request exceeds the configured timeout, THEN THE Backend SHALL record the affected Test_Case as `timeout` and continue with the next Test_Case.

### Requirement 2: Discover and Select Models

**User Story:** As a developer, I want to list and select which Ollama models to evaluate, so that I can target specific Models in a Run.

#### Acceptance Criteria

1. THE Backend SHALL expose an operation that lists all Models currently available on the Ollama_Server.
2. THE Backend SHALL accept a list of Model names when starting a Run, sourced either from a Config_File or from a Backend_API request.
3. IF a requested Model is not available on the Ollama_Server, THEN THE Backend SHALL report the missing Model by name and abort the Run before executing any Test_Case.
4. WHERE the user sets a `pull-missing-models` option to true, THE Backend SHALL request the Ollama_Server to pull each missing Model before executing Test_Cases.
5. THE Backend SHALL record, for each evaluated Model, its Ollama model tag, digest, and parameter size as reported by the Ollama_Server.

### Requirement 3: Define Evaluation Suites and Test Cases

**User Story:** As a developer, I want to define Evaluation_Suites and Test_Cases in files, so that I can version and share evaluations.

#### Acceptance Criteria

1. THE Backend SHALL load Evaluation_Suites from files in a user-specified directory.
2. THE Backend SHALL support Evaluation_Suite files in YAML and JSON formats.
3. THE Backend SHALL require each Test_Case to declare a unique identifier within its Evaluation_Suite, a prompt, and at least one Metric.
4. THE Backend SHALL support optional Test_Case fields including `system_prompt`, `expected_output`, `reference_data`, `tags`, `temperature`, `max_tokens`, and `stop_sequences`.
5. IF a Test_Case is missing a required field, THEN THE Backend SHALL report the Evaluation_Suite file path, the Test_Case identifier if available, and the missing field, and abort the Run.
6. THE Backend SHALL allow filtering Test_Cases by Evaluation_Suite name and by tag at Run time.

### Requirement 4: Parse and Serialize Evaluation Suites

**User Story:** As a developer, I want the Backend to reliably parse and serialize Evaluation_Suite files, so that authored suites round-trip without data loss.

#### Acceptance Criteria

1. THE Backend SHALL parse YAML and JSON Evaluation_Suite files into an internal Evaluation_Suite object.
2. THE Backend SHALL provide a pretty printer that serializes an internal Evaluation_Suite object back into YAML and into JSON.
3. FOR ALL valid Evaluation_Suite files, parsing then pretty-printing then parsing SHALL produce an equivalent internal Evaluation_Suite object (round-trip property).
4. IF an Evaluation_Suite file contains invalid syntax, THEN THE Backend SHALL return an error that includes the file path, line number, and a human-readable description of the syntax error.

### Requirement 5: Execute Test Cases Against Models

**User Story:** As a developer, I want the Backend to run each Test_Case against each selected Model, so that I can collect responses for scoring.

#### Acceptance Criteria

1. WHEN a Run begins, THE Backend SHALL execute every selected Test_Case against every selected Model exactly once per configured repetition.
2. THE Backend SHALL accept a `repetitions` configuration value, defaulting to 1, that specifies how many times each Test_Case is executed per Model.
3. THE Backend SHALL apply Test_Case-level generation parameters (`temperature`, `max_tokens`, `stop_sequences`) when calling the Ollama_Server.
4. WHERE a Test_Case does not specify a generation parameter, THE Backend SHALL use the Run-level default for that parameter.
5. THE Backend SHALL support a `concurrency` configuration value, defaulting to 1, that limits the number of in-flight Ollama_Server requests.
6. IF a Test_Case execution returns an error from the Ollama_Server, THEN THE Backend SHALL record the error message and status on the Test_Case result and continue with the next Test_Case.

### Requirement 6: Measure Performance Metrics

**User Story:** As a developer, I want the Backend to measure Performance_Metrics for each Model response, so that I can compare Models on speed and resource usage.

#### Acceptance Criteria

1. THE Backend SHALL record, for each Test_Case execution, the wall-clock time-to-first-token in milliseconds.
2. THE Backend SHALL record, for each Test_Case execution, the total response time in milliseconds.
3. THE Backend SHALL record, for each Test_Case execution, the number of prompt tokens and the number of response tokens reported by the Ollama_Server.
4. THE Backend SHALL compute, for each Test_Case execution, tokens per second as response tokens divided by total response time in seconds.
5. WHERE Ollama_Server response metadata does not include a token count, THE Backend SHALL record the token count field as `null` and SHALL NOT fail the Test_Case.

### Requirement 7: Score Responses with Metrics

**User Story:** As a developer, I want to score Model responses using configurable Metrics, so that I can quantify response quality.

#### Acceptance Criteria

1. THE Backend SHALL provide built-in Metrics including `exact-match`, `regex-match`, `contains`, `json-schema-valid`, and `length-range`.
2. THE Backend SHALL support an `llm-as-judge` Metric that uses a configured Judge_Model on the Ollama_Server to score responses against a rubric defined in the Test_Case.
3. THE Backend SHALL compute each configured Metric for every Test_Case execution and record the resulting score on the Test_Case result.
4. THE Backend SHALL require each Metric to return a numeric score and a pass or fail classification based on a Metric-specific threshold.
5. IF a Metric raises an error while scoring a response, THEN THE Backend SHALL record the error on the Test_Case result, mark the Metric as `error`, and continue scoring the remaining Metrics.
6. WHERE `repetitions` is greater than 1, THE Backend SHALL record per-repetition Metric scores and SHALL compute mean and standard deviation of each Metric across repetitions for the same Test_Case and Model pair.

### Requirement 8: Produce Run Reports

**User Story:** As a developer, I want the Backend to produce a structured Run_Report, so that I can review and share evaluation results.

#### Acceptance Criteria

1. WHEN a Run completes, THE Backend SHALL write a Run_Report to the configured output directory and SHALL persist the Run_Report in the History_Store.
2. THE Backend SHALL produce a JSON Run_Report containing Run metadata, per-Test_Case results, per-Metric scores, Performance_Metrics, and aggregate statistics per Model.
3. THE Backend SHALL produce a Markdown Run_Report that summarizes aggregate statistics per Model and per Evaluation_Suite in human-readable tables.
4. THE Backend SHALL include in the Run_Report the Run identifier, start time, end time, Backend version, Ollama_Server version, Config_File contents, and the Ollama model tag and digest for each evaluated Model.
5. FOR ALL valid Run_Report JSON files produced by the Backend, parsing then re-serializing SHALL produce an equivalent Run_Report object (round-trip property).

### Requirement 9: Compare Runs

**User Story:** As a developer, I want to compare two Run_Reports, so that I can detect regressions and improvements across Models or over time.

#### Acceptance Criteria

1. THE Backend SHALL provide an operation that accepts two Run_Report identifiers and produces a Comparison_Report.
2. THE Comparison_Report SHALL list, for each Model and Metric pair present in both Run_Reports, the mean score in each Run and the signed difference.
3. THE Comparison_Report SHALL list, for each Model, the difference in mean tokens per second and mean total response time between the two Runs.
4. IF the two Run_Reports do not share any Model, Evaluation_Suite, or Metric, THEN THE Backend SHALL return an error that identifies the mismatch.

### Requirement 10: Provide a Command-Line Interface

**User Story:** As a developer, I want to drive the Backend from the command line, so that I can integrate it into scripts and CI.

#### Acceptance Criteria

1. THE Backend SHALL expose subcommands for `list-models`, `run`, `compare`, `validate-suite`, and `serve`.
2. WHEN the `run` subcommand completes with at least one Test_Case whose pass or fail classification is fail, THE Backend SHALL return a non-zero exit code.
3. WHEN the `run` subcommand completes with every Test_Case classified as pass, THE Backend SHALL return exit code 0.
4. THE Backend SHALL support a `--config` flag that accepts a path to a Config_File.
5. THE Backend SHALL support a `--output-dir` flag that overrides the Config_File output directory.
6. THE Backend SHALL support a `--log-level` flag that accepts one of `debug`, `info`, `warn`, or `error`, with a default of `info`.
7. WHEN the `serve` subcommand is invoked, THE Backend SHALL start the Backend_API and block until the Backend receives a termination signal.

### Requirement 11: Handle Errors and Partial Failures

**User Story:** As a developer, I want the Backend to handle errors gracefully, so that a single failure does not lose the entire Run.

#### Acceptance Criteria

1. IF a Test_Case execution fails due to a network error, THEN THE Backend SHALL retry the request up to a configurable maximum, defaulting to 2 additional attempts, with exponential backoff starting at 1 second.
2. IF all retry attempts for a Test_Case execution fail, THEN THE Backend SHALL record the Test_Case result as `error` with the last error message and continue the Run.
3. WHEN a Run ends with any Test_Case in `error` or `timeout` state, THE Backend SHALL include a summary section in the Run_Report that lists each such Test_Case with its Model and error message.
4. IF the Backend receives a SIGINT or SIGTERM signal during a Run, THEN THE Backend SHALL stop dispatching new Test_Case executions, wait for in-flight executions up to 30 seconds, and write a partial Run_Report.

### Requirement 12: Persist Run History

**User Story:** As a developer, I want Run_Reports to persist across Backend restarts, so that I can browse and compare historical results in the UI.

#### Acceptance Criteria

1. THE Backend SHALL maintain a History_Store on local disk that persists Run_Reports, Run_Events, and indexed Evaluation_Suite metadata across Backend restarts.
2. WHEN a Run completes or is aborted, THE Backend SHALL write the final Run_Report and its terminal Run_Events to the History_Store before marking the Run as `completed` or `aborted`.
3. THE Backend SHALL assign each Run a globally unique Run identifier at Run creation time and SHALL use that identifier as the primary key in the History_Store.
4. THE Backend SHALL expose an operation to list Run identifiers with filters for Model name, Evaluation_Suite name, status, and a time range.
5. THE Backend SHALL expose an operation to delete a Run from the History_Store by Run identifier.
6. IF a History_Store read operation fails due to a corrupted record, THEN THE Backend SHALL skip the corrupted record, log a warning that identifies the Run identifier, and continue returning the remaining records.

### Requirement 13: Expose a Backend API

**User Story:** As a UI developer, I want the Backend to expose a stable HTTP API, so that the UI and other clients can drive evaluations programmatically.

#### Acceptance Criteria

1. THE Backend SHALL expose a Backend_API over HTTP on a configurable host and port, defaulting to `127.0.0.1:8765`.
2. THE Backend_API SHALL provide endpoints to list available Models, list available Evaluation_Suites, list Run identifiers, retrieve a Run_Report by identifier, and retrieve a Comparison_Report for two Run identifiers.
3. THE Backend_API SHALL provide an endpoint that creates a new Run from a submitted Run configuration and returns the new Run identifier and initial status.
4. THE Backend_API SHALL provide an endpoint that cancels an in-progress Run by Run identifier.
5. IF a Backend_API request references a Run identifier, Evaluation_Suite name, or Model name that does not exist, THEN THE Backend SHALL respond with HTTP status 404 and a JSON error body that includes an `error_code` and `message`.
6. IF a Backend_API request body fails schema validation, THEN THE Backend SHALL respond with HTTP status 400 and a JSON error body that identifies the first invalid field and the validation rule that failed.
7. THE Backend_API SHALL document its request and response schemas in a machine-readable specification (for example OpenAPI) and SHALL serve that specification from a well-known endpoint.

### Requirement 14: Stream Real-Time Run Events

**User Story:** As a user, I want to see Run progress updates in real time, so that I can monitor long-running benchmarks from the UI.

#### Acceptance Criteria

1. THE Backend_API SHALL expose a streaming endpoint that emits Run_Events for a specified Run identifier.
2. THE Backend SHALL emit a `run-started` event when a Run transitions to the `running` state.
3. WHILE a Run is in the `running` state, THE Backend SHALL emit a `test-case-completed` event after each Test_Case execution finishes, including the Test_Case identifier, Model name, Metric scores, and Performance_Metrics for that execution.
4. WHILE a Run is in the `running` state, THE Backend SHALL emit a `run-progress` event at least once every 2 seconds that includes the counts of completed, in-progress, and pending Test_Case executions.
5. THE Backend SHALL emit a terminal event of type `run-completed`, `run-aborted`, or `run-failed` exactly once per Run.
6. IF a streaming client subscribes after the Run has already emitted events, THEN THE Backend SHALL replay prior Run_Events for that Run identifier in order before emitting new events.
7. IF a streaming client disconnects, THEN THE Backend SHALL continue executing the Run and SHALL NOT lose Run_Events for other subscribers.

### Requirement 15: Provide a Web UI

**User Story:** As a user, I want a web UI that lets me configure, monitor, and review evaluations, so that I can use the Evaluator without writing config files or scripts.

#### Acceptance Criteria

1. THE UI SHALL be delivered as a web application that runs in current versions of Chrome, Firefox, and Safari.
2. WHEN the UI loads, THE UI SHALL fetch the list of available Models and the list of available Evaluation_Suites from the Backend_API.
3. THE UI SHALL provide a Run-configuration view that allows the user to select one or more Models, select one or more Evaluation_Suites, set `repetitions` and `concurrency`, and submit the configuration to start a new Run.
4. IF the user submits a Run-configuration that the Backend_API rejects with HTTP 400, THEN THE UI SHALL display the returned error message next to the offending field.
5. WHEN a new Run is submitted successfully, THE UI SHALL navigate to a Run-detail view for that Run identifier.
6. THE UI SHALL consume the Run_Events streaming endpoint for the displayed Run and SHALL update the Run-detail view with each `run-progress`, `test-case-completed`, and terminal event.
7. THE UI SHALL display, in the Run-detail view, at least: current status, percentage complete, counts of passed, failed, `error`, and `timeout` Test_Cases, and a live-updating table of completed Test_Case executions with their Metric scores and Performance_Metrics.
8. IF the streaming connection to the Backend_API is lost, THEN THE UI SHALL display a visible "disconnected" indicator and SHALL attempt to reconnect up to 5 times with exponential backoff starting at 1 second.
9. WHERE the streaming connection cannot be re-established after the retry limit, THE UI SHALL fall back to polling the Backend_API for Run status at a 5-second interval.
10. THE UI SHALL allow the user to cancel an in-progress Run from the Run-detail view; WHEN the user confirms cancellation, THE UI SHALL call the cancel endpoint and update the displayed status.

### Requirement 16: Browse History and Results in the UI

**User Story:** As a user, I want to browse historical Run_Reports in the UI, so that I can review past evaluations and compare models over time.

#### Acceptance Criteria

1. THE UI SHALL provide a history view that lists Run identifiers retrieved from the Backend_API, including each Run's start time, status, Model names, Evaluation_Suite names, and aggregate pass rate.
2. THE UI SHALL allow the user to filter the history view by Model name, Evaluation_Suite name, status, and a time range.
3. WHEN the user selects a completed Run in the history view, THE UI SHALL open a Run-detail view that displays the full Run_Report, including per-Test_Case inputs, outputs, Metric scores, and Performance_Metrics.
4. THE UI SHALL allow the user to select two Runs from the history view and request a Comparison_Report, and SHALL render the Comparison_Report as a table of per-Model and per-Metric differences.
5. THE UI SHALL allow the user to download the JSON Run_Report and the Markdown Run_Report for any completed Run.
6. IF a Run is in the `error` or `aborted` state, THEN THE UI SHALL display the terminal event's error message in the Run-detail view.

### Requirement 17: Source Evaluation Suites from Public Benchmarks and HuggingFace Datasets

**User Story:** As a developer, I want to evaluate models against well-known public benchmarks and arbitrary HuggingFace datasets without hand-authoring every Test_Case, so that I can produce reproducible, comparable results with minimal per-benchmark boilerplate.

#### Acceptance Criteria

1. THE Backend SHALL provide built-in adapters for the MMLU, HellaSwag, TruthfulQA (MC1 form), GSM8K, and HumanEval benchmarks that transform their canonical rows into Evaluation_Suite objects.
2. THE Backend SHALL provide a generic HuggingFace adapter that loads any HuggingFace dataset referenced by `repo_id[:config][:split]` and maps its rows into Test_Cases via a user-supplied field map that declares at least a prompt field and optionally expected-output, system-prompt, choices, and tag-source fields.
3. THE Backend SHALL accept a `dataset_mode` setting with values `local` and `remote`, defaulting to `local`, that controls whether adapter-backed Evaluation_Suites are read from disk or streamed from the HuggingFace Hub at Run time.
4. THE Backend SHALL allow each adapter-backed suite file to override the `dataset_mode` setting for that suite only.
5. WHEN `dataset_mode` is `remote`, THE Backend SHALL materialise every remote-mode Evaluation_Suite during preflight, before emitting the `run-started` event.
6. IF materialising a remote-mode Evaluation_Suite fails due to a network, authentication, or dataset-availability error, THEN THE Backend SHALL abort the Run with a `run-failed` event whose `error_code` is `dataset_fetch_failed` and whose message identifies the HuggingFace reference and the underlying cause.
7. IF a HuggingFace field map references a row field that is missing, `null`, or the wrong type in any row, THEN THE Backend SHALL abort the Run with a `run-failed` event whose `error_code` is `field_map_invalid` and whose message identifies the row index and the offending field path.
8. THE Backend SHALL expose CLI subcommands that convert each supported benchmark (`mmlu`, `hellaswag`, `truthfulqa`, `gsm8k`, `humaneval`) and an arbitrary HuggingFace reference (`hf`) into on-disk Evaluation_Suite files that can subsequently be loaded in `local` mode.
9. FOR THE HumanEval adapter, THE Backend SHALL in v1 record each Model response verbatim using a `response-capture` metric that always classifies the Test_Case as `pass`, and SHALL NOT execute Model-generated code; the Backend SHALL reserve the metric name `humaneval-exec` for a future sandboxed-execution implementation.
10. FOR ALL `(adapter, dataset, split)` triples, the set of Test_Cases produced by `local` mode and by `remote` mode SHALL be equivalent: same ordered Test_Case ids, with per-Test_Case `prompt`, `system_prompt`, `expected_output`, `reference_data`, `tags`, and `metrics` equal.
