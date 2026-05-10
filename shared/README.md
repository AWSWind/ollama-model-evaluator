# `shared/`

Committed contract artifacts shared between the backend and the UI. These files
are generated from the backend's Pydantic models in a later task (see
`.kiro/specs/ollama-model-evaluator/tasks.md` task 17.4). They start as
placeholders so tooling can reference stable paths from day one.

- `openapi.yaml` — OpenAPI 3.1 specification for the Backend_API.
- `evaluation-suite.schema.json` — JSON Schema for Evaluation_Suite files.
- `run-report.schema.json` — JSON Schema for Run_Report artifacts.
