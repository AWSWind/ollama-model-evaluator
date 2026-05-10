# ollama-evaluator-ui

Vite + React + TypeScript UI for the Ollama Model Evaluator. See
`../.kiro/specs/ollama-model-evaluator/` for the full design.

## Layout

```
ui/
├── package.json
├── tsconfig.json
├── vite.config.ts
├── index.html
└── src/
    ├── main.tsx
    ├── api/      # typed API client (generated)
    ├── routes/   # NewRun, RunDetail, History, Compare
    ├── stream/   # reconnecting WebSocket + polling fallback
    └── state/    # Zustand stores + TanStack Query hooks
```

## Development (not installed yet)

Dependencies are declared in `package.json`. This scaffold does **not** run
`npm install`; that is deferred to a later task when the UI routes are
implemented.
